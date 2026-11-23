"""The S2-18 eval run: answer rows, rates and cost per phase, `make judge` on saved answers,
and the committed run file and report.

The first tests use invented questions, chunks and answers (D901, as in tests/test_judges.py)
and a fake client; the last ones check the committed run in evals/results/runs/.
"""

import hashlib
import importlib.util
import json
import re

import pytest

from evals.answer_eval import (
    END,
    START,
    answer_row,
    markdown,
    report_section,
    score_dict,
    score_from_dict,
    summarize,
)
from evals.judges import FAITHFULNESS, QUALITY, claim_verdicts, quality_verdict, shown_claims
from evals.scorers import score_question
from factory_docs.answer import answer_question
from factory_docs.config import ROOT, load_config
from factory_docs.index import Hit, load_chunks
from factory_docs.test_set import TEST_SET, load_test_set
from tests.test_judges import (
    ANSWERABLE,
    CHUNKS,
    SENT,
    UNANSWERABLE,
    FakeClient,
    faithfulness_reply,
    quality_reply,
)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_answers", ROOT / "scripts" / "eval_answers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config():
    return load_config()


def answer_reply() -> str:
    return json.dumps(
        {
            "status": "answered",
            "claims": [
                {
                    "text": "The flash point is 23 °C.",
                    "citations": [{"chunk_id": "D901-s-0001", "quote": "Flash point: 23 °C"}],
                }
            ],
            "missing": None,
        }
    )


def test_answer_row_keeps_what_was_shown_sent_and_paid(config):
    hits = [Hit(1.0 - i / 10, CHUNKS[c], first_stage_rank=i + 1) for i, c in enumerate(SENT)]
    record = answer_question(
        ANSWERABLE.question, hits, FakeClient(answer_reply(), cost=0.001), config, 0.0, {}
    )
    row = answer_row(ANSWERABLE, record, "results/answers/x.json")
    assert (row["id"], row["category"], row["language"]) == ("T90", "answerable", "en")
    assert row["sent"] == SENT and row["dropped"] == []
    assert row["final_status"] == "valid"
    assert row["answer"]["claims"][0]["text"] == "The flash point is 23 °C."
    assert (row["cost_usd"], row["prompt_tokens"], row["completion_tokens"]) == (0.001, 1200, 90)
    assert row["top_score"] == 1.0 and row["errors"] == []


def test_a_gated_answer_row_has_no_answer_and_no_cost(config):
    hits = [Hit(-5.0, CHUNKS["D901-s-0001"], first_stage_rank=1)]
    client = FakeClient()
    record = answer_question(UNANSWERABLE.question, hits, client, config, 0.0, {})
    row = answer_row(UNANSWERABLE, record, "results/answers/y.json")
    assert client.calls == []
    assert (row["final_status"], row["answer"], row["cost_usd"]) == ("below_threshold", None, 0)
    assert row["total_s"] is None and row["errors"]


def test_scores_survive_the_json_round_trip():
    row = {"final_status": "valid", "answer": json.loads(answer_reply())}
    score = score_question(ANSWERABLE, row, CHUNKS)
    assert score_from_dict(json.loads(json.dumps(score_dict(score)))) == score


# --- an invented run: rates and cost by hand -----------------------------------------------


def run_row(qid, category, language, passed, quality, faithfulness, cost, called=True):
    return {
        "id": qid,
        "category": category,
        "language": language,
        "total_s": 1.0 if called else None,
        "cost_usd": cost,
        "prompt_tokens": 1000 if called else 0,
        "completion_tokens": 100 if called else 0,
        "score": {
            "question_id": qid,
            "category": category,
            "language": language,
            "outcome": "answered" if called else "below_threshold",
            "refused": not called,
            "refusal_correct": passed,
            "right_source": None if category == "unanswerable" else passed,
            "evidence_cited": None if category == "unanswerable" else passed,
            "wrong_source": None,
            "facts": [] if category == "unanswerable" else [passed],
            "forbidden": [],
            "passed": passed,
        },
        QUALITY: quality,
        FAITHFULNESS: faithfulness,
    }


def judgement(status, output=None, cost=0.0, tokens=(0, 0)):
    return {
        "status": status,
        "output": output,
        "errors": [],
        "cost_usd": cost,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "record": None,
    }


def verdict(v, cost=0.0001):
    return judgement("valid", {"reasoning": "r", "verdict": v}, cost, (500, 50))


def supports(*vs, cost=0.0003):
    claims = [{"claim": i, "reasoning": "r", "verdict": v} for i, v in enumerate(vs, 1)]
    return judgement("valid", {"claims": claims}, cost, (3000, 80))


RUN = {
    "answers_from": None,
    "questions": [
        run_row("T01", "answerable", "en", True, verdict("correct"), supports("supported"), 0.001),
        run_row(
            "T02",
            "answerable",
            "de",
            False,
            verdict("partially_correct"),
            supports("supported", "not_supported"),
            0.002,
        ),
        run_row(
            "T03",
            "unanswerable",
            "en",
            True,
            verdict("correct"),
            judgement("no_claims"),
            0.0,
            called=False,
        ),
    ],
}


def test_summary_rates_by_hand():
    s = summarize(RUN)
    assert s["overall"]["questions"] == 3
    assert s["overall"]["deterministic"]["passed"]["n"] == 2  # T01, T03
    assert s["overall"]["quality"]["correct"]["n"] == 2  # T01, T03
    # claims: T01 1 supported; T02 1 supported, 1 not -> 2 of 3 supported
    assert s["overall"]["faithfulness"]["claims_supported"]["n"] == 2
    assert s["overall"]["faithfulness"]["claims_supported"]["total"] == 3
    assert s["by_category"]["answerable"]["quality"]["partially_correct"]["n"] == 1
    assert list(s["by_category"]) == ["answerable", "unanswerable"]
    assert s["by_language"]["de"]["deterministic"]["passed"]["n"] == 0
    assert s["agreement"]["failed"]["partially_correct"] == 1


def test_cost_per_phase_by_hand():
    cost = summarize(RUN)["cost"]
    # answers: 2 calls (T03 gated), $0.003, 2000 / 200 tokens; 1000 x 0.003 / 3 questions = $1
    assert cost["answers"]["calls"] == 2
    assert cost["answers"]["usd"] == pytest.approx(0.003)
    assert (cost["answers"]["prompt_tokens"], cost["answers"]["completion_tokens"]) == (2000, 200)
    assert cost["answers"]["usd_per_1000"] == pytest.approx(1.0)
    # quality: 3 calls x $0.0001; faithfulness: 2 calls x $0.0003 (T03 has no claims)
    assert (cost[QUALITY]["calls"], cost[FAITHFULNESS]["calls"]) == (3, 2)
    assert cost[QUALITY]["usd"] == pytest.approx(0.0003)
    assert cost[FAITHFULNESS]["usd"] == pytest.approx(0.0006)
    assert cost["judges_usd"] == pytest.approx(0.0009)
    assert cost["judges_usd_per_1000"] == pytest.approx(0.3)
    assert cost["spent_this_run_usd"] == pytest.approx(0.0039)
    # a re-judged run spent nothing on answers
    rejudged = summarize({**RUN, "answers_from": "evals/results/runs/x.json"})["cost"]
    assert rejudged["spent_this_run_usd"] == pytest.approx(0.0009)


# --- make judge: the judges only, on saved answers -----------------------------------------


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """A saved run of two invented questions in a tmp repo root, and a patched script."""
    script = load_script()
    chunk_file = tmp_path / "chunks.jsonl"
    chunk_file.write_text(
        "".join(c.model_dump_json() + "\n" for c in CHUNKS.values()), encoding="utf-8"
    )
    test_file = tmp_path / "test_set.toml"
    test_file.write_text("# invented\n", encoding="utf-8")
    answered = {"final_status": "valid", "answer": json.loads(answer_reply())}
    refused = {"final_status": "below_threshold", "answer": None}
    saved = {
        "created_at": "2026-11-23T00:00:00+00:00",
        "machine": {"platform": "test"},
        "git": {"commit": "abc", "dirty": False},
        "answers_from": None,
        "test_set": {"path": "test_set.toml", "sha256": script.sha256(test_file)},
        "chunks": {"path": "chunks.jsonl", "sha256": script.sha256(chunk_file)},
        "settings": {"answer": {"model": "haiku-5.5"}, "judge": {"old": True}},
        "questions": [
            {"id": "T90", "sent": SENT, **answered, "score": "old", QUALITY: "old"},
            {"id": "T91", "sent": SENT[:1], **refused, FAITHFULNESS: "old"},
        ],
    }
    (tmp_path / "runs").mkdir()
    source = tmp_path / "runs" / "20261123T000000Z.json"
    source.write_text(json.dumps(saved), encoding="utf-8")

    clients = []

    def make_client(model_key, config):
        # one quality call each, and one faithfulness call for T90's single claim
        client = FakeClient(quality_reply(), faithfulness_reply("supported"), quality_reply())
        clients.append((model_key, client))
        return client

    monkeypatch.setattr(FakeClient, "close", lambda self: None, raising=False)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(script, "TEST_SET", test_file)
    monkeypatch.setattr(script, "JUDGEMENTS", tmp_path / "judgements")
    monkeypatch.setattr(script, "load_test_set", lambda: [ANSWERABLE, UNANSWERABLE])
    monkeypatch.setattr(script, "load_env", lambda: None)
    monkeypatch.setattr(script, "spent_on_answers", lambda: 0.0)
    monkeypatch.setattr(script, "make_client", make_client)
    return script, source, saved, clients, tmp_path


def test_make_judge_judges_the_saved_answers_without_answering_again(saved_run, config):
    script, source, saved, clients, root = saved_run
    path, run = script.run_judges(config, source)
    assert path != source and path.parent == root / "runs"
    assert run["answers_from"] == "runs/20261123T000000Z.json"
    assert [model for model, _ in clients] == [config.judge.model]  # no answer client
    [(_, client)] = clients
    assert [c["schema_name"] for c in client.calls] == [
        "judge_quality",
        "judge_faithfulness",
        "judge_quality",
    ]  # T91 was refused: no claims, no faithfulness call
    for old, new in zip(saved["questions"], run["questions"], strict=True):
        assert (new["answer"], new["final_status"], new["sent"]) == (
            old["answer"],
            old["final_status"],
            old["sent"],
        )
    t90, t91 = run["questions"]
    assert t90["score"]["passed"] is True and t91["score"]["passed"] is True
    assert quality_verdict(t90[QUALITY]) == "correct"
    assert claim_verdicts(t90[FAITHFULNESS]) == ["supported"]
    assert t91[FAITHFULNESS]["status"] == "no_claims" and t91[FAITHFULNESS]["record"] is None
    assert run["settings"]["judge"]["prompts"][QUALITY]["name"] == "judge_quality"
    assert len(list((root / "judgements").glob("*.json"))) == 3  # one log per judge call
    assert json.loads(path.read_text(encoding="utf-8")) == run
    assert json.loads(source.read_text(encoding="utf-8")) == saved  # the old run is untouched


def test_make_judge_refuses_when_the_chunks_changed(saved_run, config):
    script, source, _, clients, root = saved_run
    (root / "chunks.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed since"):
        script.run_judges(config, source)
    assert clients == []


def test_make_judge_refuses_when_the_test_set_changed(saved_run, config):
    script, source, _, clients, root = saved_run
    (root / "test_set.toml").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="test_set.toml changed"):
        script.run_judges(config, source)
    assert clients == []


# --- the committed run ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed():
    runs = sorted((ROOT / "evals" / "results" / "runs").glob("*.json"))
    assert runs, "no committed run file: run `make eval`"
    path = runs[-1]
    return str(path.relative_to(ROOT)), json.loads(path.read_text(encoding="utf-8"))


def test_committed_report_regenerates_from_the_latest_run(committed):
    path, run = committed
    md = (ROOT / "evals" / "results" / "eval.md").read_text(encoding="utf-8")
    assert md == markdown(run, path)
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    section = re.search(re.escape(START) + r"\n(.*?)" + re.escape(END), report, re.S).group(1)
    assert section.strip() == report_section(run, path).strip()


def test_committed_run_judged_every_question_and_every_claim(committed):
    _, run = committed
    questions = load_test_set()
    assert [r["id"] for r in run["questions"]] == [q.id for q in questions]
    for row in run["questions"]:
        assert row[QUALITY]["status"] in ("valid", "invalid", "error")
        n_claims = len(shown_claims(row))
        verdicts = claim_verdicts(row[FAITHFULNESS])
        if n_claims == 0:
            assert verdicts == []
        elif verdicts is not None:
            assert len(verdicts) == n_claims


def test_committed_scores_rederive_from_the_saved_answers(committed):
    _, run = committed
    by_id = {q.id: q for q in load_test_set()}
    chunks = {c.chunk_id: c for c in load_chunks(ROOT / run["chunks"]["path"])}
    for row in run["questions"]:
        assert row["score"] == score_dict(score_question(by_id[row["id"]], row, chunks))


def test_committed_run_matches_the_test_set_and_logs_cost(committed):
    _, run = committed
    assert run["test_set"]["sha256"] == hashlib.sha256(TEST_SET.read_bytes()).hexdigest()
    cost = summarize(run)["cost"]
    assert cost[QUALITY]["calls"] == len(run["questions"])
    assert cost[QUALITY]["usd"] > 0 and cost[FAITHFULNESS]["usd"] > 0
    for row in run["questions"]:
        for key in (QUALITY, FAITHFULNESS):
            if row[key]["status"] != "no_claims":
                assert row[key]["prompt_tokens"] > 0
