"""S2-19 judge calibration: agreement math, hand grades, the blind sheet and the script.

Questions and answers are invented (from tests/test_judges.py); every expected number is
counted by hand in the comments. No model is called: a fake client returns the judge replies.
"""

import hashlib
import importlib.util
import json
import re

import pytest

from evals.calibration import (
    END,
    NO_MAJORITY,
    START,
    Grade,
    cohen_kappa,
    confusion,
    direction,
    grade_problems,
    grade_spread_line,
    grades_template,
    load_grades,
    majority,
    markdown,
    report_section,
    rubric,
    sheet_markdown,
    summarize,
)
from evals.judges import QUALITY, render_answer
from factory_docs.config import ROOT, load_config
from factory_docs.prompts import Prompt, load_prompt
from factory_docs.test_set import load_test_set
from tests.test_judges import (
    ANSWERABLE,
    ANSWERED,
    CHUNKS,
    GATED,
    SENT,
    UNANSWERABLE,
    FakeClient,
    quality_reply,
)

C, P, I = "correct", "partially_correct", "incorrect"  # noqa: E741


@pytest.fixture(scope="module")
def config():
    return load_config()


# --- agreement math ------------------------------------------------------------------------


def test_majority_needs_more_than_half_of_the_passes():
    assert majority([C, C, I]) == C
    assert majority([P, I, P]) == P
    assert majority([C, C, C]) == C
    assert majority([C, P, I]) == NO_MAJORITY  # all three differ


def test_cohen_kappa_matches_a_hand_computed_value():
    human = [C, C, C, P, I, I]
    judge = [C, C, P, P, I, C]
    # observed 4/6; human C3 P1 I2, judge C3 P2 I1: chance (9 + 2 + 2) / 36 = 13/36
    # kappa = (24/36 - 13/36) / (1 - 13/36) = 11/23
    assert cohen_kappa(human, judge) == pytest.approx(11 / 23)
    assert cohen_kappa(human, human) == pytest.approx(1.0)
    assert cohen_kappa([C, C], [C, C]) is None  # chance agreement 1: kappa undefined
    assert cohen_kappa([], []) is None


def test_direction_counts_lenient_strict_and_no_verdict():
    human = [C, P, I, C, P]
    judge = [P, C, C, NO_MAJORITY, P]
    # C->P strict; P->C and I->C lenient; C->no majority: no verdict; P=P agrees
    assert direction(human, judge) == {
        "disagreements": 4,
        "lenient": 2,
        "strict": 1,
        "no_verdict": 1,
    }


def test_confusion_counts_hand_grade_against_judge():
    table = confusion([C, C, P, I], [C, P, P, NO_MAJORITY])
    assert table[C] == {C: 1, P: 1, I: 0, "not_judged": 0, NO_MAJORITY: 0}
    assert table[P][P] == 1 and table[I][NO_MAJORITY] == 1
    assert sum(sum(row.values()) for row in table.values()) == 4


def judged(verdict: str, status: str = "valid") -> dict:
    """A judgement as the results file keeps it ($0.0001, 100 + 10 tokens)."""
    return {
        "status": status,
        "output": {"reasoning": f"Reason for {verdict}.", "verdict": verdict}
        if status == "valid"
        else None,
        "errors": [] if status == "valid" else ["not JSON"],
        "cost_usd": 0.0001,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "record": "results/judgements/x.json",
    }


def row(id_, human, passes, has_claims=True, category="answerable"):
    return {
        "id": id_,
        "question": f"Question {id_}?",
        "category": category,
        "language": "en",
        "outcome": "answered" if has_claims else "not_in_documents",
        "has_claims": has_claims,
        "human": {"verdict": human, "note": ""},
        "judge": [judged(v) for v in passes],
    }


RESULT = {
    "questions": [
        row("T90", C, [C, C, C]),  # agrees
        row("T91", P, [C, C, P]),  # majority C: lenient
        row("T92", C, [C, C, C], has_claims=False, category="unanswerable"),  # agrees
        row("T93", I, [C, P, I]),  # no majority: disagreement, no verdict
    ]
}


def test_summarize_matches_hand_counts():
    s = summarize(RESULT)
    assert s["n"] == 4 and s["passes"] == 3
    assert s["agreement"] == {"n": 2, "total": 4, "rate": 0.5}
    assert s["agreement_with_claims"] == {"n": 1, "total": 3, "rate": pytest.approx(1 / 3)}
    assert s["agreement_no_claims"] == {"n": 1, "total": 1, "rate": 1.0}
    # human [C, P, C, I], judge [C, C, C, no majority]: observed 2/4,
    # chance (C: 2 x 3) / 16 = 6/16, kappa (8/16 - 6/16) / (10/16) = 0.2
    assert s["kappa"] == pytest.approx(0.2)
    # correct-or-not: T90 yes, T91 no (P vs C), T92 yes, T93 yes (neither is correct)
    assert s["binary"]["n"] == 3
    # pass 1 C,C,C,C: T90, T92; pass 2 C,C,C,P: T90, T92; pass 3 C,P,C,I: all four
    assert [f["n"] for f in s["per_pass"]] == [2, 2, 4]
    assert s["consistent"]["n"] == 2  # T90 and T92
    assert s["direction"] == {"disagreements": 2, "lenient": 1, "strict": 0, "no_verdict": 1}
    assert s["by_category"]["answerable"]["n"] == 1 and s["by_category"]["unanswerable"]["n"] == 1
    assert s["cost"]["calls"] == 12 and s["cost"]["usd"] == pytest.approx(0.0012)


def test_an_invalid_judge_reply_is_a_verdict_of_its_own():
    result = {"questions": [row("T90", C, [C, C, C])]}
    result["questions"][0]["judge"] = [judged(C, "invalid"), judged(C, "invalid"), judged(C)]
    s = summarize(result)
    assert s["agreement"]["n"] == 0  # majority not_judged
    assert s["direction"]["no_verdict"] == 1


# --- hand grades ---------------------------------------------------------------------------


def write(tmp_path, text: str):
    path = tmp_path / "grades.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_grade_problems_lists_every_problem():
    grades = [
        Grade(id="T90", verdict=C),
        Grade(id="T91", verdict=""),  # empty
        Grade(id="T92", verdict="mostly_right"),  # unknown
        Grade(id="T93", verdict=C),
        Grade(id="T93", verdict=I),  # twice
        Grade(id="T99", verdict=C),  # not an answer
    ]
    problems = grade_problems(grades, ["T90", "T91", "T92", "T93", "T94"])
    assert sorted(problems) == sorted(
        [
            "T93: graded 2 times",
            "T99: not in the calibration answers",
            "T94: no [[grade]] block",
            "T91: verdict is empty",
            "T92: unknown verdict 'mostly_right' (use correct, partially_correct, incorrect)",
        ]
    )
    assert grade_problems([Grade(id="T90", verdict=P)], ["T90"]) == []


def test_load_grades_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_grades(write(tmp_path, '[[grades]]\nid = "T90"\n'))
    with pytest.raises(ValueError, match="score"):
        load_grades(write(tmp_path, '[[grade]]\nid = "T90"\nscore = 3\n'))


def test_the_template_parses_with_every_verdict_empty(tmp_path):
    rows = [{"id": "T90"}, {"id": "T91"}]
    path = write(tmp_path, grades_template(rows, "answers.json", "sheet.md"))
    grades = load_grades(path)
    assert [(g.id, g.verdict, g.note) for g in grades] == [("T90", "", ""), ("T91", "", "")]
    assert grade_problems(grades, ["T90", "T91"]) == [
        "T90: verdict is empty",
        "T91: verdict is empty",
    ]


# --- the blind sheet -----------------------------------------------------------------------


def test_rubric_is_the_judge_rules_without_the_reply_format():
    rules = rubric(load_prompt("judge_quality", "v1"))
    assert rules[0].startswith("- Grade meaning, not wording.")
    assert any("partially_correct:" in line for line in rules)
    assert not any("JSON" in line or "reasoning" in line for line in rules)
    with pytest.raises(ValueError, match="grading rules not found"):
        rubric(Prompt(name="x", version="v1", system="No bullets.", user="", sha256="0"))


def sheet_run():
    settings = {
        "model": "haiku-5.5",
        "prompt": {"name": "answer", "version": "v2"},
        "collection": "structure",
        "mode": "hybrid",
        "reranker": "r",
        "k": 10,
        "max_context_tokens": 4000,
        "min_rerank_score": None,
    }
    return {
        "settings": {"answer": settings},
        "questions": [
            # a judge field that must never reach the sheet
            {"id": "T90", **ANSWERED, QUALITY: {"output": {"reasoning": "SECRET VERDICT"}}},
            {"id": "T91", **GATED},
        ],
    }


def test_sheet_shows_what_the_judge_sees_and_no_judge_output():
    run = sheet_run()
    questions = {q.id: q for q in (ANSWERABLE, UNANSWERABLE)}
    paths = {"answers": "a.json", "grades": "g.toml"}
    sheet = sheet_markdown(run, questions, load_prompt("judge_quality", "v1"), paths)
    for r, q in zip(run["questions"], (ANSWERABLE, UNANSWERABLE), strict=True):
        assert render_answer(r) in sheet
        assert q.question in sheet and q.expected_answer in sheet
    assert "SECRET" not in sheet
    assert "score gate off" in sheet
    assert "- Grade meaning, not wording." in sheet


# --- the script ----------------------------------------------------------------------------


def load_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))  # it imports scripts/eval_answers.py
    spec = importlib.util.spec_from_file_location(
        "calibrate_judge", ROOT / "scripts" / "calibrate_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repo(tmp_path, monkeypatch, config):
    """Saved calibration answers of two invented questions in a tmp repo root."""
    script = load_script(monkeypatch)
    chunk_file = tmp_path / "chunks.jsonl"
    chunk_file.write_text(
        "".join(c.model_dump_json() + "\n" for c in CHUNKS.values()), encoding="utf-8"
    )
    test_file = tmp_path / "test_set.toml"
    test_file.write_text("# invented\n", encoding="utf-8")
    cal = tmp_path / "evals" / "results" / "calibration"
    cal.mkdir(parents=True)
    run = {
        "created_at": "2026-11-23T00:00:00+00:00",
        "test_set": {"path": "test_set.toml", "sha256": sha(test_file)},
        "chunks": {"path": "chunks.jsonl", "sha256": sha(chunk_file)},
        "settings": {"answer": {"model": "haiku-5.5", "gate": False}},
        "questions": [
            {"id": "T90", "sent": SENT, **ANSWERED},
            {"id": "T91", "sent": SENT[:1], **GATED},
        ],
    }
    (cal / "answers.json").write_text(json.dumps(run), encoding="utf-8")
    grades = tmp_path / "evals" / "judge_grades.toml"
    grades.write_text(grades_template(run["questions"], "a", "s"), encoding="utf-8")

    clients = []

    def make_client(model_key, cfg):
        # 3 passes x 2 answers: T90 correct, correct, partially; T91 correct every time
        client = FakeClient(*(quality_reply(v) for v in (C, C, C, C, P, C)))
        clients.append((model_key, client))
        return client

    monkeypatch.setattr(FakeClient, "close", lambda self: None, raising=False)
    for name, value in {
        "ROOT": tmp_path,
        "CALIBRATION": cal,
        "ANSWERS": cal / "answers.json",
        "SHEET": cal / "sheet.md",
        "GRADES": grades,
        "RESULTS": tmp_path / "evals" / "results" / "calibration.json",
        "MD_OUT": tmp_path / "evals" / "results" / "calibration.md",
        "JUDGEMENTS": tmp_path / "judgements",
        "TEST_SET": test_file,
        "load_test_set": lambda: [ANSWERABLE, UNANSWERABLE],
        "load_env": lambda: None,
        "make_client": make_client,
    }.items():
        monkeypatch.setattr(script, name, value)
    monkeypatch.setattr(script.ea, "ROOT", tmp_path)
    monkeypatch.setattr(script.ea, "spent_on_answers", lambda: 0.0)
    return script, grades, clients, tmp_path


def grade_all(grades, t90=C, t91=C):
    grades.write_text(
        f'[[grade]]\nid = "T90"\nverdict = "{t90}"\nnote = ""\n\n'
        f'[[grade]]\nid = "T91"\nverdict = "{t91}"\nnote = "a refusal"\n',
        encoding="utf-8",
    )


def test_no_judge_is_called_until_every_answer_is_graded(repo, config):
    script, grades, clients, _ = repo
    with pytest.raises(SystemExit, match="T90: verdict is empty"):
        script.calibrate(config)
    grades.write_text('[[grade]]\nid = "T90"\nverdict = "correct"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="T91: no"):
        script.calibrate(config)
    assert clients == []


def test_calibrate_judges_every_answer_three_times_after_the_grades(repo, config):
    script, grades, clients, root = repo
    grade_all(grades, t90=C, t91=C)
    before = sha(grades)
    result = script.calibrate(config)
    assert result["grades"]["sha256"] == before
    assert [model for model, _ in clients] == [config.judge.model]
    [(_, client)] = clients
    assert [c["schema_name"] for c in client.calls] == ["judge_quality"] * 6
    t90, t91 = result["questions"]
    assert [j["output"]["verdict"] for j in t90["judge"]] == [C, C, P]
    assert [j["output"]["verdict"] for j in t91["judge"]] == [C, C, C]
    assert (t90["has_claims"], t91["has_claims"]) == (True, False)
    assert t91["outcome"] == "below_threshold" and t91["human"]["note"] == "a refusal"
    assert result["settings"]["passes"] == 3
    assert len(list((root / "judgements").glob("*.json"))) == 6
    assert summarize(result)["agreement"]["n"] == 2


def test_the_report_stops_when_the_grades_changed_after_the_judge(repo, config):
    script, grades, _, _ = repo
    grade_all(grades)
    script.write_json(script.RESULTS, script.calibrate(config))
    assert script.load_results()["grades"]["sha256"] == sha(grades)
    grade_all(grades, t90=P)
    with pytest.raises(SystemExit, match="changed after the judge ran"):
        script.load_results()


def test_the_judge_stops_when_the_chunks_changed(repo, config):
    script, grades, clients, root = repo
    grade_all(grades)
    (root / "chunks.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed since"):
        script.calibrate(config)
    assert clients == []


def test_answers_are_never_overwritten(repo, config, monkeypatch):
    script, _, _, _ = repo

    def answer_all(*args, **kwargs):
        raise AssertionError("no answer may be made")

    monkeypatch.setattr(script.ea, "answer_all", answer_all)
    with pytest.raises(SystemExit, match="exists"):
        script.make_answers(config)


def test_make_answers_answers_with_the_gate_off_and_writes_an_empty_grade_file(
    repo, config, monkeypatch
):
    script, grades, _, root = repo
    script.ANSWERS.unlink()
    grades.unlink()
    calls = []

    def answer_all(cfg, questions, spent, gate=True):
        calls.append(gate)
        rows = [{"id": "T90", "sent": SENT, **ANSWERED}, {"id": "T91", "sent": [], **GATED}]
        return {**sheet_run()["settings"]["answer"]}, rows

    monkeypatch.setattr(script.ea, "answer_all", answer_all)
    monkeypatch.setattr(script.ea, "chunks_path", lambda cfg: root / "chunks.jsonl")
    script.make_answers(config)
    assert calls == [False]
    saved = json.loads(script.ANSWERS.read_text(encoding="utf-8"))
    assert saved["settings"]["answer"]["gate"] is False
    assert [r["id"] for r in saved["questions"]] == ["T90", "T91"]
    assert [g.verdict for g in load_grades(grades)] == ["", ""]
    assert render_answer(saved["questions"][0]) in script.SHEET.read_text(encoding="utf-8")


def test_answer_all_passes_the_gate_flag_to_the_pipeline(monkeypatch, config):
    script = load_script(monkeypatch)
    ea = script.ea
    seen = []

    class Retriever:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def info(self):
            keys = ("collection", "mode", "embedding_model", "reranker", "rerank_candidates")
            return dict.fromkeys(keys, "x")

        def search(self, question):
            return []

    def answer_question(*args, gate=True, **kwargs):
        seen.append(gate)
        return {"final_status": "no_context"}

    monkeypatch.setattr(ea, "open_retriever", lambda cfg: Retriever())
    monkeypatch.setattr(ea, "make_client", lambda key, cfg: FakeClient())
    monkeypatch.setattr(FakeClient, "close", lambda self: None, raising=False)
    monkeypatch.setattr(ea, "answer_question", answer_question)
    monkeypatch.setattr(ea, "save_record", lambda record: ROOT / "results" / "answers" / "x.json")
    monkeypatch.setattr(ea, "record_cost", lambda record: 0.0)
    monkeypatch.setattr(ea, "answer_row", lambda q, record, path: {"id": q.id})
    settings, _ = ea.answer_all(config, [ANSWERABLE], 0.0, gate=False)
    assert seen == [False] and settings["min_rerank_score"] is None
    settings, _ = ea.answer_all(config, [ANSWERABLE], 0.0)
    assert seen == [False, True]
    assert settings["min_rerank_score"] == config.answer.min_rerank_score


# --- the committed calibration files -------------------------------------------------------


def test_committed_sheet_regenerates_from_the_saved_answers(monkeypatch, config):
    script = load_script(monkeypatch)
    run = json.loads(script.ANSWERS.read_text(encoding="utf-8"))
    assert run["settings"]["answer"]["gate"] is False
    assert run["settings"]["answer"]["min_rerank_score"] is None
    questions = {q.id: q for q in load_test_set()}
    assert [r["id"] for r in run["questions"]] == list(questions)
    sheet = sheet_markdown(run, questions, script.quality_prompt(config), script.paths())
    assert script.SHEET.read_text(encoding="utf-8") == sheet
    assert [g.id for g in load_grades(script.GRADES)] == list(questions)


def test_the_report_explains_kappa_when_every_hand_grade_is_the_same():
    one_level = {"questions": [row("T90", C, [C, C, C]), row("T91", C, [I, I, I])]}
    s = summarize(one_level)
    # human C, C; judge C, I: observed 1/2, chance (C: 2 x 1) / 4 = 1/2, kappa 0
    assert s["kappa"] == pytest.approx(0.0)
    assert "kappa is 0 whatever the judge does" in grade_spread_line(s)
    assert "no answer graded other than correct" in grade_spread_line(s)
    two_levels = summarize({"questions": [row("T90", C, [C] * 3), row("T91", P, [P] * 3)]})
    assert grade_spread_line(two_levels).startswith("No answer is graded incorrect")
    assert grade_spread_line(summarize(RESULT)) is None  # C, P and I all used


@pytest.fixture(scope="module")
def committed():
    path = ROOT / "evals" / "results" / "calibration.json"
    assert path.exists(), "no committed calibration: run `make calibrate-judge`"
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_grades_are_the_ones_the_judge_ran_after(committed):
    grades = ROOT / committed["grades"]["path"]
    assert sha(grades) == committed["grades"]["sha256"]
    answers = ROOT / committed["answers"]["path"]
    assert sha(answers) == committed["answers"]["sha256"]
    by_id = {g.id: g for g in load_grades(grades)}
    assert grade_problems(list(by_id.values()), [r["id"] for r in committed["questions"]]) == []
    for r in committed["questions"]:
        assert r["human"] == {"verdict": by_id[r["id"]].verdict, "note": by_id[r["id"]].note}
        assert len(r["judge"]) == committed["settings"]["passes"] == 3


def test_committed_calibration_report_regenerates(committed):
    path = "evals/results/calibration.json"
    md = (ROOT / "evals" / "results" / "calibration.md").read_text(encoding="utf-8")
    assert md == markdown(committed, path)
    report = (ROOT / "docs" / "reports" / "stage-2.md").read_text(encoding="utf-8")
    section = re.search(re.escape(START) + r"\n(.*?)" + re.escape(END), report, re.S).group(1)
    assert section.strip() == report_section(committed, path).strip()
