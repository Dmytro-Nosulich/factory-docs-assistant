"""The S2-20 eval gate: each threshold just at and just below its value on an invented run,
the errors that stop the gate, the config section, the committed results, the script and the
GitHub Actions workflow."""

import importlib.util
import json
import sys

import pytest

from evals.gate import GateError, at_least, check_fresh, evaluate, retrieval_checks
from factory_docs.config import ROOT, load_config
from tests.test_eval_answers import judgement, run_row, supports, verdict

WORKFLOW = ROOT / ".github" / "workflows" / "evals.yml"


@pytest.fixture(scope="module")
def config():
    return load_config()


def gate(config, **update):
    """[eval_gate] with every threshold at 0 (or no limit), then the given ones."""
    base = {
        "min_mrr": 0.0,
        "min_recall_at_5": 0.0,
        "min_passed": 0.0,
        "min_unanswerable_refused": 0.0,
        "min_judge_correct": 0.0,
        "min_claims_supported": 0.0,
        "max_not_judged": 99,
    }
    return config.eval_gate.model_copy(update={**base, **update})


# An invented run, counted by hand:
#   passed 6/8 (A1-A4, U1, U2); unanswerable refused 2/3 (U1, U2; U3 answered);
#   judge correct 6/8 (A1-A4, U1, U2); claims supported 6/7 (A1-A4 1 each, A5 1 of 2, U3 1).
def make_run(quality_a1=None):
    rows = [
        run_row(f"A{i}", "answerable", "en", True, verdict("correct"), supports("supported"), 0.0)
        for i in range(1, 5)
    ]
    if quality_a1 is not None:
        rows[0]["quality"] = quality_a1
    rows.append(
        run_row(
            "A5",
            "answerable",
            "de",
            False,
            verdict("incorrect"),
            supports("supported", "not_supported"),
            0.0,
        )
    )
    for qid in ("U1", "U2"):
        rows.append(
            run_row(
                qid,
                "unanswerable",
                "en",
                True,
                verdict("correct"),
                judgement("no_claims"),
                0.0,
                called=False,
            )
        )
    rows.append(
        run_row("U3", "unanswerable", "uk", False, verdict("incorrect"), supports("supported"), 0.0)
    )
    return {
        "answers_from": None,
        "git": {"commit": "abc123", "dirty": False},
        "settings": {"answer": {}, "judge": {"model": "haiku-5.5"}},
        "questions": rows,
    }


RETRIEVAL = {
    "runs": {
        "structure": {
            "hybrid+rerank": {
                "summary": {"questions": 15, "mrr": 0.6, "recall": {"5": 0.8}, "hits": {"5": 12}}
            }
        }
    }
}


def results(checks):
    return {c.name: c.ok for c in checks}


@pytest.mark.parametrize(
    ("field", "check", "at", "above"),
    [
        ("min_mrr", "retrieval structure hybrid+rerank: MRR@10", 0.6, 0.601),
        ("min_recall_at_5", "retrieval structure hybrid+rerank: recall@5", 0.8, 0.81),
        ("min_passed", "test set: passed", 6 / 8, 0.76),
        ("min_unanswerable_refused", "test set: unanswerable refused", 2 / 3, 0.67),
        ("min_judge_correct", "judge: correct", 6 / 8, 0.76),
        ("min_claims_supported", "judge: claims supported", 6 / 7, 0.86),
    ],
)
def test_each_threshold_passes_at_its_value_and_fails_just_above(config, field, check, at, above):
    passing = results(evaluate(RETRIEVAL, make_run(), gate(config, **{field: at})))
    assert all(passing.values()), passing
    failing = results(evaluate(RETRIEVAL, make_run(), gate(config, **{field: above})))
    assert failing[check] is False
    assert sum(not ok for ok in failing.values()) == 1, failing


def test_rates_on_the_threshold_survive_float_rounding(config):
    # 15/24 = 0.625 and 12/15 = 0.8 exactly at the committed thresholds
    assert at_least("x", {"n": 15, "total": 24, "rate": 15 / 24}, 0.625).ok
    assert at_least("x", {"n": 12, "total": 15, "rate": 12 / 15}, 0.80).ok
    assert not at_least("x", {"n": 14, "total": 24, "rate": 14 / 24}, 0.625).ok


def test_nothing_to_count_fails():
    check = at_least("x", {"n": 0, "total": 0, "rate": None}, 0.0)
    assert not check.ok
    assert check.value == "0/0"


def test_not_judged_answers_are_counted_for_each_judge(config):
    run = make_run(quality_a1=judgement("invalid"))
    run["questions"][1]["faithfulness"] = judgement("invalid")
    ok = results(evaluate(RETRIEVAL, run, gate(config, max_not_judged=1)))
    assert ok["judge: quality not judged"] and ok["judge: faithfulness not judged"]
    failing = results(evaluate(RETRIEVAL, run, gate(config, max_not_judged=0)))
    assert not failing["judge: quality not judged"]
    assert not failing["judge: faithfulness not judged"]


def test_a_run_without_judgements_cannot_pass(config):
    run = make_run()
    run["settings"]["judge"] = None
    with pytest.raises(GateError, match="judges never ran"):
        evaluate(RETRIEVAL, run, gate(config))


def test_missing_retrieval_setup_is_an_error(config):
    with pytest.raises(GateError, match="make eval-retrieval"):
        retrieval_checks({"runs": {"structure": {"hybrid": {}}}}, gate(config))
    with pytest.raises(GateError, match="make eval-retrieval"):
        retrieval_checks({}, gate(config))


def test_only_a_run_from_head_is_fresh():
    run = make_run()
    check_fresh(run, "abc123")
    with pytest.raises(GateError, match="not at HEAD"):
        check_fresh(run, "def456")
    with pytest.raises(GateError, match="not at HEAD"):
        check_fresh(run, None)


# --- config ---------------------------------------------------------------------------------


def test_config_names_a_measured_collection_and_setup(config):
    raw = config.model_dump()
    bad_collection = {**raw, "eval_gate": {**raw["eval_gate"], "retrieval_collection": "x"}}
    with pytest.raises(ValueError, match="retrieval_collection"):
        type(config).model_validate(bad_collection)
    bad_setup = {**raw, "eval_gate": {**raw["eval_gate"], "retrieval_setup": "hybrid+x"}}
    with pytest.raises(ValueError, match="retrieval_setup"):
        type(config).model_validate(bad_setup)
    no_k5 = {**raw, "retrieval_eval": {**raw["retrieval_eval"], "ks": [1, 3, 10]}}
    with pytest.raises(ValueError, match="needs 5"):
        type(config).model_validate(no_k5)


# --- the committed results and the script ---------------------------------------------------


def latest_committed_run():
    runs = sorted((ROOT / "evals" / "results" / "runs").glob("*.json"))
    assert runs, "no committed run file: run `make eval`"
    return runs[-1]


def test_committed_results_clear_the_committed_thresholds(config):
    retrieval = json.loads((ROOT / "evals/results/retrieval.json").read_text(encoding="utf-8"))
    run = json.loads(latest_committed_run().read_text(encoding="utf-8"))
    failed = [c for c in evaluate(retrieval, run, config.eval_gate) if not c.ok]
    assert not failed, failed


def load_script():
    spec = importlib.util.spec_from_file_location("eval_gate", ROOT / "scripts" / "eval_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_passes_writes_the_job_summary_and_fails_on_a_raised_threshold(
    config, tmp_path, monkeypatch, capsys
):
    script = load_script()
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(sys, "argv", ["eval_gate.py", "--run", str(latest_committed_run())])
    script.main()
    assert "## Eval gate: OK" in summary.read_text(encoding="utf-8")

    strict = config.model_copy(update={"eval_gate": gate(config, min_passed=1.0)})
    monkeypatch.setattr(script, "load_config", lambda: strict)
    with pytest.raises(SystemExit) as e:
        script.main()
    assert e.value.code == 1
    assert "FAIL, 1 of 8 checks" in capsys.readouterr().out


def test_script_without_run_refuses_a_run_not_from_head(tmp_path, monkeypatch):
    script = load_script()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20990101T000000Z.json").write_text(json.dumps(make_run()), encoding="utf-8")
    monkeypatch.setattr(script, "RUNS", runs)
    monkeypatch.setattr(script, "head_commit", lambda: "def456")
    monkeypatch.setattr(sys, "argv", ["eval_gate.py"])
    with pytest.raises(SystemExit, match="not at HEAD"):
        script.main()


# --- the workflow (plain text: PyYAML is not a dependency) ----------------------------------


def test_workflow_runs_make_run_with_the_key_on_push_and_pull_request():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "\n  push:" in text and "\n  pull_request:" in text
    assert "OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}" in text
    assert "run: make run-rag" in text
    assert text.index("run: make test") < text.index("run: make run-rag")
    assert "build_index.py --if-stale" in text
