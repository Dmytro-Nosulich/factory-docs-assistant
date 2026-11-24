"""The S2-21 "Results at a glance" section: values from an invented run counted by hand, missing
inputs, and the committed REPORT.md summary matching the committed result files."""

import importlib.util
import sys

import pytest

from evals.gate import evaluate
from evals.summary import END, START, section
from factory_docs.config import ROOT, load_config
from factory_docs.retrieval_eval import replace_section
from tests.test_eval_gate import RETRIEVAL, gate, make_run


@pytest.fixture(scope="module")
def config():
    return load_config()


def rows(text: str) -> dict[str, str]:
    """Measure -> value, from the table of a summary section."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 3 and cells[0] != "Measure":
            out[cells[0]] = cells[1]
    return out


def test_values_from_an_invented_run(config):
    # make_run (tests/test_eval_gate.py): passed 6/8; unanswerable refused 2/3 (U1, U2);
    # answerable refused 0/5; judge correct 6/8; claims supported 6/7. RETRIEVAL: MRR 0.6, 12/15.
    run = make_run()
    settings = gate(config)
    checks = evaluate(RETRIEVAL, run, settings)
    got = rows(section(settings, RETRIEVAL, run, "runs/x.json", None, None, checks))
    label = "gold set, `structure` chunks, hybrid+rerank"
    assert got[f"Retrieval MRR@10 ({label})"] == "0.600"
    assert got[f"Retrieval recall@5 ({label})"] == "12/15 (80%)"
    assert got["Test set passed (deterministic scorers)"] == "6/8 (75%)"
    assert got["Unanswerable test questions refused"] == "2/3 (67%)"
    assert got["Answerable and trap questions refused (lower is better)"] == "0/5 (0%)"
    assert got["Quality judge: correct"] == "6/8 (75%)"
    assert got["Faithfulness judge: claims supported"] == "6/7 (86%)"
    gate_row = "CI eval gate (`make eval-gate`, thresholds in config.toml `[eval_gate]`)"
    assert got[gate_row] == "8/8 checks pass"


def test_missing_inputs_name_the_command_that_makes_them(config):
    got = rows(section(config.eval_gate, None, None, None, None, None, None))
    values = list(got.values())
    for command in ("make eval-retrieval", "make eval", "make calibrate-judge", "make cost"):
        assert f"not run yet (`{command}`)" in values


def load_script():
    path = ROOT / "scripts" / "report_summary.py"
    spec = importlib.util.spec_from_file_location("report_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_report_summary_is_current_and_stable(tmp_path, monkeypatch):
    """Rebuilding the summary from the committed results leaves REPORT.md unchanged, twice."""
    script = load_script()
    report = ROOT / "REPORT.md"
    copy = tmp_path / "REPORT.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(script, "REPORT", copy)
    monkeypatch.setattr(sys, "argv", ["report_summary.py"])
    script.main()
    first = copy.read_text(encoding="utf-8")
    script.main()
    assert copy.read_text(encoding="utf-8") == first
    assert first == report.read_text(encoding="utf-8"), "run `make report-summary`"


def test_only_the_marked_section_changes():
    text = f"before\n{START}\nold\n{END}\nafter\n"
    assert replace_section(text, "new", START, END) == f"before\n{START}\n\nnew\n\n{END}\nafter\n"
