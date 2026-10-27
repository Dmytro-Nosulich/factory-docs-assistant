"""Every saved result file says which configuration produced it (S1-16).

Checks the committed files under results/: model, prompt version and hash, parameters, and
for a local model the sampling configuration. Runs from before S1-16 lack the later fields
(model digest, input cap, explicit `sampling: null` for the cloud model), so only these core
fields are checked here; the new ones are covered by test_extract.py and test_run.py.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
FILES = sorted([*RESULTS.glob("runs/*/*.json"), *RESULTS.glob("smoke/*.json")])


def test_there_are_results_to_check():
    assert FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(RESULTS)))
def test_result_file_names_its_configuration(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    model, prompt, params = data["model"], data["prompt"], data["params"]
    assert model["key"] and model["client"] and model["slug"]
    assert prompt["version"] and prompt["sha256"]
    assert isinstance(params, dict)
    if model["client"] == "ollama":
        assert params["sampling"]["name"]
        assert params["num_ctx"] and params["num_predict"]


def test_report_tables_are_up_to_date():
    """REPORT.md shows what `make comparison` makes from the committed runs (S1-17)."""
    from factory_docs.config import load_config
    from factory_docs.report import REPORT, comparison_section, read_section

    section, _, _ = comparison_section(load_config())
    assert read_section(REPORT.read_text(encoding="utf-8")) == section, "run `make comparison`"
