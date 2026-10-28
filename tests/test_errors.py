import copy
import json

from factory_docs.config import load_config, run_params
from factory_docs.errors import collect_errors, format_errors, item_diff, show_range
from factory_docs.metrics import FIELDS, hazard_match
from factory_docs.prompts import load_prompt

DOCS = ["G1", "G2"]
HAIKU = ("haiku-5.5", None, "v1")
QWEN = ("qwen3.5-9b", "greedy", "v1")

# An invented product, not a corpus document.
GOLDEN = {
    "product_name": "Example Degreaser 200",
    "language": "en",
    "signal_word": "danger",
    "physical_state": "liquid",
    "components": [
        {
            "name": "Ethanol",
            "cas_number": "64-17-5",
            "concentration": {"min": 10.0, "max": 25.0, "unit": "percent"},
        },
        {
            "name": "Propan-2-ol",
            "cas_number": "67-63-0",
            "concentration": {"min": None, "max": 3.0, "unit": "percent"},
        },
    ],
    "hazard_statements": [{"code": "H225", "text": "Highly flammable liquid and vapour."}],
    "flash_point": {"min": 12.0, "max": None, "unit": "celsius"},
    "un_number": "UN1993",
    "revision_date": "2024-05-02",
}


def wrong_output():
    """GOLDEN with two misses: `< 3 %` as min = max = 3, and the language."""
    output = copy.deepcopy(GOLDEN)
    output["components"][1]["concentration"] = {"min": 3.0, "max": 3.0, "unit": "percent"}
    output["language"] = "de"
    return output


def write_run(runs_dir, run_id, cell, records):
    """A run of `cell` that is complete with the current configuration (as in test_report)."""
    config = load_config()
    model_key, sampling, version = cell
    summary = {
        "run_id": run_id,
        "model": {"key": model_key},
        "prompt": {"version": version, "sha256": load_prompt("extract", version).sha256},
        "params": run_params(config, model_key, sampling),
        "max_input_tokens": config.extraction.max_input_tokens,
        "docs": DOCS,
        "counts": {"not_run": 0},
        "finished_at": "2026-10-27T10:00:00+00:00",
    }
    (runs_dir / run_id).mkdir(parents=True)
    (runs_dir / run_id / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    for doc_id, status, output in records:
        record = {"doc_id": doc_id, "status": status, "output": output}
        (runs_dir / run_id / f"{doc_id}.json").write_text(json.dumps(record), encoding="utf-8")


def collect(tmp_path, runs):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    for doc_id in DOCS:
        (golden_dir / f"{doc_id}.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    for run_id, cell, records in runs:
        write_run(tmp_path / "runs", run_id, cell, records)
    return collect_errors(load_config(), tmp_path / "runs", golden_dir, DOCS)


def test_same_miss_in_two_runs_is_one_row_and_correct_fields_are_not_listed(tmp_path):
    run_ids, misses = collect(
        tmp_path,
        [
            ("20261027T000000Z_haiku-5.5_v1", HAIKU, [("G1", "valid", wrong_output())]),
            ("20261027T000001Z_qwen3.5-9b_greedy_v1", QWEN, [("G1", "valid", wrong_output())]),
        ],
    )
    assert len(run_ids) == 2
    assert [(m.doc_id, m.field, m.cells) for m in misses] == [
        ("G1", "language", ["haiku-5.5 v1", "qwen3.5-9b greedy v1"]),
        ("G1", "components", ["haiku-5.5 v1", "qwen3.5-9b greedy v1"]),
    ]
    language = misses[0]
    assert (language.extracted, language.expected) == ('"de"', '"en"')


def test_list_field_shows_only_the_differing_items(tmp_path):
    _, misses = collect(
        tmp_path, [("20261027T000000Z_haiku-5.5_v1", HAIKU, [("G1", "valid", wrong_output())])]
    )
    components = next(m for m in misses if m.field == "components")
    assert components.extracted == "Propan-2-ol 67-63-0 [3.0, 3.0] %"  # Ethanol matches
    assert components.expected == "Propan-2-ol 67-63-0 [null, 3.0] %"


def test_failed_result_misses_every_field(tmp_path):
    _, misses = collect(
        tmp_path,
        [
            (
                "20261027T000000Z_haiku-5.5_v1",
                HAIKU,
                [("G1", "invalid", None), ("G2", "valid", GOLDEN)],
            )
        ],
    )
    assert [m.field for m in misses] == FIELDS
    assert {(m.doc_id, m.extracted, m.expected) for m in misses} == {("G1", "<invalid>", "")}


def test_item_diff_pairs_one_to_one():
    h225 = {"code": "H225", "text": "Highly flammable liquid and vapour."}
    h319 = {"code": "H319", "text": "Causes serious eye irritation."}
    assert item_diff([h225, h225], [h225, h319], hazard_match) == ([h225], [h319])


def test_show_range():
    assert show_range(None) == "null"
    assert show_range({"min": 190.0, "max": None, "unit": "celsius"}) == "[190.0, null] °C"


def test_format_counts_misses_per_field(tmp_path):
    run_ids, misses = collect(
        tmp_path, [("20261027T000000Z_haiku-5.5_v1", HAIKU, [("G1", "valid", wrong_output())])]
    )
    text = format_errors(run_ids, misses, languages={"G1": "en"})
    assert '| G1 | en | language | 1/1: haiku-5.5 v1 | "de" | "en" |' in text
    assert "language 1, signal_word 0" in text and "total 2." in text
