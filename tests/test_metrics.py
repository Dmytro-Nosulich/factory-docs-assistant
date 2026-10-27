import json

import pytest

from factory_docs.config import load_config
from factory_docs.metrics import (
    ItemCounts,
    component_match,
    format_tables,
    hazard_match,
    match_list,
    normalize_text,
    percentile,
    ranges_match,
    run_metrics,
    score_document,
)

# An invented product, not a corpus document. Also the worked example in docs/METRICS.md.
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
            "concentration": {"min": 1.0, "max": 3.0, "unit": "percent"},
        },
    ],
    "hazard_statements": [
        {"code": "H225", "text": "Highly flammable liquid and vapour."},
        {"code": "H319", "text": "Causes serious eye irritation."},
    ],
    "flash_point": {"min": 12.0, "max": 12.0, "unit": "celsius"},
    "un_number": "UN1993",
    "revision_date": "2024-05-02",
}

# Hand-scored against GOLDEN: 7 of 9 fields match.
OUTPUT = {
    "product_name": "EXAMPLE  degreaser-200",  # match: case, hyphen, spaces normalized
    "language": "en",  # match
    "signal_word": "warning",  # MISS
    "physical_state": "liquid",  # match
    "components": [GOLDEN["components"][1], GOLDEN["components"][0]],  # match: order ignored
    "hazard_statements": [
        {"code": "H319", "text": "Causes eye irritation."},  # MISS: text differs
        {"code": "H225", "text": "Highly flammable liquid and vapour"},  # match: "." dropped
    ],
    "flash_point": {"min": 12.4, "max": 12.4, "unit": "celsius"},  # match: within 0.5
    "un_number": "UN1993",  # match
    "revision_date": "2024-05-02",  # match
}


def test_normalize_text():
    assert normalize_text("BIOGON® N") == "biogon n"
    assert normalize_text("Stickstoff,  verdichtet\n") == "stickstoff verdichtet"
    assert normalize_text("ＡＢＣ") == "abc"  # NFKC: full-width letters
    assert normalize_text("Паспорт безпеки  можна") == "паспорт безпеки можна"
    assert normalize_text("Propan-2-ol") == normalize_text("propan 2 ol")


def test_ranges_match_tolerance_and_nulls():
    def celsius(lo, hi):
        return {"min": lo, "max": hi, "unit": "celsius"}

    assert ranges_match(celsius(12.4, 12.4), celsius(12.0, 12.0))
    assert not ranges_match(celsius(12.6, 12.6), celsius(12.0, 12.0))
    assert ranges_match(celsius(65.0, None), celsius(65.0, None))
    assert not ranges_match(celsius(65.0, 65.0), celsius(65.0, None))
    assert not ranges_match(None, celsius(65.0, None))
    assert ranges_match(None, None)
    percent = {"min": 12.0, "max": 12.0, "unit": "percent"}
    assert not ranges_match(percent, celsius(12.0, 12.0))
    assert not ranges_match({**percent, "min": 12.02}, percent)  # percent tolerance 0.01


def test_components_wrong_cas_is_a_binary_miss_with_partial_f1():
    ethanol, propanol = GOLDEN["components"]
    predicted = [{**ethanol, "cas_number": "64-17-6"}, propanol, {**propanol, "name": "Water"}]
    matched = match_list(predicted, GOLDEN["components"], component_match)
    counts = ItemCounts(matched, len(predicted), len(GOLDEN["components"]))
    assert counts == ItemCounts(1, 3, 2)  # precision 1/3, recall 1/2
    assert counts.f1 == pytest.approx(0.4)  # 2·1 / (3 + 2)
    output = {**GOLDEN, "components": predicted}
    assert score_document(output, GOLDEN).fields["components"] is False


def test_hazard_code_case_matters_and_text_must_match():
    statement = {"code": "H360Fd", "text": "May damage fertility."}
    assert hazard_match(statement, {**statement, "text": "may damage fertility"})
    assert not hazard_match(statement, {**statement, "code": "H360FD"})
    assert not hazard_match(statement, {**statement, "text": "May damage the unborn child."})


def test_score_document_matches_hand_count():
    score = score_document(OUTPUT, GOLDEN)
    missed = sorted(f for f, ok in score.fields.items() if not ok)
    assert missed == ["hazard_statements", "signal_word"]
    assert sum(score.fields.values()) == 7
    assert score.items["components"] == ItemCounts(2, 2, 2)
    assert score.items["hazard_statements"] == ItemCounts(1, 2, 2)
    assert score.items["hazard_statements"].f1 == 0.5
    assert score.items["hazard_codes"] == ItemCounts(2, 2, 2)


def test_failed_document_misses_every_field():
    score = score_document(None, GOLDEN)
    assert not any(score.fields.values())
    assert len(score.fields) == 9
    assert score.items["components"] == ItemCounts(0, 0, 2)
    assert score.items["components"].f1 == 0.0


def test_both_lists_empty_is_a_match():
    golden = {**GOLDEN, "components": [], "hazard_statements": []}
    score = score_document(golden, golden)
    assert all(score.fields.values())
    assert score.items["components"].f1 == 1.0


def test_percentile_nearest_rank():
    values = [float(v) for v in range(20, 0, -1)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([5.0], 95) == 5.0


def record(doc_id, status, output=None, latency=1.0, prompt_tokens=1000, completion_tokens=100):
    call = None
    if status != "error":
        call = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": 0.001,  # deliberately not tokens x price: the two columns are separate
            "latency_s": latency,
        }
    return {"doc_id": doc_id, "status": status, "call": call, "output": output}


def write_run(tmp_path, model_key, client, records):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "run",
        "model": {"key": model_key, "client": client},
        "prompt": {"version": "v1"},
        "params": {},
        "docs": [r["doc_id"] for r in records],
    }
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    for r in records:
        (run_dir / f"{r['doc_id']}.json").write_text(json.dumps(r), encoding="utf-8")
    return run_dir


def test_run_metrics_matches_hand_computed_values(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    for doc_id in ("G1", "G2", "G3"):
        (golden_dir / f"{doc_id}.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    run_dir = write_run(
        tmp_path,
        "cloud",
        "openrouter",
        [
            record("G1", "valid", OUTPUT, 1.0, 1000, 100),  # 7/9
            record("G2", "valid", GOLDEN, 4.0, 3000, 200),  # 9/9
            record("G3", "invalid", None, 2.0, 2000, 0),  # 0/9
            record("X1", "valid", OUTPUT, 3.0, 2000, 400),  # no golden label: not scored
            record("E1", "error"),  # no call: no latency, no tokens
        ],
    )
    config = load_config()
    cloud = config.models["haiku-5.5"].model_copy(
        update={"price_in_usd_per_mtok": 0.10, "price_out_usd_per_mtok": 0.50}
    )
    config = config.model_copy(update={"models": {"cloud": cloud}})

    row = run_metrics(run_dir, config, golden_dir)

    assert (row["docs"], row["valid"]) == (5, 3)
    assert (row["golden_docs"], row["fields_matched"]) == (3, 16)  # 7 + 9 + 0 of 27
    assert (row["valid_golden_docs"], row["fields_matched_valid"]) == (2, 16)  # of 18
    assert row["per_field"]["signal_word"] == 1  # only G2
    assert row["items"]["hazard_statements"] == ItemCounts(3, 4, 6)
    assert row["latency"] == {"n": 4, "p50": 2.0, "p95": 4.0, "cached": 0}
    # tokens x price per document: 0.00015, 0.0004, 0.0002, 0.0004; mean 0.0002875
    assert row["cost_per_1000"]["priced"] == pytest.approx(0.2875)
    assert row["cost_per_1000"]["reported"] == pytest.approx(1.0)
    assert row["valid_after_repair"] is None  # results from before the repair step (S1-15)
    assert row["fields_matched_after_repair"] is None
    table = format_tables([row])
    assert "16/27 (59.3%)" in table and "16/18 (88.9%)" in table
    assert "| 3/5 (60.0%) | - | 3 | 16/27 (59.3%) | - |" in table


def repaired(record_, status, output=None):
    """record_ after its repair attempt; status is the repair reply's."""
    record_["repair"] = {
        "status": status,
        "call": {"prompt_tokens": 1500, "completion_tokens": 100, "cost_usd": 0.001},
        "output": output,
    }
    record_["final_status"] = "repaired" if status == "valid" else "failed"
    return record_


def test_after_repair_metrics_match_hand_computed_values(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    for doc_id in ("G1", "G2", "G3"):
        (golden_dir / f"{doc_id}.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    valid, error = record("G1", "valid", OUTPUT), record("E1", "error")  # 7/9 both ways
    valid["final_status"], error["final_status"] = "valid", "error"
    records = [
        valid,
        repaired(record("G2", "invalid", latency=2.0), "valid", GOLDEN),  # 0/9 -> 9/9
        repaired(record("G3", "invalid", latency=3.0), "invalid"),  # 0/9 -> 0/9
        error,
    ]
    run_dir = write_run(tmp_path, "cloud", "openrouter", records)
    config = load_config()
    cloud = config.models["haiku-5.5"].model_copy(
        update={"price_in_usd_per_mtok": 0.10, "price_out_usd_per_mtok": 0.50}
    )
    config = config.model_copy(update={"models": {"cloud": cloud}})

    row = run_metrics(run_dir, config, golden_dir)

    assert (row["valid"], row["valid_after_repair"], row["repair_attempts"]) == (1, 2, 2)
    assert (row["fields_matched"], row["fields_matched_after_repair"]) == (7, 16)  # of 27
    assert row["per_field"]["signal_word"] == 0  # the per-field table stays first-response
    assert row["latency"]["n"] == 3  # first calls only: G1, G2, G3
    # per document: G1 0.00015; G2 and G3 0.00015 + repair 0.0002 = 0.00035; mean 0.000283
    assert row["cost_per_1000"]["priced"] == pytest.approx(0.85 / 3)
    assert row["cost_per_1000"]["reported"] == pytest.approx(5 / 3)  # 0.001, 0.002, 0.002
    table = format_tables([row])
    assert "| 1/4 (25.0%) | 2/4 (50.0%) | 3 | 7/27 (25.9%) | 16/27 (59.3%) |" in table


def test_local_run_costs_nothing_and_counts_cached_prompts(tmp_path):
    cached = record("G1", "valid", OUTPUT)
    cached["call"]["details"] = {"prompt_cached_tokens": 900}
    prefix_only = record("G3", "valid", OUTPUT)  # only the shared system prompt was cached
    prefix_only["call"]["details"] = {"prompt_cached_tokens": 114}
    records = [cached, record("G2", "valid", OUTPUT), prefix_only]
    run_dir = write_run(tmp_path, "qwen3.5-9b", "ollama", records)

    row = run_metrics(run_dir, load_config(), tmp_path)

    assert row["golden_docs"] == 0
    assert row["latency"]["cached"] == 1  # 900/1000 counts, 114/1000 does not
    assert row["cost_per_1000"]["priced"] == 0.0
    assert "0 (local)" in format_tables([row])


def test_per_language_counts_and_table(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    for doc_id in ("G1", "G2"):
        (golden_dir / f"{doc_id}.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    records = [
        record("G1", "valid", OUTPUT),  # en, 7/9
        record("X1", "invalid"),  # en, no golden label
        record("G2", "invalid"),  # uk, 0/9
        record("X2", "valid", OUTPUT),  # sv, no golden label
    ]
    run_dir = write_run(tmp_path, "qwen3.5-9b", "ollama", records)
    languages = {"G1": "en", "X1": "en", "G2": "uk", "X2": "sv"}

    row = run_metrics(run_dir, load_config(), golden_dir, languages)

    assert row["per_language"] == {
        "en": {"docs": 2, "valid": 1, "golden_docs": 1, "fields_matched": 7},
        "uk": {"docs": 1, "valid": 0, "golden_docs": 1, "fields_matched": 0},
        "sv": {"docs": 1, "valid": 1, "golden_docs": 0, "fields_matched": 0},
    }
    # most documents first, then by code
    table = format_tables([row])
    assert "| Run | en | sv | uk |\n| --- | --- | --- | --- |\n" in table
    assert "| run | 1/2 · 7/9 | 1/1 · - | 0/1 · 0/9 |" in table
