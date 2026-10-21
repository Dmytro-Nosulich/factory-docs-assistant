"""Crude preview: which fields of a run's outputs equal the golden labels exactly (S1-09 bake-off).

Usage: uv run python scripts/compare_golden.py RUN_ID [RUN_ID ...]
Exact equality only (after both sides pass the schema), lists in order. This is a quick aid
for picking the local model, not the S1-11 metrics: S1-11 replaces it with agreed matching
rules (normalized strings, numeric tolerance, set match for lists).
"""

import argparse
import json
from pathlib import Path

from factory_docs.schema import SafetyDataSheet

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
RUNS = ROOT / "results" / "runs"
FIELDS = list(SafetyDataSheet.model_fields)


def load_golden(doc_id: str) -> dict | None:
    path = GOLDEN / f"{doc_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SafetyDataSheet.model_validate(data).model_dump(mode="json")


def compare_run(run_id: str) -> None:
    summary = json.loads((RUNS / run_id / "run.json").read_text(encoding="utf-8"))
    sampling = (summary["params"].get("sampling") or {}).get("name")
    print(f"\n{run_id} ({summary['model']['slug']}" + (f", {sampling})" if sampling else ")"))
    total_match = total_fields = 0
    for doc_id in summary["docs"]:
        golden = load_golden(doc_id)
        path = RUNS / run_id / f"{doc_id}.json"
        if golden is None or not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        call = record["call"] or {}
        line = f"  {doc_id} {record['status']:<7}"
        if record["status"] != "valid":  # no usable output: every field counts as missed
            print(f"{line} 0/{len(FIELDS)} fields  {record['errors'][0][:100]}")
            total_fields += len(FIELDS)
            continue
        output = record["output"]
        missed = [f for f in FIELDS if output.get(f) != golden[f]]
        total_match += len(FIELDS) - len(missed)
        total_fields += len(FIELDS)
        print(
            f"{line} {len(FIELDS) - len(missed)}/{len(FIELDS)} fields"
            f"  {call.get('latency_s', 0):6.1f} s  {call.get('prompt_tokens')} prompt tokens"
            + (f"  missed: {', '.join(missed)}" if missed else "")
        )
    if total_fields:
        print(f"  total: {total_match}/{total_fields} fields equal the golden labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="directory names under results/runs/")
    for run_id in parser.parse_args().run_ids:
        compare_run(run_id)


if __name__ == "__main__":
    main()
