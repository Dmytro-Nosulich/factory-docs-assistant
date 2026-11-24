"""Print every test-set error of a `make eval` run, for the stage 2 error analysis (S2-21).

A question is listed when it failed the S2-17 pass or the quality judge did not call it correct.
For each one: expected answer, outcome, top score against the score gate, the context ranks of
the evidence chunk, the documents ranked above it, the judge's reasoning, and the same question
answered with the gate off (the S2-19 calibration answers). Rules in docs/METRICS.md.
Reads only saved results: no model calls, no files written.
Usage: PYTHONPATH=. uv run python scripts/answer_errors.py [--run RUN_FILE] [--gate-off FILE]
  --run       a run file; default the latest in evals/results/runs/
  --gate-off  gate-off answers; default evals/results/calibration/answers.json (skipped if absent)
"""

import argparse
import json
from pathlib import Path

from evals.error_analysis import collect, format_cases, gate_off_line
from factory_docs.config import ROOT
from factory_docs.index import load_chunks
from factory_docs.test_set import load_test_set

RUNS = ROOT / "evals" / "results" / "runs"
GATE_OFF = ROOT / "evals" / "results" / "calibration" / "answers.json"
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def shown(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, help="run file (default: the latest)")
    parser.add_argument("--gate-off", type=Path, help="gate-off answers file")
    args = parser.parse_args()
    if args.run:
        run_path = resolve(args.run)
    else:
        runs = sorted(RUNS.glob("*.json"))
        if not runs:
            raise SystemExit(f"no run file in {shown(RUNS)}; run `make eval`")
        run_path = runs[-1]
    if not run_path.exists():
        raise SystemExit(f"no run file {args.run}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["settings"].get("judge") is None:
        raise SystemExit(f"{shown(run_path)} has no judgements; run `make eval`")
    gate_off_path = resolve(args.gate_off) if args.gate_off else GATE_OFF
    if args.gate_off and not gate_off_path.exists():
        raise SystemExit(f"no gate-off answers file {args.gate_off}")
    gate_off = (
        json.loads(gate_off_path.read_text(encoding="utf-8")) if gate_off_path.exists() else None
    )
    if not CHUNKS.exists():
        raise SystemExit(f"no {shown(CHUNKS)}; run `make ingest`")
    chunks = {c.chunk_id: c for c in load_chunks(CHUNKS)}
    questions = load_test_set()
    cases = collect(run, questions, chunks, gate_off)
    print(
        format_cases(
            cases,
            run,
            shown(run_path),
            shown(gate_off_path) if gate_off else None,
            gate_off_line(gate_off, questions) if gate_off else None,
        )
    )


if __name__ == "__main__":
    main()
