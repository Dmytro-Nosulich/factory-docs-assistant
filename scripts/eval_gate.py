"""Fail when an eval result is below its `[eval_gate]` threshold (S2-20).

The last step of `make run-rag`, and so of the GitHub Actions workflow
(.github/workflows/evals.yml).
Checks evals/results/retrieval.json and the latest run file in evals/results/runs/; the run must
have been made at the checked-out commit (HEAD), unless a run file is named. Prints a table, adds
it to the GitHub job summary when GITHUB_STEP_SUMMARY is set, and exits 1 when a check fails.
No model calls. Rules in docs/METRICS.md.
Usage: PYTHONPATH=. uv run python scripts/eval_gate.py [--run RUN_FILE]
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from evals.gate import GateError, check_fresh, evaluate, table
from factory_docs.config import ROOT, load_config

RETRIEVAL = ROOT / "evals" / "results" / "retrieval.json"
RUNS = ROOT / "evals" / "results" / "runs"


def head_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def latest_run() -> Path:
    runs = sorted(RUNS.glob("*.json"))
    if not runs:
        raise GateError(f"no run file in {RUNS.relative_to(ROOT)}; run `make eval`")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, help="check this run file (no HEAD check)")
    args = parser.parse_args()
    settings = load_config().eval_gate
    try:
        if args.run:
            path = args.run if args.run.is_absolute() else Path.cwd() / args.run
            if not path.exists():
                raise GateError(f"no run file {args.run}")
            run = json.loads(path.read_text(encoding="utf-8"))
        else:
            path = latest_run()
            run = json.loads(path.read_text(encoding="utf-8"))
            check_fresh(run, head_commit())
        if not RETRIEVAL.exists():
            raise GateError(f"no {RETRIEVAL.relative_to(ROOT)}; run `make eval-retrieval`")
        retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
        checks = evaluate(retrieval, run, settings)
    except GateError as e:
        raise SystemExit(f"Eval gate: FAIL, {e}") from e
    failed = [c for c in checks if not c.ok]
    verdict = f"FAIL, {len(failed)} of {len(checks)} checks below threshold" if failed else "OK"
    source = path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path
    lines = [
        f"## Eval gate: {verdict}",
        "",
        f"Run `{source}` and `{RETRIEVAL.relative_to(ROOT)}`; thresholds in config.toml "
        "[eval_gate].",
        "",
        *table(checks),
    ]
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
