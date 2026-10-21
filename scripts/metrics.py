"""Print the step 2 metrics table of one or more runs, S1-11 (rules in docs/METRICS.md).

Usage: uv run python scripts/metrics.py [RUN_ID ...]
With no run IDs, every run under results/runs/. Reads only saved results: no model calls.
"""

import argparse
import sys

from factory_docs.config import load_config
from factory_docs.metrics import format_tables, run_metrics
from factory_docs.run import RUNS, SUMMARY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="directory names under results/runs/")
    run_ids = parser.parse_args().run_ids or sorted(
        p.parent.name for p in RUNS.glob(f"*/{SUMMARY}")
    )
    missing = [r for r in run_ids if not (RUNS / r / SUMMARY).exists()]
    if missing:
        sys.exit(f"no {SUMMARY} in results/runs/ for: {', '.join(missing)}")
    if not run_ids:
        sys.exit("no runs in results/runs/: run `make run` first")
    config = load_config()
    print(format_tables([run_metrics(RUNS / run_id, config) for run_id in run_ids]))


if __name__ == "__main__":
    main()
