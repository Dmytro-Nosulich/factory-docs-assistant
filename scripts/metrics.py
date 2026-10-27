"""Print the step 2 metrics table of one or more runs, S1-11 (rules in docs/METRICS.md).

Usage: uv run python scripts/metrics.py [RUN_ID ...]
       uv run python scripts/metrics.py --report
With no run IDs, every run under results/runs/. Reads only saved results: no model calls.
--report rewrites the comparison tables in REPORT.md instead (S1-17, `make comparison`): the
newest complete run of each [comparison] configuration (factory_docs.report), with no
timestamp, so the same runs always give the same text.
"""

import argparse
import sys

from factory_docs.config import load_config
from factory_docs.metrics import format_tables, run_metrics
from factory_docs.report import cell_label, update_report
from factory_docs.run import RUNS, SUMMARY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="directory names under results/runs/")
    parser.add_argument(
        "--report", action="store_true", help="rewrite the comparison tables in REPORT.md"
    )
    args = parser.parse_args()
    config = load_config()
    if args.report:
        if args.run_ids:
            parser.error("--report picks its runs from [comparison] in config.toml")
        run_ids, missing = update_report(config)
        print(f"REPORT.md: comparison tables rebuilt from {len(run_ids)} runs")
        for run_id in run_ids:
            print(f"  {run_id}")
        if missing:
            print("  no complete run: " + ", ".join(cell_label(c) for c in missing))
        return
    run_ids = args.run_ids or sorted(p.parent.name for p in RUNS.glob(f"*/{SUMMARY}"))
    missing = [r for r in run_ids if not (RUNS / r / SUMMARY).exists()]
    if missing:
        sys.exit(f"no {SUMMARY} in results/runs/ for: {', '.join(missing)}")
    if not run_ids:
        sys.exit("no runs in results/runs/: run `make run` first")
    print(format_tables([run_metrics(RUNS / run_id, config) for run_id in run_ids]))


if __name__ == "__main__":
    main()
