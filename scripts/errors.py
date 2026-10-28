"""Print every golden-set miss of the comparison runs, S1-18 (the error analysis in REPORT.md).

Usage: uv run python scripts/errors.py
Uses the same runs as the comparison tables in REPORT.md and the rules in docs/METRICS.md.
Reads only saved results: no model calls, no files written.
"""

import argparse
import sys

from factory_docs.config import load_config
from factory_docs.errors import collect_errors, format_errors


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_ids, misses = collect_errors(load_config())
    if not run_ids:
        sys.exit("no complete comparison runs in results/runs/: run `make run` first")
    print(format_errors(run_ids, misses))


if __name__ == "__main__":
    main()
