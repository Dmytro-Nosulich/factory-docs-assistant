"""Check every data/golden/<id>.json: valid JSON, no TODO left, valid against the schema.

Usage: uv run python scripts/check_golden.py
Exits with status 1 if any file has a problem.
"""

import sys
from pathlib import Path

from factory_docs.golden import check_golden

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden"


def main() -> None:
    files = sorted(GOLDEN.glob("*.json"))
    failed = 0
    for path in files:
        problems = check_golden(path)
        if problems:
            failed += 1
            print(f"{path.stem}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  {problem}")
        else:
            print(f"{path.stem}: OK")
    print(f"\n{len(files) - failed} of {len(files)} golden files OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
