"""Check that `make run-rag` can finish before it starts the slow steps (S2-15).

Reads .env, then fails in about a second when the OpenRouter key is missing or the stage 2
budget guard (`[answer] stop_at_usd`) is already reached, instead of after ten minutes of
parsing and indexing. Rules in `factory_docs.pipeline.preflight_problems`.
Usage: uv run python scripts/preflight.py
"""

import os
import platform

from factory_docs.answer import spent_on_answers
from factory_docs.config import load_config, load_env
from factory_docs.pipeline import preflight_problems


def main() -> None:
    config = load_config()
    load_env()
    spent = spent_on_answers()
    problems = preflight_problems(config, os.environ, spent)
    if problems:
        raise SystemExit("make run-rag cannot start:\n" + "\n".join(f"  - {p}" for p in problems))
    print(
        f"Preflight OK: Python {platform.python_version()} on {platform.platform()}; key set; "
        f"stage 2 answers so far ${spent:.4f} of ${config.answer.stop_at_usd:.2f}"
    )


if __name__ == "__main__":
    main()
