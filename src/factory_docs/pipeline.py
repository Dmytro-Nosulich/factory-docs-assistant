"""Helpers for `make run`, the stage 2 one-command run (S2-15).

`make run` chains make targets (see the Makefile): preflight, parse, clean, chunk, index (only
when stale), the gold and dev-set checks, the evals, sample answers and the cost report. This
module holds the parts with logic in them, so they are unit-tested: what preflight checks, and
which development questions are the samples.
"""

from collections.abc import Mapping, Sequence

from factory_docs.config import Config
from factory_docs.dev_questions import DevQuestion

KEY = "OPENROUTER_API_KEY"


def preflight_problems(config: Config, env: Mapping[str, str], spent_usd: float) -> list[str]:
    """What stops `make run` before any slow step; empty when it can go ahead.

    env: the environment after `.env` is read. spent_usd: the stage 2 answer logs so far
    (`answer.spent_on_answers`), against the guard `[answer] stop_at_usd` (decision 18).
    """
    problems = []
    if not env.get(KEY, "").strip():
        problems.append(f"{KEY} is not set: copy .env.example to .env and fill it in")
    limit = config.answer.stop_at_usd
    if spent_usd >= limit:
        problems.append(
            f"stage 2 answers have cost ${spent_usd:.4f}, at or above [answer] stop_at_usd "
            f"${limit:.2f} (decision 18); raise it in config.toml or clear results/answers/"
        )
    return problems


def sample_questions(ids: Sequence[str], questions: Sequence[DevQuestion]) -> list[DevQuestion]:
    """The `[run] sample_questions`, in their configured order; unknown IDs are an error."""
    by_id = {q.id: q for q in questions}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"[run] sample_questions not in evals/dev_questions.toml: {unknown}")
    return [by_id[i] for i in ids]
