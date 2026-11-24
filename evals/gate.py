"""CI thresholds (S2-20): does a set of eval results clear `[eval_gate]`? Rules: docs/METRICS.md.

Reads two results that `make run` writes just before the gate: the retrieval results
(`evals/results/retrieval.json`, `make eval-retrieval`) and a test-set run file
(`evals/results/runs/<time>.json`, `make eval`). Every check is a rate with its count, compared
with a minimum, except the not-judged answers, a count compared with a maximum. A rate with
nothing to count (0 of 0) fails: a gate that cannot measure does not pass.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evals.answer_eval import summarize
from evals.judges import NOT_JUDGED, QualityVerdict, SupportVerdict
from factory_docs.config import EvalGateSettings

EPSILON = 1e-9  # so 15/24 >= 0.625 is never lost to float rounding


class GateError(Exception):
    """The results cannot be checked: a missing setup, a run without judges, a stale run."""


@dataclass(frozen=True)
class Check:
    name: str
    value: str  # as shown, e.g. "18/24 (75.0%)"
    threshold: str  # e.g. ">= 62.5%"
    ok: bool


def percent(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def at_least(name: str, part: Mapping, minimum: float) -> Check:
    """part: {n, total, rate} as `factory_docs.refusal_eval.fraction` gives it."""
    rate = part["rate"]
    value = f"{part['n']}/{part['total']}" + (f" ({percent(rate)})" if rate is not None else "")
    ok = rate is not None and rate >= minimum - EPSILON
    return Check(name, value, f">= {percent(minimum)}", ok)


def at_most(name: str, n: int, maximum: int) -> Check:
    return Check(name, str(n), f"<= {maximum}", n <= maximum)


def retrieval_checks(retrieval: Mapping, settings: EvalGateSettings) -> list[Check]:
    runs = retrieval.get("runs") or {}
    collection, setup = settings.retrieval_collection, settings.retrieval_setup
    if setup not in runs.get(collection, {}):
        raise GateError(
            f"no {setup} results for collection {collection} in the retrieval results; "
            "run `make eval-retrieval`"
        )
    summary = runs[collection][setup]["summary"]
    total = summary["questions"]
    label = f"retrieval {collection} {setup}"
    mrr = Check(
        f"{label}: MRR@10",
        f"{summary['mrr']:.3f}",
        f">= {settings.min_mrr:.3f}",
        summary["mrr"] >= settings.min_mrr - EPSILON,
    )
    recall = {"n": summary["hits"]["5"], "total": total, "rate": summary["recall"]["5"]}
    return [mrr, at_least(f"{label}: recall@5", recall, settings.min_recall_at_5)]


def answer_checks(run: Mapping, settings: EvalGateSettings) -> list[Check]:
    if run["settings"].get("judge") is None:
        raise GateError("the run has no judgements (the judges never ran); run `make eval`")
    overall = summarize(run)["overall"]
    scores, quality, faith = overall["deterministic"], overall["quality"], overall["faithfulness"]
    return [
        at_least("test set: passed", scores["passed"], settings.min_passed),
        at_least(
            "test set: unanswerable refused",
            scores["unanswerable_refused"],
            settings.min_unanswerable_refused,
        ),
        at_least("judge: correct", quality[QualityVerdict.correct], settings.min_judge_correct),
        at_least(
            "judge: claims supported",
            faith[f"claims_{SupportVerdict.supported}"],
            settings.min_claims_supported,
        ),
        at_most("judge: quality not judged", quality[NOT_JUDGED]["n"], settings.max_not_judged),
        at_most(
            "judge: faithfulness not judged",
            faith["answers_not_judged"]["n"],
            settings.max_not_judged,
        ),
    ]


def check_fresh(run: Mapping, head: str | None) -> None:
    """The run must have been made at the checked-out commit, so a `make eval` that failed
    cannot leave the gate reading an older committed run."""
    commit = (run.get("git") or {}).get("commit")
    if head is None or commit != head:
        raise GateError(
            f"the latest run was made at commit {str(commit)[:12]}, not at HEAD "
            f"{str(head)[:12]}; run `make eval`, or name a run with RUN=<file>"
        )


def evaluate(retrieval: Mapping, run: Mapping, settings: EvalGateSettings) -> list[Check]:
    return retrieval_checks(retrieval, settings) + answer_checks(run, settings)


def table(checks: Sequence[Check]) -> list[str]:
    lines = ["| Check | Value | Threshold | Result |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {c.value} | {c.threshold} | {'OK' if c.ok else 'FAIL'} |")
    return lines
