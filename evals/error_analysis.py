"""The facts behind the stage 2 error analysis (S2-21). Rules: docs/METRICS.md.

Works on a saved `make eval` run file and, optionally, on the S2-19 calibration answers (the same
test questions answered with the score gate off). No model calls. A question is selected when it
failed the S2-17 pass or the quality judge did not call it correct. For each one it reports
whether the answer was in front of the model: the context ranks whose chunk holds an evidence
quote (the gold-set hit rule `is_hit`), and which documents filled the ranks before it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from evals.judges import QUALITY, QualityVerdict, quality_verdict
from evals.scorers import claim_text, doc_of, refused
from factory_docs.chunking import Chunk
from factory_docs.language import answer_text, detect
from factory_docs.retrieval_gold import is_hit
from factory_docs.test_set import TestQuestion, fact_in


def selected(row: Mapping) -> bool:
    """Failed the deterministic pass, or not judged correct (incl. not judged at all)."""
    return not row["score"]["passed"] or quality_verdict(row[QUALITY]) != QualityVerdict.correct


def evidence_ranks(
    question: TestQuestion, sent: Sequence[str], chunks: Mapping[str, Chunk]
) -> list[int]:
    """1-based context ranks of the sent chunks that hold an evidence quote."""
    return [
        rank
        for rank, chunk_id in enumerate(sent, start=1)
        if chunk_id in chunks and any(is_hit(chunks[chunk_id], e) for e in question.evidence)
    ]


def docs_before(sent: Sequence[str], ranks: Sequence[int]) -> list[str]:
    """Documents of the chunks ranked above the first evidence chunk (all, if none holds it)."""
    stop = ranks[0] - 1 if ranks else len(sent)
    return [doc_of(chunk_id) for chunk_id in sent[:stop]]


def missing_text(answer: Mapping | None) -> str:
    return (answer or {}).get("missing") or ""


@dataclass(frozen=True)
class Outcome:
    """One shown answer: its status and where the key facts appear."""

    status: str
    claims: str
    missing: str
    facts_in_claims: list[bool]
    facts_in_missing: list[bool]
    language: str | None  # detected in claims and missing (S2-14 heuristic); None: no answer


def outcome(question: TestQuestion, row: Mapping) -> Outcome:
    answer = row.get("answer")
    status = (answer or {}).get("status") or row["final_status"]
    claims, missing = claim_text(answer), missing_text(answer)
    return Outcome(
        status=status,
        claims=claims,
        missing=missing,
        facts_in_claims=[fact_in(claims, f) for f in question.key_facts],
        facts_in_missing=[fact_in(missing, f) for f in question.key_facts],
        language=detect(answer_text(answer)) if answer else None,
    )


@dataclass(frozen=True)
class ErrorCase:
    question: TestQuestion
    top_score: float | None
    threshold: float | None
    sent: list[str]
    evidence_ranks: list[int]
    docs_before: list[str]
    passed: bool
    verdict: str
    reasoning: str
    forbidden_stated: list[bool]
    shown: Outcome
    gate_off: Outcome | None = None
    gate_off_sent: list[str] = field(default_factory=list)
    gate_off_ranks: list[int] = field(default_factory=list)

    @property
    def gated(self) -> bool:
        return (
            self.threshold is not None
            and self.top_score is not None
            and self.top_score < self.threshold
        )


def collect(
    run: Mapping,
    questions: Sequence[TestQuestion],
    chunks: Mapping[str, Chunk],
    gate_off: Mapping | None = None,
) -> list[ErrorCase]:
    """Every selected question of `run`, in test-set order."""
    by_id = {q.id: q for q in questions}
    off_rows = {r["id"]: r for r in (gate_off or {}).get("questions", [])}
    threshold = run["settings"]["answer"].get("min_rerank_score")
    cases = []
    for row in run["questions"]:
        if not selected(row):
            continue
        q = by_id[row["id"]]
        ranks = evidence_ranks(q, row["sent"], chunks)
        judgement = row[QUALITY]
        off = off_rows.get(q.id)
        cases.append(
            ErrorCase(
                question=q,
                top_score=row["top_score"],
                threshold=threshold,
                sent=list(row["sent"]),
                evidence_ranks=ranks,
                docs_before=docs_before(row["sent"], ranks),
                passed=row["score"]["passed"],
                verdict=quality_verdict(judgement),
                reasoning=((judgement.get("output") or {}).get("reasoning") or ""),
                forbidden_stated=list(row["score"]["forbidden"]),
                shown=outcome(q, row),
                gate_off=outcome(q, off) if off else None,
                gate_off_sent=list(off["sent"]) if off else [],
                gate_off_ranks=evidence_ranks(q, off["sent"], chunks) if off else [],
            )
        )
    return cases


def language_mismatches(run: Mapping) -> tuple[int, list[tuple[str, str, str]]]:
    """Over every shown answer of the run: how many there are, and (id, question language,
    answer language) for each one not in the question's language."""
    shown = [r for r in run["questions"] if r.get("answer")]
    wrong = []
    for r in shown:
        found = detect(answer_text(r["answer"]))
        if found != r["language"]:
            wrong.append((r["id"], r["language"], found))
    return len(shown), wrong


def gate_off_refusals(
    gate_off: Mapping, questions: Sequence[TestQuestion]
) -> tuple[list[str], int, list[str], int]:
    """Refusals among the gate-off answers: (unanswerable ids refused, unanswerable total,
    other ids refused, other total). Refused as in S2-17 (`scorers.refused`)."""
    category = {q.id: q.answerable for q in questions}
    rows = [r for r in gate_off.get("questions", []) if r["id"] in category]
    unanswerable = [r for r in rows if not category[r["id"]]]
    others = [r for r in rows if category[r["id"]]]
    return (
        [r["id"] for r in unanswerable if refused(r)],
        len(unanswerable),
        [r["id"] for r in others if refused(r)],
        len(others),
    )


# --- rendering -----------------------------------------------------------------------------


def flags(values: Sequence[bool]) -> str:
    return f"{sum(values)}/{len(values)}" if values else "-"


def score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def one_line(text: str) -> str:
    return " ".join(text.split()) or "-"


def ranks_text(ranks: Sequence[int], n: int) -> str:
    return (", ".join(map(str, ranks)) if ranks else "none") + f" (of {n} sent)"


def outcome_lines(label: str, o: Outcome) -> list[str]:
    return [
        f"- {label}: {o.status}",
        f"  - claims: {one_line(o.claims)}",
        f"  - missing: {one_line(o.missing)}",
        f"  - key facts in claims {flags(o.facts_in_claims)}, in missing "
        f"{flags(o.facts_in_missing)}; answer language {o.language or '-'}",
    ]


def case_lines(c: ErrorCase) -> list[str]:
    q = c.question
    gate = "-" if c.threshold is None else f"{c.threshold:.2f}"
    lines = [
        f"## {q.id} ({q.category.value}, {q.language})",
        f"- question: {q.question}",
        f"- expected: {q.expected_answer}",
        f"- evidence documents: {', '.join(e.doc_id for e in q.evidence) or '-'}",
        f"- top score {score(c.top_score)}, gate {gate}: "
        + ("below the gate, no model call" if c.gated else "reaches the model"),
        f"- evidence chunk at context rank {ranks_text(c.evidence_ranks, len(c.sent))}",
        f"- documents ranked above it: {', '.join(c.docs_before) or '-'}",
        f"- deterministic pass: {'yes' if c.passed else 'no'}; forbidden facts stated "
        f"{flags(c.forbidden_stated)}",
        f"- judge: {c.verdict}: {one_line(c.reasoning)}",
        *outcome_lines("shown answer", c.shown),
    ]
    if c.gate_off is not None:
        lines.append(
            f"- gate off (calibration answers): evidence at context rank "
            f"{ranks_text(c.gate_off_ranks, len(c.gate_off_sent))}"
        )
        lines += outcome_lines("gate-off answer", c.gate_off)
    return lines


def summary_table(cases: Sequence[ErrorCase]) -> list[str]:
    lines = [
        "| ID | Category | Lang | Top score | Gated | Evidence rank | Shown | Answer lang "
        "| Passed | Judge | Gate off |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        off = "-" if c.gate_off is None else c.gate_off.status
        lines.append(
            f"| {c.question.id} | {c.question.category.value} | {c.question.language} "
            f"| {score(c.top_score)} | {'yes' if c.gated else 'no'} "
            f"| {', '.join(map(str, c.evidence_ranks)) or 'none'} | {c.shown.status} "
            f"| {c.shown.language or '-'} | {'yes' if c.passed else 'no'} | {c.verdict} | {off} |"
        )
    return lines


def language_line(run: Mapping) -> str:
    n, wrong = language_mismatches(run)
    listed = "; ".join(f"{i} ({q} question, {a} answer)" for i, q, a in wrong) or "none"
    return (
        f"Language over every shown answer of the run (S2-14 heuristic): "
        f"{n - len(wrong)}/{n} in the question's language; not: {listed}."
    )


def gate_off_line(gate_off: Mapping, questions: Sequence[TestQuestion]) -> str:
    u_ids, u_total, o_ids, o_total = gate_off_refusals(gate_off, questions)
    return (
        f"Gate-off answers, every question: unanswerable refused {len(u_ids)}/{u_total}; "
        f"answerable and trap refused {len(o_ids)}/{o_total} ({', '.join(o_ids) or 'none'})."
    )


def format_cases(
    cases: Sequence[ErrorCase],
    run: Mapping,
    run_path: str,
    gate_off_path: str | None,
    gate_off_line_text: str | None = None,
) -> str:
    head = [
        f"# Test-set errors in {run_path}",
        "",
        "Selected: failed the S2-17 pass, or the quality judge did not say correct. Evidence "
        "rank: positions in the context whose chunk holds an evidence quote (`is_hit`). "
        + (
            f"Gate off: the same question in {gate_off_path} (S2-19, score gate off)."
            if gate_off_path
            else "No gate-off answers given."
        ),
        "",
        f"{len(cases)} of {len(run['questions'])} questions.",
        "",
        language_line(run),
        "",
        *([gate_off_line_text, ""] if gate_off_line_text else []),
        *summary_table(cases),
    ]
    body = [line for c in cases for line in ("", *case_lines(c))]
    return "\n".join(head + body)
