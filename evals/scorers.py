"""Deterministic scorers for answers to the test set (S2-17). Rules: `docs/METRICS.md`.

Each scorer compares one shown answer with one question of `evals/test_set.toml`, with no model:

- "I don't know" when expected: an unanswerable question is refused, an answerable one is not.
- Correct source cited: a citation names a chunk of an evidence document (the headline), and,
  stricter, a cited chunk holds an evidence quote (the gold-set hit rule, `is_hit`).
- Key facts present: every key fact has a spelling in the claim texts. Quotes and `missing` are
  not searched, because a quote holds the fact whenever the right chunk is cited.
- Traps: whether a wrong document is cited (reported only, user's choice) and whether a
  forbidden fact (the sibling's or the old version's value) is in the claims (a failure).

A row is `{"final_status": ..., "answer": <shown Answer dict or None>}`, the shape of the rows in
`evals/results/refusal.json`, so "refused" means what it means in S2-14 (`refusal_eval.outcome`):
`partial` is an answer, and a rejected answer, a call error, no hit or the score gate is a
refusal. A refused answer has no claims or citations, so it misses every fact and every source.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from factory_docs.chunking import Chunk
from factory_docs.refusal_eval import ANSWERED, fraction, outcome
from factory_docs.retrieval_gold import is_hit
from factory_docs.test_set import TRAPS, Category, TestQuestion, fact_in

Row = Mapping  # {"final_status": str, "answer": dict | None}


def claims(answer: Mapping | None) -> list[Mapping]:
    return list((answer or {}).get("claims") or [])


def claim_text(answer: Mapping | None) -> str:
    """The claim texts, one per line; quotes and `missing` are left out."""
    return "\n".join(claim["text"] for claim in claims(answer))


def cited_chunk_ids(answer: Mapping | None) -> list[str]:
    """Every cited chunk ID once, in citation order."""
    ids = [c["chunk_id"] for claim in claims(answer) for c in claim["citations"]]
    return list(dict.fromkeys(ids))


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("-")[0]  # D033-s-0012 -> D033


def cited_docs(answer: Mapping | None) -> set[str]:
    return {doc_of(chunk_id) for chunk_id in cited_chunk_ids(answer)}


def refused(row: Row) -> bool:
    return outcome(row) not in ANSWERED


def refusal_correct(question: TestQuestion, row: Row) -> bool:
    """Refused exactly when the question is unanswerable."""
    return refused(row) == (question.category is Category.unanswerable)


def right_source(question: TestQuestion, answer: Mapping | None) -> bool | None:
    """A citation names a chunk of an evidence document; None for unanswerable questions."""
    if not question.answerable:
        return None
    return bool(cited_docs(answer) & {e.doc_id for e in question.evidence})


def evidence_cited(
    question: TestQuestion, answer: Mapping | None, chunks: Mapping[str, Chunk]
) -> bool | None:
    """A cited chunk holds an evidence quote (`is_hit`); None for unanswerable questions.

    chunks: chunk ID -> chunk, for every chunk an answer may cite. A cited ID missing from it
    counts as no hit.
    """
    if not question.answerable:
        return None
    cited = [chunks[c] for c in cited_chunk_ids(answer) if c in chunks]
    return any(is_hit(chunk, e) for chunk in cited for e in question.evidence)


def wrong_source(question: TestQuestion, answer: Mapping | None) -> bool | None:
    """A citation names a chunk of a wrong document; None for questions that are not traps."""
    if question.category not in TRAPS:
        return None
    return bool(cited_docs(answer) & set(question.wrong_docs))


def facts_found(question: TestQuestion, answer: Mapping | None) -> list[bool]:
    """For each key fact: is one of its spellings in the claim texts?"""
    text = claim_text(answer)
    return [fact_in(text, fact) for fact in question.key_facts]


def forbidden_found(question: TestQuestion, answer: Mapping | None) -> list[bool]:
    """For each forbidden fact: is one of its spellings in the claim texts?"""
    text = claim_text(answer)
    return [fact_in(text, fact) for fact in question.forbidden_facts]


@dataclass(frozen=True)
class QuestionScore:
    question_id: str
    category: Category
    language: str
    outcome: str  # answered, partial, not_in_documents, rejected, error, ...
    refused: bool
    refusal_correct: bool
    right_source: bool | None  # None: unanswerable
    evidence_cited: bool | None  # None: unanswerable
    wrong_source: bool | None  # None: not a trap
    facts: list[bool]  # one per key fact
    forbidden: list[bool]  # one per forbidden fact
    passed: bool

    @property
    def facts_total(self) -> int:
        return len(self.facts)

    @property
    def facts_present(self) -> int:
        return sum(self.facts)

    @property
    def all_facts(self) -> bool | None:
        return all(self.facts) if self.facts else None

    @property
    def fell_for_forbidden(self) -> bool:
        return any(self.forbidden)


def score_question(question: TestQuestion, row: Row, chunks: Mapping[str, Chunk]) -> QuestionScore:
    """Every deterministic score of one answer. A refused answer shows nothing, so it is scored
    with no claims and no citations.

    passed: an unanswerable question is refused; any other is answered, cites an evidence
    document and states every key fact, and a trap also states no forbidden fact. Citing a wrong
    document is reported in `wrong_source` but does not fail the question (user's choice).
    """
    was_refused = refused(row)
    answer = None if was_refused else row["answer"]
    facts = facts_found(question, answer)
    forbidden = forbidden_found(question, answer)
    source = right_source(question, answer)
    if question.category is Category.unanswerable:
        passed = was_refused
    else:
        passed = not was_refused and bool(source) and all(facts) and not any(forbidden)
    return QuestionScore(
        question_id=question.id,
        category=question.category,
        language=question.language,
        outcome=outcome(row),
        refused=was_refused,
        refusal_correct=refusal_correct(question, row),
        right_source=source,
        evidence_cited=evidence_cited(question, answer, chunks),
        wrong_source=wrong_source(question, answer),
        facts=facts,
        forbidden=forbidden,
        passed=passed,
    )


def counts(scores: Sequence[QuestionScore]) -> dict:
    """The rates of one group of scores, each as {n, total, rate}.

    Denominators: `passed` over all questions; `unanswerable_refused` over unanswerable ones;
    `answerable_refused`, `right_source`, `evidence_cited` and `all_facts` over every other
    question (traps included; a refusal counts as a miss); `wrong_source` and `forbidden` over
    traps (a refusal falls into no trap); `forbidden` only over traps that have forbidden facts.
    """
    unanswerable = [s for s in scores if s.category is Category.unanswerable]
    answerable = [s for s in scores if s.category is not Category.unanswerable]
    traps = [s for s in scores if s.category in TRAPS]
    with_forbidden = [s for s in traps if s.forbidden]
    return {
        "passed": fraction(sum(s.passed for s in scores), len(scores)),
        "unanswerable_refused": fraction(sum(s.refused for s in unanswerable), len(unanswerable)),
        "answerable_refused": fraction(sum(s.refused for s in answerable), len(answerable)),
        "right_source": fraction(sum(bool(s.right_source) for s in answerable), len(answerable)),
        "evidence_cited": fraction(
            sum(bool(s.evidence_cited) for s in answerable), len(answerable)
        ),
        "all_facts": fraction(sum(bool(s.all_facts) for s in answerable), len(answerable)),
        "facts_present": fraction(
            sum(s.facts_present for s in answerable), sum(s.facts_total for s in answerable)
        ),
        "wrong_source": fraction(sum(bool(s.wrong_source) for s in traps), len(traps)),
        "forbidden": fraction(
            sum(s.fell_for_forbidden for s in with_forbidden), len(with_forbidden)
        ),
    }


def summarize(scores: Sequence[QuestionScore]) -> dict:
    """The rates over all scores, per category (in test-set order) and per language."""
    categories = [c for c in Category if any(s.category is c for s in scores)]
    languages = sorted({s.language for s in scores})
    return {
        "overall": counts(scores),
        "by_category": {
            c.value: counts([s for s in scores if s.category is c]) for c in categories
        },
        "by_language": {
            lang: counts([s for s in scores if s.language == lang]) for lang in languages
        },
    }
