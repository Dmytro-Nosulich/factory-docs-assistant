"""The test set (S2-16): held-out questions with expected answers, for the scorers of S2-17-S2-20.

Each question has a category:
- `answerable`: the corpus answers it.
- `similar_doc`: a sibling document holds a similar fact, so citing it is wrong.
- `outdated_version`: the fact changed between versions, so citing the superseded version is wrong.
- `unanswerable`: no document states the fact; the expected status is `not_in_documents`.

An answer is described by key facts. Each fact is a list of accepted spellings, and S2-17 counts
a fact as present when one spelling is in the answer. Evidence uses the gold-set format
(`retrieval_gold.Evidence`). Traps add the documents whose citation is wrong and, optionally, the
facts that show the trap caught the model. Near misses can name terms that no chunk may contain.

`check_question` grounds the labels in the corpus: every key fact is in an evidence quote, every
forbidden fact is in a wrong document and in no evidence quote, and an outdated version is really
superseded by the evidence according to data/manifest.csv.
"""

import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from factory_docs.chunking import Chunk, render_block
from factory_docs.cleaning import CleanedDocument
from factory_docs.config import ROOT
from factory_docs.dev_questions import DEV, load_dev, overlaps
from factory_docs.retrieval_gold import (
    GOLD,
    Evidence,
    EvidenceCheck,
    check_evidence,
    data_location,
    load_gold,
    normalize,
)

TEST_SET = ROOT / "evals" / "test_set.toml"
MIN_QUESTIONS = 20


class Category(StrEnum):
    answerable = "answerable"
    similar_doc = "similar_doc"
    outdated_version = "outdated_version"
    unanswerable = "unanswerable"


TRAPS = {Category.similar_doc, Category.outdated_version}

Fact = list[str]  # accepted spellings of one fact; one of them must be present


def fact_in(text: str, fact: Fact) -> bool:
    """A fact is in a text when one of its spellings is, ignoring whitespace and case."""
    haystack = normalize(text).casefold()
    return any(normalize(spelling).casefold() in haystack for spelling in fact)


class TestQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    __test__ = False  # not a pytest test class

    id: str = Field(pattern=r"^T\d{2}$")
    question: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2}$")
    category: Category
    expected_answer: str = Field(min_length=1)  # English reference answer, for people and judge
    notes: str = ""
    key_facts: list[list[str]] = []
    forbidden_facts: list[list[str]] = []  # traps only: the sibling's or the old version's value
    wrong_docs: list[str] = []  # traps only: documents whose citation is wrong
    absent_terms: list[str] = []  # unanswerable only: no chunk may contain them
    evidence: list[Evidence] = []  # alternatives, any one of them answers it

    @model_validator(mode="after")
    def fields_match_category(self) -> "TestQuestion":
        name = self.category.value
        for field in ("key_facts", "forbidden_facts"):
            for fact in getattr(self, field):
                if not fact or any(not spelling.strip() for spelling in fact):
                    raise ValueError(f"{field}: every fact needs non-empty spellings")
        for doc_id in self.wrong_docs:
            if not re.fullmatch(r"D\d{3}", doc_id):
                raise ValueError(f"wrong_docs: {doc_id!r} is not a document ID")
        if self.category is Category.unanswerable:
            if self.evidence or self.key_facts:
                raise ValueError("an unanswerable question must have no evidence or key facts")
        else:
            if not self.evidence:
                raise ValueError(f"a {name} question needs evidence")
            if not self.key_facts:
                raise ValueError(f"a {name} question needs key facts")
            if self.absent_terms:
                raise ValueError("absent_terms are for unanswerable questions only")
        if self.category in TRAPS:
            if not self.wrong_docs:
                raise ValueError(f"a {name} question needs wrong_docs")
            shared = sorted({e.doc_id for e in self.evidence} & set(self.wrong_docs))
            if shared:
                raise ValueError(f"wrong_docs are also evidence: {', '.join(shared)}")
        elif self.wrong_docs or self.forbidden_facts:
            raise ValueError("wrong_docs and forbidden_facts are for trap questions only")
        return self

    @property
    def answerable(self) -> bool:
        return self.category is not Category.unanswerable


class TestSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    __test__ = False

    question: list[TestQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "TestSet":
        ids = [q.id for q in self.question]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question id: {', '.join(duplicates)}")
        return self


def load_test_set(
    path: Path = TEST_SET, dev: Path | None = DEV, gold: Path | None = GOLD
) -> list[TestQuestion]:
    """The questions of a test file; raises ValueError naming every problem.

    dev, gold: question files whose questions must not reappear here (None: no check).
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name}: not valid TOML: {error}") from error
    try:
        questions = TestSet.model_validate(data).question
    except ValidationError as error:
        problems = []
        for err in error.errors():
            loc = data_location(data, err["loc"])
            problems.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(problems)) from error
    others = []
    if dev is not None:
        others.append((dev, [q.question for q in load_dev(dev, gold=None)]))
    if gold is not None:
        others.append((gold, [q.question for q in load_gold(gold)]))
    for other, texts in others:
        shared = overlaps(questions, texts)  # works on anything with .id and .question
        if shared:
            raise ValueError(
                f"{path.name}: also in {other.name}, so not held out: " + ", ".join(shared)
            )
    return questions


def coverage_problems(questions: Sequence[TestQuestion], minimum: int = MIN_QUESTIONS) -> list[str]:
    """What the set as a whole lacks: enough questions and every category."""
    problems = []
    if len(questions) < minimum:
        problems.append(f"{len(questions)} questions, at least {minimum} needed")
    counts = Counter(q.category for q in questions)
    missing = [category.value for category in Category if not counts[category]]
    if missing:
        problems.append(f"no question in category: {', '.join(missing)}")
    return problems


def superseded_by(manifest_rows: Sequence[Mapping[str, str]]) -> dict[str, set[str]]:
    """Document ID -> every document that supersedes it, directly or through a chain."""
    newer: dict[str, set[str]] = {}
    for row in manifest_rows:
        if row.get("supersedes"):
            newer.setdefault(row["supersedes"], set()).add(row["id"])
    result: dict[str, set[str]] = {}
    for doc_id in newer:
        seen: set[str] = set()
        todo = list(newer[doc_id])
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo += newer.get(current, set())
        result[doc_id] = seen
    return result


def document_text(document: CleanedDocument) -> str:
    return "\n".join(render_block(block) for block in document.blocks)


@dataclass(frozen=True)
class QuestionCheck:
    question_id: str
    evidence: list[EvidenceCheck]
    problems: list[str]  # problems beyond the evidence checks

    @property
    def ok(self) -> bool:
        return not self.problems and not any(check.problems for check in self.evidence)


def check_question(
    question: TestQuestion,
    documents: Mapping[str, CleanedDocument],
    collections: Mapping[str, Sequence[Chunk]],
    newer: Mapping[str, set[str]],
) -> QuestionCheck:
    """Check a question's labels against the corpus.

    documents: document ID -> cleaned document, for every document in the RAG index.
    collections: collection name -> its chunks.
    newer: document ID -> the documents that supersede it (`superseded_by`).
    """
    problems: list[str] = []
    evidence = check_evidence(question, documents, collections)  # type: ignore[arg-type]
    quotes = [e.quote for e in question.evidence]

    for fact in question.key_facts:
        if not any(fact_in(quote, fact) for quote in quotes):
            problems.append(f"key fact {fact} is in no evidence quote")

    for doc_id in question.wrong_docs:
        if doc_id not in documents:
            problems.append(f"wrong doc {doc_id} is not in the RAG index")
    wrong_texts = [document_text(documents[d]) for d in question.wrong_docs if d in documents]
    for fact in question.forbidden_facts:
        if not any(fact_in(text, fact) for text in wrong_texts):
            problems.append(f"forbidden fact {fact} is in no wrong doc")
        if any(fact_in(quote, fact) for quote in quotes):
            problems.append(f"forbidden fact {fact} is in an evidence quote")
        for key in question.key_facts:
            if fact == key:
                problems.append(f"forbidden fact {fact} is also a key fact")

    if question.category is Category.outdated_version:
        evidence_docs = {e.doc_id for e in question.evidence}
        for doc_id in question.wrong_docs:
            if not newer.get(doc_id, set()) & evidence_docs:
                problems.append(f"wrong doc {doc_id} is not superseded by an evidence document")

    for term in question.absent_terms:
        found = sorted(
            {
                chunk.doc_id
                for chunks in collections.values()
                for chunk in chunks
                if fact_in(chunk.text, [term])
            }
        )
        if found:
            problems.append(f"absent term {term!r} is in {', '.join(found)}")

    return QuestionCheck(question.id, evidence, problems)
