"""The retrieval gold set (S2-09, decision 16): questions with the evidence that answers them.

Evidence is a document, a section, a page and a short quote, not a chunk ID, so one gold set
scores every chunking strategy. A retrieved chunk is a hit for a piece of evidence when it comes
from that document and contains the quote (`is_hit`); S2-10 uses the same rule.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from factory_docs.chunking import Chunk, render_block
from factory_docs.cleaning import CleanedDocument
from factory_docs.config import ROOT

GOLD = ROOT / "evals" / "retrieval_gold.toml"
CLEANED = ROOT / "data" / "cleaned"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^D\d{3}$")
    section: str = Field(min_length=1)  # for people; parsers do not always find the heading
    page: int | None = Field(default=None, ge=1)  # PDF only
    quote: str = Field(min_length=1)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^R\d{2}$")
    question: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2}$")
    notes: str = ""
    evidence: list[Evidence] = Field(min_length=1)  # alternatives: any one of them is a hit


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "GoldSet":
        ids = [q.id for q in self.question]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question id: {', '.join(duplicates)}")
        return self


def normalize(text: str) -> str:
    """Collapse whitespace to single spaces: PDF line breaks do not matter, all else does."""
    return " ".join(text.split())


def contains_quote(text: str, quote: str) -> bool:
    return normalize(quote) in normalize(text)


def is_hit(chunk: Chunk, evidence: Evidence) -> bool:
    return chunk.doc_id == evidence.doc_id and contains_quote(chunk.text, evidence.quote)


def load_gold(path: Path = GOLD) -> list[Question]:
    """The questions of a gold file; raises ValueError naming every schema problem."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name}: not valid TOML: {error}") from error
    try:
        return GoldSet.model_validate(data).question
    except ValidationError as error:
        problems = []
        for err in error.errors():
            loc = data_location(data, err["loc"])
            problems.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(problems)) from error


def data_location(data: dict, loc: tuple[str | int, ...]) -> str:
    """'R05 evidence 1 page' instead of 'question.4.evidence.0.page'."""
    if len(loc) < 2 or loc[0] != "question" or not isinstance(loc[1], int):
        return ".".join(str(part) for part in loc)
    try:
        label = str(data["question"][loc[1]].get("id", f"question {loc[1] + 1}"))
    except (IndexError, AttributeError):
        label = f"question {loc[1] + 1}"
    rest = list(loc[2:])
    if len(rest) >= 2 and rest[0] == "evidence" and isinstance(rest[1], int):
        rest = [f"evidence {rest[1] + 1}", *rest[2:]]
    return " ".join([label, *(str(part) for part in rest)])


def quote_pages(document: CleanedDocument, quote: str) -> list[int | None] | None:
    """Pages of the blocks that contain the quote (None for formats without pages); None if none."""
    pages = [block.page for block in document.blocks if contains_quote(render_block(block), quote)]
    return pages or None


@dataclass(frozen=True)
class EvidenceCheck:
    question_id: str
    number: int  # 1-based position among the question's evidence
    evidence: Evidence
    hits: dict[str, list[str]]  # collection: IDs of the chunks that contain the quote
    problems: list[str]


def check_evidence(
    question: Question,
    documents: Mapping[str, CleanedDocument],
    collections: Mapping[str, Sequence[Chunk]],
) -> list[EvidenceCheck]:
    """Check every piece of evidence of a question against the cleaned text and the chunks.

    documents maps document ID -> cleaned document, for every document in the RAG index.
    collections maps collection name -> its chunks.
    """
    checks = []
    for number, evidence in enumerate(question.evidence, start=1):
        problems: list[str] = []
        hits = {
            name: [chunk.chunk_id for chunk in chunks if is_hit(chunk, evidence)]
            for name, chunks in collections.items()
        }
        document = documents.get(evidence.doc_id)
        if document is None:
            problems.append(f"{evidence.doc_id} is not in the RAG index")
        else:
            problems += check_quote(document, evidence)
            if not problems:
                problems += [
                    f"quote is in no `{name}` chunk (split across a chunk boundary?)"
                    for name, ids in hits.items()
                    if not ids
                ]
        checks.append(EvidenceCheck(question.id, number, evidence, hits, problems))
    return checks


def check_quote(document: CleanedDocument, evidence: Evidence) -> list[str]:
    pages = quote_pages(document, evidence.quote)
    if pages is None:
        return [f"quote not found in {evidence.doc_id} (it must sit in one block)"]
    if document.format != "pdf":
        if evidence.page is not None:
            return [f"{evidence.doc_id} is {document.format}, which has no pages: remove `page`"]
        return []
    if evidence.page is None:
        return [f"{evidence.doc_id} is a PDF: `page` is required (quote found on {fmt(pages)})"]
    if evidence.page not in pages:
        return [f"quote is on page {fmt(pages)} of {evidence.doc_id}, not page {evidence.page}"]
    return []


def fmt(pages: list[int | None]) -> str:
    return ", ".join(str(page) for page in sorted({p for p in pages if p is not None}))


def load_documents(doc_ids: set[str], cleaned: Path = CLEANED) -> dict[str, CleanedDocument]:
    """The cleaned documents among doc_ids that have a cleaned file."""
    documents = {}
    for doc_id in sorted(doc_ids):
        path = cleaned / f"{doc_id}.json"
        if path.exists():
            documents[doc_id] = CleanedDocument.model_validate_json(
                path.read_text(encoding="utf-8")
            )
    return documents
