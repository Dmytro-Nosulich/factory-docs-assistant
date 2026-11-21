"""Answer schema for questions over the indexed documents (RAG), S2-11.

The model answers with claims, and every claim cites the chunks it comes from: the chunk ID
from the context plus a short quote copied from that chunk. Code maps a chunk ID to its
document, section and page, so the model never has to copy those. A status says whether the
context answered the question fully, partly or not at all.

Rules that a strict JSON schema cannot express (a list that must not be empty, fields that
depend on the status) are checked here in validators, as in stage 1 (`docs/ISSUES.md` I-005).
Whether a cited chunk was retrieved and whether its quote is in it needs the retrieved chunks,
so the answer pipeline checks that (S2-13).
"""

import json
from collections.abc import Iterable
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from factory_docs.chunking import Chunk
from factory_docs.schema import Strict

CHUNK_ID = r"^D\d{3}-[sf]-\d{4}$"  # D033-s-0012: document, strategy (structure/fixed), position


class AnswerStatus(StrEnum):
    answered = "answered"
    partial = "partial"
    not_in_documents = "not_in_documents"


class Citation(Strict):
    chunk_id: str = Field(
        pattern=CHUNK_ID, description="ID of a context chunk that supports the claim."
    )
    quote: str = Field(
        description="A short passage copied exactly from that chunk, in the chunk's language."
    )

    @model_validator(mode="after")
    def check_quote(self) -> Self:
        if not self.quote.strip():
            raise ValueError("quote must not be empty")
        return self


class Claim(Strict):
    text: str = Field(description="One statement that answers the question, in its language.")
    citations: list[Citation] = Field(description="At least one chunk that supports the claim.")

    @model_validator(mode="after")
    def check_claim(self) -> Self:
        if not self.text.strip():
            raise ValueError("claim text must not be empty")
        if not self.citations:
            raise ValueError("a claim needs at least one citation")
        return self


class Answer(Strict):
    status: AnswerStatus = Field(
        description="answered: the context answers every part of the question; partial: only "
        "some parts; not_in_documents: the context does not answer it."
    )
    claims: list[Claim] = Field(
        description="The answer as separate claims, each with its citations; empty for "
        "not_in_documents."
    )
    missing: str | None = Field(
        description="What the question asks that the context does not give, in the question's "
        "language; null when the status is answered."
    )

    @model_validator(mode="after")
    def check_status(self) -> Self:
        has_missing = bool(self.missing and self.missing.strip())
        if self.status is AnswerStatus.not_in_documents:
            if self.claims:
                raise ValueError("not_in_documents must have no claims")
        elif not self.claims:
            raise ValueError(f"{self.status.value} needs at least one claim")
        if self.status is AnswerStatus.answered:
            if self.missing is not None:
                raise ValueError("answered must have missing = null")
        elif not has_missing:
            raise ValueError(f"{self.status.value} must say what is missing")
        return self

    def cited_chunk_ids(self) -> set[str]:
        return {c.chunk_id for claim in self.claims for c in claim.citations}


def answer_schema_text() -> str:
    """The JSON Schema of Answer as text, for the `{{schema}}` prompt placeholder."""
    return json.dumps(Answer.model_json_schema(), indent=2, ensure_ascii=False)


def attribute(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("\n", " ")


def render_context(chunks: Iterable[Chunk]) -> str:
    """The retrieved chunks as the `{{context}}` of the answer prompt, one `<chunk>` each.

    The chunk text is left as it is, so a quote the model copies from it can be found in the
    chunk again. `page` is left out for formats without pages (HTML, DOCX, Markdown), and
    `supersedes` (the ID of the older version this document replaces) when there is none.
    """
    blocks = []
    for chunk in chunks:
        attrs = {
            "id": chunk.chunk_id,
            "document": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section,
            "date": chunk.version_or_date,
        }
        if chunk.supersedes:
            attrs["supersedes"] = chunk.supersedes
        if chunk.page_start is not None:
            pages = {chunk.page_start, chunk.page_end or chunk.page_start}
            attrs["page"] = "-".join(str(p) for p in sorted(pages))
        head = " ".join(f'{key}="{attribute(value)}"' for key, value in attrs.items())
        blocks.append(f"<chunk {head}>\n{chunk.text}\n</chunk>")
    return "\n\n".join(blocks)
