"""One structure for every document format (S2-03).

Parsers only tag content and never delete it: `zone` marks page furniture and navigation, and
S2-04 decides what to remove and reports it.
"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Zone = Literal["main", "header", "nav", "aside", "footer", "toc"]
DocumentFormat = Literal["pdf", "docx", "html", "md"]


class BlockBase(BaseModel):
    page: int | None = None  # 1-based; only PDF has real pages
    section_path: list[str] = Field(default_factory=list)
    anchor: str | None = None  # a stable reference in the source, e.g. 1910.1200(j)
    zone: Zone = "main"


class Heading(BlockBase):
    kind: Literal["heading"] = "heading"
    level: int = Field(ge=1, le=6)
    text: str


class Paragraph(BlockBase):
    kind: Literal["paragraph"] = "paragraph"
    text: str


class ListItem(BaseModel):
    text: str
    depth: int = Field(default=0, ge=0)


class ListBlock(BlockBase):
    kind: Literal["list"] = "list"
    ordered: bool = False
    items: list[ListItem]


class Table(BlockBase):
    kind: Literal["table"] = "table"
    rows: list[list[str]]
    header_rows: int = Field(default=0, ge=0)
    caption: str | None = None


class Code(BlockBase):
    kind: Literal["code"] = "code"
    text: str
    language: str | None = None


Block = Annotated[Heading | Paragraph | ListBlock | Table | Code, Field(discriminator="kind")]


class ParsedDocument(BaseModel):
    doc_id: str
    format: DocumentFormat
    blocks: list[Block]
    page_count: int | None = None  # None for formats without real pages
    pages_without_text: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def assign_section_paths(blocks: list[Block]) -> None:
    """Fill `section_path` from the headings; blocks outside the main zone do not affect it.

    A heading's own path ends with itself, so a chunk that starts at a heading can cite it.
    """
    open_headings: list[Heading] = []
    for block in blocks:
        if block.zone != "main":
            block.section_path = []
            continue
        if isinstance(block, Heading):
            while open_headings and open_headings[-1].level >= block.level:
                open_headings.pop()
            open_headings.append(block)
        block.section_path = [heading.text for heading in open_headings]


def rectangular(rows: list[list[str]]) -> list[list[str]]:
    """Pad short rows with empty cells so every row has the same number of columns."""
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def to_review_json(document: ParsedDocument) -> str:
    """Return stable, readable JSON: document facts first, then blocks with their kind first."""
    data = document.model_dump(exclude_none=True)
    blocks = data.pop("blocks")
    data["blocks"] = [{"kind": block.pop("kind"), **block} for block in blocks]
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
