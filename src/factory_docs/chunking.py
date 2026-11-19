"""Split cleaned documents into chunks with citation metadata (S2-05).

Two strategies share one way of turning blocks into text, so only the splitting differs:
- `structure`: packs whole blocks into chunks of at most MAX_CHARS, never across a top-level
  section, and closes a chunk at a heading once it holds MIN_CHARS. A chunk left shorter than
  MIN_CHARS joins a neighbour in the same top-level section when both fit. Only a block longer
  than MAX_CHARS is split: a paragraph at sentence ends, a list between items, a table between
  rows with its header repeated, code between lines.
- `fixed`: the baseline. The whole document text is cut every MAX_CHARS, neighbours share
  OVERLAP characters, and each cut moves back to whitespace so no word is cut.

Sizes are characters, not tokens: the embedding model and its tokenizer are chosen in S2-06.
1500 characters is about 375-500 tokens, inside the 512-token input of common embedding models.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from factory_docs.cleaning import CleanedDocument
from factory_docs.parsing.model import Block, Code, Heading, ListBlock, Table

MAX_CHARS = 1500
MIN_CHARS = 500  # at a heading, a chunk at least this long closes, so small sections merge
OVERLAP = 200  # fixed baseline only
SEPARATOR = "\n\n"  # between blocks
NO_SECTION = "(no section heading)"
SHORT_CHARS = 100  # the comparison counts chunks shorter than this: too little to answer from

Strategy = Literal["structure", "fixed"]
STRATEGY_CODES: dict[str, str] = {"structure": "s", "fixed": "f"}

# Where an oversized paragraph may break, best first, and how its pieces are joined again.
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
LINE_BREAK = re.compile(r"\s*\n\s*")
SPACE = re.compile(r"\s+")
BREAKS = ((SENTENCE_END, " "), (LINE_BREAK, "\n"), (SPACE, " "))
WORD = re.compile(r"\w+")


class DocumentInfo(BaseModel):
    """What the manifest knows about a document and every chunk repeats."""

    title: str
    source_url: str
    version_or_date: str
    supersedes: str | None
    language: str
    doc_type: str

    @classmethod
    def from_manifest(cls, row: dict[str, str]) -> "DocumentInfo":
        return cls(
            title=row["title"],
            source_url=row["source_url"],
            version_or_date=row["version_or_date"],
            supersedes=row["supersedes"] or None,
            language=row["language"],
            doc_type=row["doc_type"],
        )


class Chunk(BaseModel):
    chunk_id: str  # D001-s-0007: document, strategy, position
    strategy: Strategy
    doc_id: str
    index: int
    title: str
    section: str  # the section path joined with " > ", or NO_SECTION
    section_path: list[str]
    page_start: int | None  # None for formats without pages (HTML, DOCX, Markdown)
    page_end: int | None
    anchor: str | None  # a stable reference in the source, e.g. 1910.1200(j)
    version_or_date: str
    supersedes: str | None
    language: str
    doc_type: str
    format: str
    source_url: str
    content_hash: str  # the document's hash from S2-04
    chunk_hash: str  # SHA-256 of `text`, for an embedding cache later
    chars: int
    block_kinds: list[str]
    table_part: str | None  # "2/5" when an oversized table was split by rows
    text: str


@dataclass(frozen=True)
class Unit:
    """A block's text, or one part of an oversized block."""

    text: str
    block: Block
    table_part: str | None = None


@dataclass(frozen=True)
class Draft:
    """A chunk before metadata: its text and the blocks it came from."""

    text: str
    blocks: tuple[Block, ...]
    section_path: tuple[str, ...]
    table_part: str | None = None
    start: int | None = None  # offsets in the document text (fixed strategy only)
    end: int | None = None


# Blocks as text
def cell_text(cell: str) -> str:
    return " ".join(cell.split()).replace("|", "\\|")


def table_lines(table: Table) -> tuple[list[str], list[str]]:
    """Return (header lines, body lines) as Markdown pipe rows, so values stay by their column."""
    rows = ["| " + " | ".join(cell_text(cell) for cell in row) + " |" for row in table.rows]
    header = rows[: table.header_rows]
    if header:
        header.append("| " + " | ".join("---" for _ in table.rows[0]) + " |")
    if table.caption:
        header.insert(0, " ".join(table.caption.split()))
    return header, rows[table.header_rows :]


def list_lines(block: ListBlock) -> list[str]:
    """One line per item, indented by depth; ordered lists are numbered per depth."""
    lines: list[str] = []
    numbers: dict[int, int] = {}
    for item in block.items:
        numbers = {depth: n for depth, n in numbers.items() if depth <= item.depth}
        numbers[item.depth] = numbers.get(item.depth, 0) + 1
        marker = f"{numbers[item.depth]}." if block.ordered else "-"
        lines.append("  " * item.depth + f"{marker} {item.text.strip()}")
    return lines


def code_fence(block: Code) -> str:
    return f"```{block.language or ''}"


def render_block(block: Block) -> str:
    if isinstance(block, ListBlock):
        return "\n".join(list_lines(block))
    if isinstance(block, Table):
        header, body = table_lines(block)
        return "\n".join(header + body)
    if isinstance(block, Code):
        return f"{code_fence(block)}\n{block.text}\n```"
    return block.text.strip()


# Splitting what is too big
def pack(pieces: Iterable[str], limit: int, joiner: str, level: int = 0) -> list[str]:
    """Join pieces greedily into parts of at most `limit` characters; split longer pieces."""
    parts: list[str] = []
    current: str | None = None
    for piece in pieces:
        for part in [piece] if len(piece) <= limit else split_text(piece, limit, level):
            if current is not None and len(current) + len(joiner) + len(part) <= limit:
                current += joiner + part
            else:
                if current is not None:
                    parts.append(current)
                current = part
    if current is not None:
        parts.append(current)
    return [part for part in parts if part.strip()]


def split_text(text: str, limit: int, level: int = 0) -> list[str]:
    """Split text at sentence ends, then line breaks, then spaces; cut only an endless word."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    if level == len(BREAKS):
        return [text[i : i + limit] for i in range(0, len(text), limit)]
    pattern, joiner = BREAKS[level]
    pieces = pattern.split(text)
    if len(pieces) == 1:
        return split_text(text, limit, level + 1)
    return pack(pieces, limit, joiner, level + 1)


def table_parts(table: Table, limit: int) -> list[str]:
    """Split a table between rows; every part repeats the header, unless the header is huge."""
    header, body = table_lines(table)
    head = "\n".join(header)
    if len(head) > limit // 2:
        header, body, head = [], header + body, ""
    room = limit - (len(head) + 1 if head else 0)
    return [f"{head}\n{part}" if head else part for part in pack(body, room, "\n")]


def code_parts(block: Code, limit: int) -> list[str]:
    fence = code_fence(block)
    room = limit - len(fence) - len("\n\n```")
    return [f"{fence}\n{part}\n```" for part in pack(block.text.split("\n"), room, "\n")]


def block_units(block: Block, limit: int = MAX_CHARS) -> list[Unit]:
    text = render_block(block)
    if len(text) <= limit:
        return [Unit(text, block)] if text else []
    if isinstance(block, Table):
        parts = table_parts(block, limit)
        return [Unit(part, block, f"{i}/{len(parts)}") for i, part in enumerate(parts, 1)]
    if isinstance(block, ListBlock):
        parts = pack(list_lines(block), limit, "\n")
    elif isinstance(block, Code):
        parts = code_parts(block, limit)
    else:
        parts = split_text(text, limit)
    return [Unit(part, block) for part in parts]


# Structure-aware chunks
def joined_length(units: Sequence[Unit]) -> int:
    return sum(len(unit.text) for unit in units) + len(SEPARATOR) * max(len(units) - 1, 0)


def top_section(block: Block) -> tuple[str, ...]:
    return tuple(block.section_path[:1])


def common_prefix(paths: Iterable[Sequence[str]]) -> tuple[str, ...]:
    paths = list(paths)
    prefix = tuple(paths[0]) if paths else ()
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def unique_blocks(units: Iterable[Unit]) -> tuple[Block, ...]:
    blocks: dict[int, Block] = {}
    for unit in units:
        blocks.setdefault(id(unit.block), unit.block)
    return tuple(blocks.values())


def draft_from_units(units: list[Unit]) -> Draft:
    blocks = unique_blocks(units)
    return Draft(
        text=SEPARATOR.join(unit.text for unit in units),
        blocks=blocks,
        # a chunk that covers 1.1 to 1.4 cites section 1, the part all its blocks share
        section_path=common_prefix(block.section_path for block in blocks),
        table_part=next((unit.table_part for unit in units if unit.table_part), None),
    )


def closes_before(current: list[Unit], unit: Unit, first: bool, limit: int, minimum: int) -> bool:
    """Whether the current chunk ends before `unit` (rules 1-3 in the module docstring)."""
    if joined_length([*current, unit]) > limit:
        return True
    if not first:
        return False  # the parts of one split block only break for size
    if top_section(unit.block) != top_section(current[-1].block):
        return True
    return isinstance(unit.block, Heading) and joined_length(current) >= minimum


def structured_drafts(
    document: CleanedDocument, limit: int = MAX_CHARS, minimum: int = MIN_CHARS
) -> list[Draft]:
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    for block in document.blocks:
        for i, unit in enumerate(block_units(block, limit)):
            if current and closes_before(current, unit, i == 0, limit, minimum):
                # a heading never ends a chunk: it moves on with the text it introduces
                carried = 0
                while (
                    carried < len(current) - 1
                    and isinstance(current[-1 - carried].block, Heading)
                    and top_section(current[-1 - carried].block) == top_section(unit.block)
                ):
                    carried += 1
                if carried and joined_length([*current[-carried:], unit]) > limit:
                    carried = 0
                split = len(current) - carried
                groups.append(current[:split])
                current = current[split:]
            current.append(unit)
    if current:
        groups.append(current)
    return [draft_from_units(group) for group in merge_small(groups, limit, minimum)]


def can_join(first: list[Unit], second: list[Unit], limit: int) -> bool:
    same_top = top_section(first[-1].block) == top_section(second[0].block)
    return same_top and joined_length([*first, *second]) <= limit


def merge_small(groups: list[list[Unit]], limit: int, minimum: int) -> list[list[Unit]]:
    """Join a chunk shorter than `minimum` to a neighbour in the same top-level section.

    The heading rule leaves orphans such as `2.3 Other hazards: None.` when a section's last
    subsection is short; joined to the chunk before it, the chunk still ends at a section end.
    """
    backward: list[list[Unit]] = []
    for group in groups:
        if backward and joined_length(group) < minimum and can_join(backward[-1], group, limit):
            backward[-1] = backward[-1] + group
        else:
            backward.append(group)
    merged: list[list[Unit]] = []  # what could not join the chunk before tries the one after
    for group in backward:
        if merged and joined_length(merged[-1]) < minimum and can_join(merged[-1], group, limit):
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)
    return merged


# Fixed-size baseline
def render_document(document: CleanedDocument) -> tuple[str, list[tuple[int, int, Block]]]:
    """Return the document text and where each block's text lies in it."""
    pieces: list[str] = []
    spans: list[tuple[int, int, Block]] = []
    position = 0
    for block in document.blocks:
        text = render_block(block)
        if pieces:
            position += len(SEPARATOR)
        spans.append((position, position + len(text), block))
        pieces.append(text)
        position += len(text)
    return SEPARATOR.join(pieces), spans


def fixed_drafts(
    document: CleanedDocument, size: int = MAX_CHARS, overlap: int = OVERLAP
) -> list[Draft]:
    text, spans = render_document(document)
    drafts: list[Draft] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            cut = end
            while cut > start and not text[cut].isspace():
                cut -= 1
            if cut > start:
                end = cut  # else one word is longer than `size` and gets cut
        blocks = tuple(block for s, e, block in spans if s < end and e > start)
        chunk_text = text[start:end].strip()
        if chunk_text and blocks:
            section = tuple(blocks[0].section_path)
            drafts.append(Draft(chunk_text, blocks, section, start=start, end=end))
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1  # start the overlap at a word
        start = next_start if next_start > start else end
    return drafts


# Records
def section_label(path: Sequence[str]) -> str:
    return " > ".join(" ".join(heading.split()) for heading in path) or NO_SECTION


def to_chunks(
    drafts: Sequence[Draft], document: CleanedDocument, info: DocumentInfo, strategy: Strategy
) -> list[Chunk]:
    chunks = []
    for index, draft in enumerate(drafts):
        pages = [block.page for block in draft.blocks if block.page is not None]
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}-{STRATEGY_CODES[strategy]}-{index:04d}",
                strategy=strategy,
                doc_id=document.doc_id,
                index=index,
                title=info.title,
                section=section_label(draft.section_path),
                section_path=list(draft.section_path),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                anchor=next((block.anchor for block in draft.blocks if block.anchor), None),
                version_or_date=info.version_or_date,
                supersedes=info.supersedes,
                language=info.language,
                doc_type=info.doc_type,
                format=document.format,
                source_url=info.source_url,
                content_hash=document.content_hash,
                chunk_hash=hashlib.sha256(draft.text.encode("utf-8")).hexdigest(),
                chars=len(draft.text),
                block_kinds=list(dict.fromkeys(block.kind for block in draft.blocks)),
                table_part=draft.table_part,
                text=draft.text,
            )
        )
    return chunks


def chunk_line(chunk: Chunk) -> str:
    return json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"


def drafts_for(document: CleanedDocument, strategy: Strategy) -> list[Draft]:
    return structured_drafts(document) if strategy == "structure" else fixed_drafts(document)


# Comparison
def missing_words(document: CleanedDocument, texts: Iterable[str]) -> Counter[str]:
    """Words of the document that no chunk holds; empty when chunking lost nothing."""
    expected = Counter(w for block in document.blocks for w in WORD.findall(render_block(block)))
    found = Counter(w for text in texts for w in WORD.findall(text))
    return expected - found


@dataclass(frozen=True)
class Stats:
    chunks: int
    chars: int
    sizes: tuple[int, ...]
    one_section: int  # chunks whose blocks all lie in one top-level section
    tables: int
    tables_whole: int  # tables whose whole text is inside one chunk
    lowercase_starts: int  # chunks that start with a lowercase letter: likely mid-sentence
    short: int  # chunks shorter than SHORT_CHARS


def chunk_stats(pairs: Iterable[tuple[CleanedDocument, Sequence[Draft]]]) -> Stats:
    sizes: list[int] = []
    one_section = tables = tables_whole = lowercase = 0
    for document, drafts in pairs:
        sizes += [len(draft.text) for draft in drafts]
        one_section += sum(len({top_section(b) for b in d.blocks}) == 1 for d in drafts)
        lowercase += sum(draft.text[:1].islower() for draft in drafts)
        for block in document.blocks:
            if isinstance(block, Table):
                tables += 1
                text = render_block(block)
                tables_whole += any(
                    text in d.text for d in drafts if any(b is block for b in d.blocks)
                )
    short = sum(size < SHORT_CHARS for size in sizes)
    return Stats(
        len(sizes), sum(sizes), tuple(sizes), one_section, tables, tables_whole, lowercase, short
    )
