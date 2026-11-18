"""PDF to blocks with layout rules (S2-03).

PDF files store positioned text, not structure, so headings, lists and tables are inferred. These
rules are heuristics. Paragraphs keep their line breaks, because S2-04 repairs hyphenated words.
"""

import logging
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from factory_docs.parsing.model import (
    Block,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    ParsedDocument,
    Table,
    Zone,
    assign_section_paths,
)
from factory_docs.pdf_text import StyledLine, extract_styled_lines, squash

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 20  # fewer visible characters than this means no usable text layer
MIN_TABLE_ROWS = 3
COLUMN_TOLERANCE = 3
MAX_HEADING_CHARS = 120
REPEAT_SHARE = 0.5  # a line on at least half of the pages is page furniture
TOP_LINES = 6
BOTTOM_LINES = 4
MIN_SDS_SECTIONS = 5
DECORATIVE_DOT_SHARE = 0.25

SDS_SECTION = re.compile(
    r"^(section|abschnitt|розділ|sekcja|avsnitt|rubrique|secci[oó]n|sezione|rubriek)\s*\d{1,2}\b",
    re.IGNORECASE,
)
NUMBERED_TOP = re.compile(r"^\d{1,2}\.\s+[^\W\d_]")
NUMBERED_SUB = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\.?\s+[^\W\d_]")
SDS_SUBSECTION_WORD = re.compile(r"^[\d.]+\s+[^\W\d_]{3,}")
BULLET = re.compile(r"^[•●▪■◦‣*·–—-]\s+(.*)$")
ORDERED = re.compile(r"^(?:\d{1,2}[.)]|[a-z]\))\s+(.*)$")
COLUMN = re.compile(r"\S+(?: {1,2}\S+)*")
DECORATIVE_DOT = re.compile(r"^(\s*)·\s*")


def collapse(text: str) -> str:
    return " ".join(text.split())


def strip_decorative_dots(pages: list[list[StyledLine]]) -> list[list[StyledLine]]:
    """Some SDS layouts start many lines with `·`; there the dot is not a list marker.

    Decided per document, because pages with long tables have fewer dotted lines.
    """
    filled = [line for lines in pages for line in lines if line.text.strip()]
    dotted = sum(line.text.lstrip().startswith("·") for line in filled)
    if not filled or dotted / len(filled) < DECORATIVE_DOT_SHARE:
        return pages
    return [
        [StyledLine(DECORATIVE_DOT.sub(r"\1", line.text), line.size, line.bold) for line in lines]
        for lines in pages
    ]


def furniture_key(text: str) -> str:
    """Compare repeated lines without their digits, so `1/12` and `2/12` count as the same."""
    return re.sub(r"\d+", "#", collapse(text)).casefold()


def find_furniture(pages: list[list[StyledLine]]) -> tuple[set[str], set[str]]:
    """Return the lines repeated at the top and at the bottom of most pages."""
    if len(pages) < 3:
        return set(), set()
    top: Counter[str] = Counter()
    bottom: Counter[str] = Counter()
    for lines in pages:
        filled = [line.text for line in lines if line.text.strip()]
        top.update({furniture_key(text) for text in filled[:TOP_LINES]})
        bottom.update({furniture_key(text) for text in filled[-BOTTOM_LINES:]})
    needed = max(2, int(len(pages) * REPEAT_SHARE))
    return (
        {key for key, count in top.items() if count >= needed},
        {key for key, count in bottom.items() if count >= needed},
    )


def is_sds_section(text: str) -> bool:
    return SDS_SECTION.match(collapse(text)) is not None


def heading_level(
    line: StyledLine, body_size: float | None, larger_sizes: list[float], structured: bool
) -> int | None:
    """Return the heading level of a line, or None when it is not a heading."""
    text = collapse(line.text)
    if not text or len(text) > MAX_HEADING_CHARS:
        return None
    if SDS_SECTION.match(text):
        return 1
    is_larger = bool(line.size and body_size and line.size >= body_size * 1.15)
    if NUMBERED_TOP.match(text) and (line.bold or is_larger):
        return 1
    sub = NUMBERED_SUB.match(text)
    # Outside SDS numbering, plain numbered lines are clauses (D029). In an SDS the font is not
    # always readable (D033), so the number counts, unless a short unit follows it (`1.5 mg/l`).
    sds_subsection = structured and SDS_SUBSECTION_WORD.match(text) is not None
    if sub and (line.bold or is_larger or sds_subsection):
        return min(sub.group(1).count(".") + 1, 3)
    if structured or len(split_columns(line.text)) > 1:
        return None  # SDS numbering is reliable; bold labels there are not headings
    if text.endswith((":", ".", ",", ";")) or " : " in text or not re.search(r"[^\W\d_]", text):
        return None
    if is_larger:
        return min(larger_sizes.index(line.size) + 1, 3) if line.size in larger_sizes else 3
    return 4 if line.bold else None


def split_columns(text: str) -> list[tuple[int, str]]:
    """Split a layout line on gaps of 3 or more spaces into (start position, cell text)."""
    return [(match.start(), match.group()) for match in COLUMN.finditer(text)]


def near(position: int, positions: set[int]) -> bool:
    return any(abs(position - other) <= COLUMN_TOLERANCE for other in positions)


def table_end(lines: list[StyledLine], start: int) -> int:
    """Return the end of an aligned-column run, or `start` when there is no table here."""
    first = split_columns(lines[start].text)
    if len(first) < 2:
        return start
    seen = {position for position, _ in first[1:]}
    rows = 1
    end = start + 1
    while end < len(lines):
        columns = split_columns(lines[end].text)
        if len(columns) >= 2 and any(near(position, seen) for position, _ in columns[1:]):
            seen.update(position for position, _ in columns[1:])
            rows += 1
        elif not (len(columns) == 1 and near(columns[0][0], seen)):
            break
        end += 1
    return end if rows >= MIN_TABLE_ROWS else start


def build_table(lines: list[StyledLine]) -> Table:
    """Cluster column positions over the whole run, then place every cell in its column."""
    parsed = [split_columns(line.text) for line in lines]
    starts = sorted(
        {position for columns in parsed if len(columns) >= 2 for position, _ in columns}
    )
    anchors: list[int] = []
    for position in starts:
        if not anchors or position - anchors[-1] > COLUMN_TOLERANCE:
            anchors.append(position)

    def column_of(position: int) -> int:
        return min(range(len(anchors)), key=lambda index: abs(anchors[index] - position))

    rows: list[list[str]] = []
    for columns in parsed:
        if len(columns) >= 2:
            rows.append([""] * len(anchors))
        for position, text in columns:
            index = column_of(position)
            rows[-1][index] = f"{rows[-1][index]} {text}".strip()
    return Table(rows=rows)


def list_marker(text: str) -> tuple[bool, str] | None:
    """Return (ordered, item text) when a line starts a list item."""
    stripped = text.strip()
    if match := BULLET.match(stripped):
        return False, match.group(1)
    if match := ORDERED.match(stripped):
        return True, match.group(1)
    return None


def indent_of(text: str) -> int:
    return len(text) - len(text.lstrip())


def build_list(lines: list[StyledLine], start: int) -> tuple[ListBlock, int]:
    ordered = list_marker(lines[start].text)[0]
    items: list[ListItem] = []
    indents: list[int] = []
    index = start
    while index < len(lines) and lines[index].text.strip():
        text = lines[index].text
        marker = list_marker(text)
        if marker:
            indent = indent_of(text)
            while indents and indents[-1] > indent + 1:
                indents.pop()
            if not indents or indent > indents[-1] + 1:
                indents.append(indent)
            items.append(ListItem(text=collapse(marker[1]), depth=len(indents) - 1))
        elif indents and indent_of(text) > indents[-1]:
            items[-1].text = f"{items[-1].text} {collapse(text)}"
        else:
            break
        index += 1
    return ListBlock(ordered=ordered, items=items, page=None), index


def parse_page(
    lines: list[StyledLine],
    page_number: int,
    top_furniture: set[str],
    bottom_furniture: set[str],
    body_size: float | None,
    larger_sizes: list[float],
    structured: bool,
) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Paragraph(text="\n".join(paragraph), page=page_number))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        text = line.text
        if not text.strip():
            flush()
            index += 1
            continue
        key = furniture_key(text)
        zone: Zone | None = (
            "header" if key in top_furniture else "footer" if key in bottom_furniture else None
        )
        if zone:
            flush()
            blocks.append(Paragraph(text=collapse(text), page=page_number, zone=zone))
            index += 1
            continue
        level = heading_level(line, body_size, larger_sizes, structured)
        if level:
            flush()
            blocks.append(Heading(level=level, text=collapse(text), page=page_number))
            index += 1
            continue
        if list_marker(text):
            flush()
            block, index = build_list(lines, index)
            block.page = page_number
            blocks.append(block)
            continue
        end = table_end(lines, index)
        if end > index:
            flush()
            table = build_table(lines[index:end])
            table.page = page_number
            blocks.append(table)
            index = end
            continue
        paragraph.append(collapse(text))
        index += 1
    flush()
    return blocks


def font_profile(pages: list[list[StyledLine]]) -> tuple[float | None, list[float]]:
    """Return the body font size and the larger sizes, biggest first."""
    sizes: Counter[float] = Counter()
    for lines in pages:
        for line in lines:
            if line.size:
                sizes[line.size] += len(squash(line.text))
    if not sizes:
        return None, []
    body = sizes.most_common(1)[0][0]
    return body, sorted((size for size in sizes if size >= body * 1.15), reverse=True)


def parse_pdf(doc_id: str, path: Path) -> ParsedDocument:
    reader = PdfReader(path)
    pages: list[list[StyledLine]] = []
    warnings: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(extract_styled_lines(page))
        except Exception as error:  # pypdf raises many types on damaged pages
            fallback = (page.extract_text() or "").splitlines()
            pages.append([StyledLine(text) for text in fallback])
            warnings.append(f"page {number}: layout reading failed ({type(error).__name__})")
    pages = strip_decorative_dots(pages)

    without_text = [
        number
        for number, lines in enumerate(pages, start=1)
        if sum(len(squash(line.text)) for line in lines) < MIN_PAGE_CHARS
    ]
    if without_text:
        message = f"{doc_id}: no text layer on pages {without_text}"
        logger.warning(message)
        warnings.append(message)

    top, bottom = find_furniture(pages)
    body_size, larger_sizes = font_profile(pages)
    section_count = sum(is_sds_section(line.text) for lines in pages for line in lines)
    structured = section_count >= MIN_SDS_SECTIONS
    blocks: list[Block] = []
    for number, lines in enumerate(pages, start=1):
        if number not in without_text:
            blocks.extend(
                parse_page(lines, number, top, bottom, body_size, larger_sizes, structured)
            )
    assign_section_paths(blocks)
    return ParsedDocument(
        doc_id=doc_id,
        format="pdf",
        blocks=blocks,
        page_count=len(pages),
        pages_without_text=without_text,
        warnings=warnings,
    )
