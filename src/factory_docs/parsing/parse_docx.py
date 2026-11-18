"""DOCX to blocks by walking the document XML (S2-03).

python-docx opens the file and resolves styles, but its `paragraphs` and `tables` lists skip text
inside content controls (`w:sdt`), which the EPA templates use a lot. Text boxes are stored twice
(`mc:Choice` and `mc:Fallback`), so only the first copy is read.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.styles.style import BaseStyle

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
    rectangular,
)

MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
SKIPPED_IN_TEXT = {MC_FALLBACK, qn("w:txbxContent"), qn("w:del"), qn("w:moveFrom")}
HEADING_STYLE = re.compile(r"^heading ([1-6])$", re.IGNORECASE)


def collapse(text: str) -> str:
    return " ".join(text.split())


def paragraph_text(element) -> str:
    """Return the visible text of a paragraph, without text boxes anchored to it."""
    parts: list[str] = []

    def visit(node) -> None:
        for child in node:
            if child.tag in SKIPPED_IN_TEXT:
                continue
            if child.tag == qn("w:t"):
                parts.append(child.text or "")
            elif child.tag in (qn("w:tab"), qn("w:br"), qn("w:cr")):
                parts.append(" ")
            elif child.tag == qn("w:noBreakHyphen"):
                parts.append("-")
            else:
                visit(child)

    visit(element)
    return collapse("".join(parts))


def text_boxes(paragraph) -> Iterator:
    """Yield the text boxes anchored to a paragraph, skipping the duplicate fallback copies."""

    def visit(node) -> Iterator:
        for child in node:
            if child.tag == MC_FALLBACK:
                continue
            if child.tag == qn("w:txbxContent"):
                yield child
            else:
                yield from visit(child)

    yield from visit(paragraph)


def block_elements(parent) -> Iterator:
    """Yield paragraphs and tables in order, looking inside content controls."""
    for child in parent:
        if child.tag in (qn("w:p"), qn("w:tbl")):
            yield child
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from block_elements(content)


def children_through_sdt(parent, tag: str) -> Iterator:
    """Yield rows or cells, which a content control may also wrap."""
    for child in parent:
        if child.tag == tag:
            yield child
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from children_through_sdt(content, tag)


def cell_text(cell) -> str:
    parts: list[str] = []
    for element in block_elements(cell):
        if element.tag == qn("w:tbl"):
            parts.extend(
                cell_text(inner)
                for row in children_through_sdt(element, qn("w:tr"))
                for inner in children_through_sdt(row, qn("w:tc"))
            )
        else:
            parts.append(paragraph_text(element))
            parts.extend(cell_text(box) for box in text_boxes(element))
    return collapse(" ".join(parts))


def int_value(parent, name: str, default: int) -> int:
    node = parent.find(qn(name)) if parent is not None else None
    value = node.get(qn("w:val")) if node is not None else None
    return int(value) if value is not None else default


def table_grid(table) -> tuple[list[list[str]], int]:
    """Return complete rows, with merged cells repeated, and the number of header rows."""
    rows: list[list[str]] = []
    header_rows = 0
    counting_headers = True
    for row in children_through_sdt(table, qn("w:tr")):
        properties = row.find(qn("w:trPr"))
        is_header = properties is not None and properties.find(qn("w:tblHeader")) is not None
        counting_headers = counting_headers and is_header
        header_rows += counting_headers
        cells = [""] * int_value(properties, "w:gridBefore", 0)
        for cell in children_through_sdt(row, qn("w:tc")):
            cell_properties = cell.find(qn("w:tcPr"))
            span = int_value(cell_properties, "w:gridSpan", 1)
            merge = cell_properties.find(qn("w:vMerge")) if cell_properties is not None else None
            if merge is not None and merge.get(qn("w:val")) != "restart" and rows:
                above = rows[-1][len(cells)] if len(cells) < len(rows[-1]) else ""
                cells.extend([above] * span)
            else:
                cells.extend([cell_text(cell)] * span)
        rows.append(cells)
    return rectangular(rows), header_rows


class DocxWalker:
    def __init__(self, document) -> None:
        self.styles: dict[str, BaseStyle] = {style.style_id: style for style in document.styles}
        self.default_style = document.styles.default(WD_STYLE_TYPE.PARAGRAPH)
        self.ordered_levels = self.read_numbering(document)
        self.blocks: list[Block] = []

    @staticmethod
    def read_numbering(document) -> dict[tuple[str, int], bool]:
        """Map (numbering id, level) to whether that level is a numbered (ordered) list."""
        try:
            numbering = document.part.numbering_part.element
        except (KeyError, NotImplementedError):
            return {}
        abstract = {a.get(qn("w:abstractNumId")): a for a in numbering.findall(qn("w:abstractNum"))}
        ordered: dict[tuple[str, int], bool] = {}
        for num in numbering.findall(qn("w:num")):
            link = num.find(qn("w:abstractNumId"))
            definition = abstract.get(link.get(qn("w:val"))) if link is not None else None
            if definition is None:
                continue
            for level in definition.findall(qn("w:lvl")):
                number_format = level.find(qn("w:numFmt"))
                value = number_format.get(qn("w:val")) if number_format is not None else "bullet"
                key = (num.get(qn("w:numId")), int(level.get(qn("w:ilvl"))))
                ordered[key] = value not in ("bullet", "none")
        return ordered

    def style_of(self, paragraph) -> BaseStyle | None:
        properties = paragraph.find(qn("w:pPr"))
        style = properties.find(qn("w:pStyle")) if properties is not None else None
        if style is None:
            return self.default_style
        return self.styles.get(style.get(qn("w:val")), self.default_style)

    @staticmethod
    def style_chain(style: BaseStyle | None) -> Iterator[BaseStyle]:
        seen = 0
        while style is not None and seen < 20:
            yield style
            style = getattr(style, "base_style", None)
            seen += 1

    def heading_level(self, paragraph) -> int | None:
        for style in self.style_chain(self.style_of(paragraph)):
            match = HEADING_STYLE.match(style.name or "")
            if match:
                return int(match.group(1))
            if (style.name or "").casefold() == "title":
                return 1
        return None

    def zone_of(self, paragraph) -> Zone:
        style = self.style_of(paragraph)
        name = (style.name or "").casefold() if style is not None else ""
        return "toc" if name.startswith("toc ") or name == "toc heading" else "main"

    def list_info(self, paragraph) -> tuple[bool, int] | None:
        """Return (ordered, depth) when the paragraph or its style has list numbering."""
        sources = [paragraph.find(qn("w:pPr"))]
        sources += [s.element.find(qn("w:pPr")) for s in self.style_chain(self.style_of(paragraph))]
        for properties in sources:
            numbering = properties.find(qn("w:numPr")) if properties is not None else None
            if numbering is None:
                continue
            number_id = numbering.find(qn("w:numId"))
            value = number_id.get(qn("w:val")) if number_id is not None else None
            if value in (None, "0"):
                return None
            level = numbering.find(qn("w:ilvl"))
            depth = int(level.get(qn("w:val"))) if level is not None else 0
            return self.ordered_levels.get((value, depth), False), depth
        return None

    def add_list_item(self, text: str, ordered: bool, depth: int) -> None:
        last = self.blocks[-1] if self.blocks else None
        if isinstance(last, ListBlock):
            last.items.append(ListItem(text=text, depth=depth))
        else:
            self.blocks.append(ListBlock(ordered=ordered, items=[ListItem(text=text, depth=depth)]))

    def add_paragraph(self, paragraph) -> None:
        text = paragraph_text(paragraph)
        zone = self.zone_of(paragraph)
        if text:
            level = self.heading_level(paragraph)
            list_info = self.list_info(paragraph)
            if zone != "main":
                self.blocks.append(Paragraph(text=text, zone=zone))
            elif level:
                self.blocks.append(Heading(level=level, text=text))
            elif list_info:
                self.add_list_item(text, *list_info)
            else:
                self.blocks.append(Paragraph(text=text))
        for box in text_boxes(paragraph):
            self.walk(box)

    def add_table(self, table) -> None:
        rows = list(children_through_sdt(table, qn("w:tr")))
        cells = list(children_through_sdt(rows[0], qn("w:tc"))) if len(rows) == 1 else []
        if len(cells) == 1:  # a one-cell table is a frame around ordinary content
            self.walk(cells[0])
            return
        grid, header_rows = table_grid(table)
        if any(cell for row in grid for cell in row):
            self.blocks.append(Table(rows=grid, header_rows=header_rows))

    def walk(self, parent) -> None:
        for element in block_elements(parent):
            if element.tag == qn("w:tbl"):
                self.add_table(element)
            else:
                self.add_paragraph(element)


def parse_docx(doc_id: str, path: Path) -> ParsedDocument:
    document = Document(str(path))
    walker = DocxWalker(document)
    walker.walk(document.element.body)
    assign_section_paths(walker.blocks)
    return ParsedDocument(doc_id=doc_id, format="docx", blocks=walker.blocks)
