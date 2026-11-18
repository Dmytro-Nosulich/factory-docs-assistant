"""HTML to blocks, for full web pages (HSE) and eCFR regulation fragments (S2-03).

Navigation and footers are tagged with a zone and kept, because S2-04 removes them and reports it.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

from factory_docs.parsing.model import (
    Block,
    Code,
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

IGNORED = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "button",
    "select",
    "input",
    "textarea",
    "head",
}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTAINERS = {
    "html",
    "body",
    "main",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "fieldset",
    "figure",
    "figcaption",
    "blockquote",
    "details",
    "summary",
    "dl",
    "dt",
    "dd",
    "address",
    "center",
}
LANDMARK_ZONES: dict[str, Zone] = {
    "header": "header",
    "nav": "nav",
    "aside": "aside",
    "footer": "footer",
}
APPENDIX_NUMBER = re.compile(r"^[A-F]\.\d+(?:\.\d+)*\b")


def collapse(text: str) -> str:
    return " ".join(text.split())


def tag_text(tag: Tag) -> str:
    return collapse(tag.get_text())


def span(cell: Tag, attribute: str) -> int:
    value = str(cell.get(attribute, "1")).strip()
    return max(int(value), 1) if value.isdigit() else 1


def table_rows(table: Tag) -> tuple[list[list[str]], int]:
    """Return complete rows, with spanned cells repeated, and the number of header rows."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}  # column -> (rows still covered, text)
    header_rows = 0
    counting_headers = True
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["td", "th"], recursive=False)
        output: list[str] = []
        queue = list(cells)
        while queue or any(column >= len(output) for column in pending):
            column = len(output)
            if column in pending:
                remaining, text = pending[column]
                output.append(text)
                if remaining <= 1:
                    del pending[column]
                else:
                    pending[column] = (remaining - 1, text)
            elif queue:
                cell = queue.pop(0)
                text = tag_text(cell)
                rowspan = span(cell, "rowspan")
                for offset in range(span(cell, "colspan")):
                    if rowspan > 1:
                        pending[column + offset] = (rowspan - 1, text)
                    output.append(text)
                    if offset + 1 < span(cell, "colspan") and column + offset + 1 in pending:
                        del pending[column + offset + 1]
            else:
                output.append("")
        if not any(output):
            continue
        in_head = row.find_parent("thead") is not None
        all_header_cells = bool(cells) and all(cell.name == "th" for cell in cells)
        counting_headers = counting_headers and (in_head or all_header_cells)
        header_rows += counting_headers
        grid.append(output)
    return rectangular(grid), header_rows


def list_items(list_tag: Tag, depth: int = 0) -> list[ListItem]:
    items: list[ListItem] = []
    for item in list_tag.find_all("li", recursive=False):
        own = "".join(text for text in item.strings if text.find_parent(["ul", "ol"]) is list_tag)
        if collapse(own):
            items.append(ListItem(text=collapse(own), depth=depth))
        for nested in item.find_all(["ul", "ol"]):
            if nested.find_parent("li") is item:
                items.extend(list_items(nested, depth + 1))
    return items


class HtmlWalker:
    def __init__(self, soup: BeautifulSoup) -> None:
        self.has_main = soup.find("main") is not None
        self.is_ecfr = soup.find(attrs={"data-hierarchy-metadata": True}) is not None
        self.blocks: list[Block] = []
        self.table_title: str | None = None  # older eCFR pages put it in a div before the table

    def child_zone(self, tag: Tag, zone: Zone, inside_main: bool) -> tuple[Zone, bool]:
        if tag.name == "main":
            return "main", True
        if self.is_ecfr and "editorial-note" in (tag.get("class") or []):
            return "aside", inside_main
        landmark = LANDMARK_ZONES.get(tag.name)
        if landmark is None or (inside_main and tag.name == "header"):
            return zone, inside_main  # a <header> inside <main> only introduces the content
        # Outside <main> the default zone is `nav`, and a landmark there is more precise.
        replaceable = zone == "main" or (zone == "nav" and not inside_main and tag.name != "nav")
        return (landmark if replaceable else zone), inside_main

    def heading_level(self, tag: Tag, text: str) -> int:
        level = int(tag.name[1])
        if not self.is_ecfr:
            return level
        if tag.has_attr("data-hierarchy-metadata") or text.startswith("Appendix "):
            return 1
        if number := APPENDIX_NUMBER.match(text):
            return min(number.group().count(".") + 2, 6)
        return min(level + 1, 6)

    def add_ecfr_paragraph(self, tag: Tag, zone: Zone) -> bool:
        """Turn a top-level paragraph such as `(j) Dates` into a heading plus its body text."""
        title = tag.find("em", class_="paragraph-heading")
        if title is None or "indent-1" not in (tag.get("class") or []):
            return False
        marker = tag.find("span", class_="paragraph-hierarchy")
        label = tag_text(marker) if marker else ""
        name = tag_text(title).rstrip(" .—-:")
        anchor = tag.get("data-title")
        self.blocks.append(
            Heading(level=2, text=collapse(f"{label} {name}"), anchor=anchor, zone=zone)
        )
        body = "".join(
            text
            for text in tag.strings
            if not (marker and marker in text.parents) and title not in text.parents
        )
        body = collapse(body).lstrip(" .—-:")
        if body:
            self.blocks.append(Paragraph(text=body, anchor=anchor, zone=zone))
        return True

    def add_table(self, tag: Tag, zone: Zone) -> None:
        rows, header_rows = table_rows(tag)
        caption = tag.find("caption")
        title = tag_text(caption) if caption else self.table_title
        self.table_title = None
        if rows:
            self.blocks.append(Table(rows=rows, header_rows=header_rows, caption=title, zone=zone))

    def walk(self, node: Tag, zone: Zone, inside_main: bool) -> None:
        inline: list[str] = []

        def flush() -> None:
            text = collapse("".join(inline))
            inline.clear()
            if text:
                self.blocks.append(Paragraph(text=text, zone=zone))

        for child in node.children:
            if isinstance(child, PreformattedString):  # comments, doctype, CDATA
                continue
            if isinstance(child, NavigableString):
                inline.append(str(child))
                continue
            if not isinstance(child, Tag) or child.name in IGNORED:
                continue
            if child.name in HEADINGS:
                flush()
                if text := tag_text(child):
                    anchor = child.get("data-title") or child.get("id")
                    level = self.heading_level(child, text)
                    self.blocks.append(Heading(level=level, text=text, anchor=anchor, zone=zone))
            elif child.name == "p":
                flush()
                if self.is_ecfr and self.add_ecfr_paragraph(child, zone):
                    continue
                if text := tag_text(child):
                    self.blocks.append(
                        Paragraph(text=text, anchor=child.get("data-title"), zone=zone)
                    )
            elif child.name in ("ul", "ol"):
                flush()
                if items := list_items(child):
                    self.blocks.append(
                        ListBlock(ordered=child.name == "ol", items=items, zone=zone)
                    )
            elif child.name == "table":
                flush()
                self.add_table(child, zone)
            elif child.name == "pre":
                flush()
                self.blocks.append(Code(text=child.get_text().strip("\n"), zone=zone))
            elif self.is_ecfr and "table-header" in (child.get("class") or []):
                flush()
                self.table_title = tag_text(child)
            elif child.name in CONTAINERS:
                flush()
                self.walk(child, *self.child_zone(child, zone, inside_main))
            else:
                inline.append(child.get_text())
        flush()


def parse_html_text(doc_id: str, source: bytes | str) -> ParsedDocument:
    soup = BeautifulSoup(source, "html.parser")
    for line_break in soup.find_all("br"):
        line_break.replace_with(" ")
    walker = HtmlWalker(soup)
    walker.walk(soup, "nav" if walker.has_main else "main", False)
    assign_section_paths(walker.blocks)
    return ParsedDocument(doc_id=doc_id, format="html", blocks=walker.blocks)


def parse_html(doc_id: str, path: Path) -> ParsedDocument:
    return parse_html_text(doc_id, path.read_bytes())
