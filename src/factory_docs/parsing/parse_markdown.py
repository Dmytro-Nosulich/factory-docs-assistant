"""Markdown to blocks: headings, nested lists, pipe tables and fenced code (S2-03).

A small line parser is enough for the corpus (Telegraf READMEs). A `#` inside a code fence is a
config comment, not a heading.
"""

import re
from pathlib import Path

from factory_docs.parsing.model import (
    Block,
    Code,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    ParsedDocument,
    Table,
    assign_section_paths,
    rectangular,
)

HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*(\S*)")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
LINK_REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s+\S+")
TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def split_table_row(line: str) -> list[str]:
    """Split on unescaped pipes outside inline code."""
    body = line.strip().removeprefix("|")
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    cells, current, in_code = [], "", False
    for index, char in enumerate(body):
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code and (index == 0 or body[index - 1] != "\\"):
            cells.append(current.strip())
            current = ""
        else:
            current += char
    cells.append(current.strip())
    return [cell.replace("\\|", "|") for cell in cells]


def is_table_start(lines: list[str], index: int) -> bool:
    return (
        "|" in lines[index]
        and index + 1 < len(lines)
        and "-" in lines[index + 1]
        and TABLE_DIVIDER.match(lines[index + 1]) is not None
    )


def starts_new_block(lines: list[str], index: int) -> bool:
    line = lines[index]
    return (
        not line.strip()
        or HEADING.match(line) is not None
        or FENCE.match(line) is not None
        or LIST_ITEM.match(line) is not None
        or LINK_REFERENCE.match(line) is not None
        or is_table_start(lines, index)
    )


def parse_list(lines: list[str], start: int) -> tuple[ListBlock, int]:
    items: list[ListItem] = []
    indents: list[int] = []
    ordered = LIST_ITEM.match(lines[start]).group(2)[0].isdigit()
    index = start
    while index < len(lines):
        line = lines[index]
        match = LIST_ITEM.match(line)
        if match:
            indent = len(match.group(1).expandtabs(4))
            while indents and indents[-1] > indent:
                indents.pop()
            if not indents or indents[-1] < indent:
                indents.append(indent)
            items.append(ListItem(text=match.group(3).strip(), depth=len(indents) - 1))
        elif line.startswith((" ", "\t")) and line.strip() and items:
            items[-1].text = f"{items[-1].text} {line.strip()}"
        else:
            break
        index += 1
    return ListBlock(ordered=ordered, items=items), index


def parse_markdown_text(doc_id: str, source: str) -> ParsedDocument:
    lines = HTML_COMMENT.sub("", source).splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or LINK_REFERENCE.match(line):
            index += 1
        elif fence := FENCE.match(line):
            marker = fence.group(1)
            body: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(marker):
                body.append(lines[index])
                index += 1
            index += 1  # the closing fence
            blocks.append(Code(text="\n".join(body), language=fence.group(2) or None))
        elif heading := HEADING.match(line):
            blocks.append(Heading(level=len(heading.group(1)), text=heading.group(2)))
            index += 1
        elif is_table_start(lines, index):
            rows = [split_table_row(line)]
            index += 2  # the header and its divider
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(split_table_row(lines[index]))
                index += 1
            blocks.append(Table(rows=rectangular(rows), header_rows=1))
        elif LIST_ITEM.match(line):
            block, index = parse_list(lines, index)
            blocks.append(block)
        else:
            text = [line.strip()]
            index += 1
            while index < len(lines) and not starts_new_block(lines, index):
                text.append(lines[index].strip())
                index += 1
            blocks.append(Paragraph(text=" ".join(text)))
    assign_section_paths(blocks)
    return ParsedDocument(doc_id=doc_id, format="md", blocks=blocks)


def parse_markdown(doc_id: str, path: Path) -> ParsedDocument:
    return parse_markdown_text(doc_id, path.read_text(encoding="utf-8"))
