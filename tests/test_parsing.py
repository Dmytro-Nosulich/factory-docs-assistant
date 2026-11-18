import csv
import logging
from functools import lru_cache
from pathlib import Path

import pytest
from docx import Document

from factory_docs.parsing import parse_document
from factory_docs.parsing.model import (
    Code,
    Heading,
    ListBlock,
    Paragraph,
    ParsedDocument,
    Table,
    assign_section_paths,
    rectangular,
    to_review_json,
)
from factory_docs.parsing.parse_html import parse_html_text
from factory_docs.parsing.parse_markdown import parse_markdown_text, split_table_row
from factory_docs.parsing.parse_pdf import (
    build_table,
    find_furniture,
    heading_level,
    strip_decorative_dots,
    table_end,
)
from factory_docs.pdf_text import StyledLine

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
PARSED = DATA / "parsed"
logging.getLogger("pypdf").setLevel(logging.CRITICAL)


@lru_cache
def parsed(doc_id: str, suffix: str) -> ParsedDocument:
    return parse_document(doc_id, RAW / f"{doc_id}.{suffix}")


def of_kind(document: ParsedDocument, kind: type, page: int | None = None) -> list:
    return [
        block
        for block in document.blocks
        if isinstance(block, kind) and (page is None or block.page == page)
    ]


def test_section_path_follows_heading_levels_and_ignores_other_zones():
    blocks = [
        Heading(level=1, text="A"),
        Heading(level=2, text="A.1"),
        Paragraph(text="running header", zone="header"),
        Paragraph(text="body"),
        Heading(level=2, text="A.2"),
        Heading(level=1, text="B"),
        Paragraph(text="end"),
    ]
    assign_section_paths(blocks)
    assert blocks[2].section_path == []
    assert blocks[3].section_path == ["A", "A.1"]
    assert blocks[4].section_path == ["A", "A.2"]
    assert blocks[6].section_path == ["B"]


def test_rectangular_pads_short_rows():
    assert rectangular([["a", "b"], ["c"]]) == [["a", "b"], ["c", ""]]


@pytest.mark.parametrize(
    ("line", "structured", "expected"),
    [
        (StyledLine("SECTION 3: Composition"), True, 1),
        (StyledLine("РОЗДІЛ 1: Назва"), True, 1),
        (StyledLine("3.1 Substances", 10.0, True), True, 2),
        (StyledLine("3.2 Gemische", 10.0, False), True, 2),
        (StyledLine("1.5 mg/l", 10.0, False), True, None),
        (StyledLine("1.10 The quality unit should be involved", 10.0, False), False, None),
        (StyledLine("Product name:", 10.0, True), False, None),
        (StyledLine("Bold label", 10.0, True), True, None),
        (StyledLine("Bold title", 10.0, True), False, 4),
        (StyledLine("Large title", 14.0, False), False, 1),
    ],
)
def test_pdf_heading_rules(line, structured, expected):
    assert heading_level(line, 10.0, [14.0], structured) == expected


def test_pdf_table_joins_continuation_lines_to_the_previous_row():
    lines = [
        StyledLine("Name          Nitrogen"),
        StyledLine("CAS-No.:      7727-37-9"),
        StyledLine("Trade name:   BIOGON N,"),
        StyledLine("              Nitrogen 4.0"),
        StyledLine("Next paragraph starts here"),
    ]
    end = table_end(lines, 0)
    assert end == 4
    assert build_table(lines[:end]).rows == [
        ["Name", "Nitrogen"],
        ["CAS-No.:", "7727-37-9"],
        ["Trade name:", "BIOGON N, Nitrogen 4.0"],
    ]


def test_pdf_two_aligned_lines_are_not_a_table():
    lines = [StyledLine("Name      Nitrogen"), StyledLine("CAS       7727-37-9")]
    assert table_end(lines, 0) == 0


def test_pdf_repeated_lines_are_furniture_even_when_page_numbers_change():
    pages = [
        [StyledLine("Report"), StyledLine(f"body {n * 'x'}"), StyledLine(f"{n}/3")]
        for n in (1, 2, 3)
    ]
    top, bottom = find_furniture(pages)
    assert "report" in top
    assert "#/#" in bottom


def test_pdf_decorative_dots_are_removed_only_when_common():
    dotted = [[StyledLine("· 2.1 Title"), StyledLine("· Signal word"), StyledLine("plain")]]
    assert strip_decorative_dots(dotted)[0][0].text == "2.1 Title"
    rare = [[StyledLine("· item")] + [StyledLine("plain")] * 9]
    assert strip_decorative_dots(rare)[0][0].text == "· item"


def test_real_pdf_keeps_headings_tables_and_marks_running_headers():
    document = parsed("D001", "pdf")
    assert document.page_count == 12
    headings = [block.text for block in of_kind(document, Heading, page=3)]
    assert "SECTION 3: Composition/information on ingredients" in headings
    table = next(t for t in of_kind(document, Table, page=3) if ["CAS-No.:", "7727-37-9"] in t.rows)
    assert table.section_path == [
        "SECTION 3: Composition/information on ingredients",
        "3.1 Substances",
    ]
    running = [b for b in of_kind(document, Paragraph) if b.text == "SAFETY DATA SHEET"]
    assert running
    assert {block.zone for block in running} == {"header"}


def test_real_pdf_keeps_bullet_lists():
    lists = of_kind(parsed("D019", "pdf"), ListBlock, page=14)
    items = [item.text for block in lists for item in block.items]
    assert any(text.startswith("Order code for measuring device") for text in items)


def test_real_pdf_with_decorative_dots_keeps_numbered_headings():
    headings = of_kind(parsed("D033", "pdf"), Heading, page=2)
    match = next(h for h in headings if h.text == "2.1 Einstufung des Stoffs oder Gemischs")
    assert match.level == 2
    assert match.section_path[0] == "ABSCHNITT 2: Mögliche Gefahren"


@pytest.mark.parametrize(
    ("doc_id", "pages"),
    [("D014", [3, 4, 5, 6, 7, 8, 9]), ("D019", [37, 38, 39]), ("D021", [47])],
)
def test_pages_without_text_are_reported(doc_id, pages):
    document = parsed(doc_id, "pdf")
    assert document.pages_without_text == pages
    assert any(str(pages) in warning for warning in document.warnings)


def test_markdown_structure_from_inline_text():
    document = parse_markdown_text(
        "T1",
        "<!-- note -->\n# Title\n\nSome text\ncontinues.\n\n- one\n  - nested\n- two\n\n"
        "| A | B |\n|---|---|\n| `x|y` | 2 |\n\n```toml\n# not a heading\nkey = 1\n```\n\n"
        "[ref]: https://example.com\n",
    )
    kinds = [block.kind for block in document.blocks]
    assert kinds == ["heading", "paragraph", "list", "table", "code"]
    assert document.blocks[1].text == "Some text continues."
    assert [item.depth for item in document.blocks[2].items] == [0, 1, 0]
    assert document.blocks[3].rows == [["A", "B"], ["`x|y`", "2"]]
    assert document.blocks[4].language == "toml"
    assert document.blocks[4].section_path == ["Title"]


def test_markdown_table_row_keeps_escaped_pipes():
    assert split_table_row(r"| a\|b | c |") == ["a|b", "c"]


def test_real_markdown_comments_in_code_are_not_headings():
    document = parsed("D075", "md")
    headings = of_kind(document, Heading)
    assert len(headings) == 53
    assert not any(block.text.startswith("Retrieve data") for block in headings)
    assert any("slave_id = 1" in block.text for block in of_kind(document, Code))


def test_real_markdown_keeps_tables_and_lists():
    document = parsed("D076", "md")
    table = next(t for t in of_kind(document, Table) if t.rows[0] == ["Token", "Meaning"])
    assert table.header_rows == 1
    assert {len(row) for row in table.rows} == {2}
    assert of_kind(document, ListBlock)


def test_docx_structure_and_merged_cells(tmp_path):
    source = Document()
    source.add_heading("Scope", level=1)
    source.add_heading("Details", level=2)
    source.add_paragraph("First item", style="List Bullet")
    source.add_paragraph("Second item", style="List Number")
    table = source.add_table(rows=2, cols=3)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Wide"
    table.cell(0, 2).text = "Tall"
    table.cell(0, 2).merge(table.cell(1, 2))
    table.cell(1, 0).text = "a"
    table.cell(1, 1).text = "b"
    path = tmp_path / "sample.docx"
    source.save(path)

    document = parse_document("T2", path)
    assert [block.kind for block in document.blocks] == ["heading", "heading", "list", "table"]
    assert document.blocks[1].section_path == ["Scope", "Details"]
    assert [item.text for item in document.blocks[2].items] == ["First item", "Second item"]
    assert document.blocks[3].rows == [["Wide", "Wide", "Tall"], ["a", "b", "Tall"]]


def test_real_docx_keeps_headings_lists_and_tables():
    document = parsed("D072", "docx")
    heading = next(h for h in of_kind(document, Heading) if h.text == "Facility Information.")
    assert heading.level == 2
    assert heading.section_path[0] == "FACILITY DESCRIPTION AND CONTACT INFORMATION"
    items = [item.text for block in of_kind(document, ListBlock) for item in block.items]
    assert any(text.startswith("Boundaries of the property") for text in items)
    tables = of_kind(document, Table)
    assert any(len(table.rows) == 28 and len(table.rows[0]) == 6 for table in tables)
    assert all(len({len(row) for row in table.rows}) == 1 for table in tables)


def test_real_docx_marks_the_table_of_contents():
    document = parsed("D072", "docx")
    assert any(block.zone == "toc" for block in document.blocks)
    assert all(block.zone == "main" for block in of_kind(document, Heading))


def test_real_docx_reads_text_inside_content_controls():
    tables = of_kind(parsed("D073", "docx"), Table)
    assert tables[0].rows[0][0] == "Training Date: Insert Date of Training"


def test_real_docx_reads_each_text_box_once():
    lists = of_kind(parsed("D071", "docx"), ListBlock)
    items = [item.text for block in lists for item in block.items]
    assert items.count("You will need the information from this section to complete your NOI.") == 1


def test_html_structure_spans_and_zones():
    document = parse_html_text(
        "T3",
        """<html><head><title>Ignored</title><script>var x = 1;</script></head><body>
        <header><p>Site name</p></header>
        <nav><ul><li>Home</li></ul></nav>
        <main><h1>Title</h1><!-- hidden --><p>Body <em>text</em>.</p>
        <ul><li>One<ul><li>Nested</li></ul></li><li>Two</li></ul>
        <table><caption>Limits</caption>
        <thead><tr><th>A</th><th colspan="2">B</th></tr></thead>
        <tbody><tr><td rowspan="2">x</td><td>1</td><td>2</td></tr>
        <tr><td>3</td><td>4</td></tr></tbody>
        </table>
        <aside><p>Related</p></aside></main>
        <footer><p>Contact</p></footer></body></html>""",
    )
    main = [block for block in document.blocks if block.zone == "main"]
    assert [block.kind for block in main] == ["heading", "paragraph", "list", "table"]
    assert main[1].text == "Body text."
    assert [(item.text, item.depth) for item in main[2].items] == [
        ("One", 0),
        ("Nested", 1),
        ("Two", 0),
    ]
    assert main[3].caption == "Limits"
    assert main[3].header_rows == 1
    assert main[3].rows == [["A", "B", "B"], ["x", "1", "2"], ["x", "3", "4"]]
    zones = {block.text: block.zone for block in of_kind(document, Paragraph)}
    assert zones["Site name"] == "header"
    assert zones["Related"] == "aside"
    assert zones["Contact"] == "footer"
    assert next(b for b in document.blocks if b.kind == "list").zone == "nav"


def test_real_ecfr_keeps_named_paragraphs_as_sections():
    document = parsed("D067", "html")
    dates = next(h for h in of_kind(document, Heading) if h.text == "(j) Dates")
    assert dates.anchor == "1910.1200(j)"
    assert dates.section_path == ["§ 1910.1200 Hazard communication.", "(j) Dates"]
    effective = next(b for b in of_kind(document, Paragraph) if b.anchor == "1910.1200(j)(1)")
    assert effective.section_path[-1] == "(j) Dates"


def test_real_ecfr_nests_appendix_headings():
    headings = of_kind(parsed("D067", "html"), Heading)
    classification = next(h for h in headings if h.text == "A.0.1 Classification")
    assert classification.section_path[0].startswith("Appendix A to § 1910.1200")
    assert classification.section_path[1] == "A.0 General Classification Considerations"


@pytest.mark.parametrize("doc_id", ["D066", "D067"])
def test_real_ecfr_tables_keep_captions_and_complete_rows(doc_id):
    tables = of_kind(parsed(doc_id, "html"), Table)
    table = next(t for t in tables if t.caption and t.caption.startswith("Table A.1.1"))
    assert table.rows[0] == [
        "Exposure route",
        "Category 1",
        "Category 2",
        "Category 3",
        "Category 4",
    ]
    assert table.header_rows == 1
    assert all(len({len(row) for row in t.rows}) == 1 for t in tables)


def test_real_web_page_separates_content_from_navigation():
    document = parsed("D069", "html")
    main_headings = [b.text for b in of_kind(document, Heading) if b.zone == "main"]
    assert "What is DSEAR?" in main_headings
    assert "Related content" not in main_headings
    assert {"header", "nav", "aside", "footer"} <= {block.zone for block in document.blocks}
    assert any(block.zone == "main" for block in of_kind(document, ListBlock))


def test_every_in_scope_document_has_a_valid_parsed_file():
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as f:
        in_scope = {row["id"] for row in csv.DictReader(f) if row["in_rag_index"] == "yes"}
    files = {path.stem: path for path in PARSED.glob("D*.json")}
    assert set(files) == in_scope
    for doc_id, path in files.items():
        document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        assert document.doc_id == doc_id
        assert document.blocks


@pytest.mark.parametrize(
    ("doc_id", "suffix"),
    [("D011", "pdf"), ("D073", "docx"), ("D068", "html"), ("D077", "md")],
)
def test_committed_parsed_file_matches_the_parser(doc_id, suffix):
    committed = (PARSED / f"{doc_id}.json").read_text(encoding="utf-8")
    assert committed == to_review_json(parsed(doc_id, suffix))
