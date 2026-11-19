import json
from functools import lru_cache
from pathlib import Path

import pytest

from factory_docs.chunking import (
    MAX_CHARS,
    NO_SECTION,
    OVERLAP,
    Chunk,
    DocumentInfo,
    block_units,
    chunk_line,
    chunk_stats,
    drafts_for,
    fixed_drafts,
    list_lines,
    missing_words,
    render_block,
    render_document,
    split_text,
    structured_drafts,
    to_chunks,
)
from factory_docs.cleaning import CleanedDocument
from factory_docs.corpus import in_scope
from factory_docs.parsing.model import (
    Block,
    Code,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    Table,
    assign_section_paths,
)

DATA = Path(__file__).resolve().parent.parent / "data"
CLEANED = DATA / "cleaned"
CHUNKS = DATA / "chunks"
INFO = DocumentInfo(
    title="Test sheet",
    source_url="https://example.org/t.pdf",
    version_or_date="v1, 2026-01-01",
    supersedes=None,
    language="en",
    doc_type="sds",
)


def document(blocks: list[Block], fmt: str = "pdf") -> CleanedDocument:
    for block in blocks:
        if fmt == "pdf" and block.page is None:
            block.page = 1
    assign_section_paths(blocks)
    return CleanedDocument(doc_id="T001", format=fmt, blocks=blocks, content_hash="0" * 64)


def sentences(count: int, words: int = 8) -> str:
    return " ".join(f"Sentence {i} " + "word " * words + "ends." for i in range(count))


def texts(blocks: list[Block]) -> list[str]:
    return [draft.text for draft in structured_drafts(document(blocks))]


# Structure-aware chunks
def test_small_subsections_merge_into_one_chunk():
    chunks = texts(
        [
            Heading(level=1, text="SECTION 1: Identification"),
            Heading(level=2, text="1.1 Product identifier"),
            Paragraph(text="Product name: Nitrogen"),
            Heading(level=2, text="1.2 Uses"),
            Paragraph(text="Industrial use."),
        ]
    )
    assert len(chunks) == 1
    assert "1.1 Product identifier" in chunks[0] and "Industrial use." in chunks[0]


def test_a_new_top_level_section_always_starts_a_chunk():
    chunks = texts(
        [
            Heading(level=1, text="SECTION 1: Identification"),
            Paragraph(text="Product name: Nitrogen"),
            Heading(level=1, text="SECTION 2: Hazards"),
            Paragraph(text="H280: Contains gas under pressure."),
        ]
    )
    assert len(chunks) == 2
    assert chunks[1].startswith("SECTION 2: Hazards")


def test_a_heading_closes_a_chunk_that_is_long_enough():
    chunks = texts(
        [
            Heading(level=1, text="SECTION 9"),
            Heading(level=2, text="9.1 Properties"),
            Paragraph(text=sentences(10)),  # about 570 characters
            Heading(level=2, text="9.2 Other information"),
            Paragraph(text=sentences(10)),
        ]
    )
    assert len(chunks) == 2
    assert chunks[1].startswith("9.2 Other information")


def test_a_trailing_heading_moves_to_the_next_chunk():
    chunks = texts(
        [
            Heading(level=1, text="SECTION 8"),
            Paragraph(text=sentences(4)),  # under the 500-character minimum
            Heading(level=2, text="8.2 Exposure controls"),
            Paragraph(text=sentences(24)),  # does not fit after the heading
        ]
    )
    assert len(chunks) == 2
    assert not chunks[0].endswith("8.2 Exposure controls")
    assert chunks[1].startswith("8.2 Exposure controls")


def test_a_short_last_subsection_joins_the_chunk_before_it():
    chunks = texts(
        [
            Heading(level=1, text="SECTION 2: Hazards"),
            Heading(level=2, text="2.2 Label elements"),
            Paragraph(text=sentences(8)),
            Heading(level=2, text="2.3 Other hazards: None."),
            Heading(level=1, text="SECTION 3: Composition"),
            Paragraph(text="Nitrogen 100 %"),
        ]
    )
    assert len(chunks) == 2
    assert chunks[0].endswith("2.3 Other hazards: None.")


def test_no_chunk_is_longer_than_the_limit():
    blocks: list[Block] = [Heading(level=1, text="SECTION 1")]
    blocks += [Paragraph(text=sentences(n)) for n in (3, 30, 1, 12, 50)]
    assert all(len(text) <= MAX_CHARS for text in texts(blocks))


def test_a_long_paragraph_splits_at_sentence_ends():
    parts = split_text(sentences(60), MAX_CHARS)
    assert len(parts) > 1
    assert all(len(part) <= MAX_CHARS and part.endswith("ends.") for part in parts)


def test_text_without_sentence_ends_splits_at_lines_then_spaces():
    lines = "\n".join("line " * 20 for _ in range(30))
    assert all(len(part) <= MAX_CHARS for part in split_text(lines, MAX_CHARS))
    one_word = "x" * 4000  # only an endless word is cut
    assert [len(part) for part in split_text(one_word, MAX_CHARS)] == [1500, 1500, 1000]


def table(rows: int, header_rows: int = 1) -> Table:
    body = [[f"Component {i}", f"{i}-00-0", f"{i} %"] for i in range(rows)]
    return Table(rows=[["Name", "CAS", "Share"]] * header_rows + body, header_rows=header_rows)


def test_a_table_that_fits_stays_whole():
    small = table(10)
    units = block_units(small)
    assert len(units) == 1 and units[0].table_part is None
    assert units[0].text.splitlines()[:2] == ["| Name | CAS | Share |", "| --- | --- | --- |"]
    chunks = texts([Heading(level=1, text="SECTION 3"), Paragraph(text=sentences(14)), small])
    assert any(render_block(small) in chunk for chunk in chunks)


def test_an_oversized_table_splits_by_rows_and_repeats_its_header():
    big = table(120)
    units = block_units(big)
    assert len(units) > 1
    assert [unit.table_part for unit in units] == [
        f"{i}/{len(units)}" for i in range(1, len(units) + 1)
    ]
    for unit in units:
        assert len(unit.text) <= MAX_CHARS
        assert unit.text.startswith("| Name | CAS | Share |\n| --- | --- | --- |\n")
    body_rows = [line for unit in units for line in unit.text.splitlines()[2:]]
    assert body_rows == render_block(big).splitlines()[2:]  # every row once, in order


def test_a_list_splits_between_items():
    items = [ListItem(text=f"Item {i}: " + "detail " * 20) for i in range(30)]
    units = block_units(ListBlock(items=items))
    assert len(units) > 1
    assert all(line.startswith("- Item") for unit in units for line in unit.text.splitlines())


def test_ordered_lists_are_numbered_per_depth():
    block = ListBlock(
        ordered=True,
        items=[
            ListItem(text="First"),
            ListItem(text="Sub a", depth=1),
            ListItem(text="Sub b", depth=1),
            ListItem(text="Second"),
            ListItem(text="Sub c", depth=1),
        ],
    )
    assert list_lines(block) == ["1. First", "  1. Sub a", "  2. Sub b", "2. Second", "  1. Sub c"]


def test_oversized_code_splits_between_lines_and_keeps_its_fence():
    code = Code(text="\n".join(f"option_{i} = {i}" for i in range(200)), language="toml")
    units = block_units(code)
    assert len(units) > 1
    assert all(u.text.startswith("```toml\n") and u.text.endswith("\n```") for u in units)
    assert all(len(unit.text) <= MAX_CHARS for unit in units)


def test_a_document_without_headings_still_gets_chunks_under_the_limit():
    doc = document([Paragraph(text=sentences(5)) for _ in range(20)])
    chunks = to_chunks(structured_drafts(doc), doc, INFO, "structure")
    assert len(chunks) > 1
    assert all(c.chars <= MAX_CHARS and c.section == NO_SECTION for c in chunks)


# Fixed-size baseline
def test_fixed_chunks_overlap_and_never_cut_a_word():
    doc = document([Heading(level=1, text="SECTION 1")] + [Paragraph(text=sentences(20))] * 6)
    text, _ = render_document(doc)
    drafts = fixed_drafts(doc)
    assert len(drafts) > 2
    for before, after in zip(drafts, drafts[1:], strict=False):
        assert 0 < before.end - after.start <= OVERLAP
    for draft in drafts:
        assert len(draft.text) <= MAX_CHARS
        assert draft.start == 0 or text[draft.start - 1].isspace()
        assert draft.end == len(text) or text[draft.end].isspace()


def test_fixed_chunks_cite_the_section_of_their_first_block():
    doc = document(
        [
            Heading(level=1, text="SECTION 1"),
            Paragraph(text=sentences(30)),
            Heading(level=1, text="SECTION 2"),
            Paragraph(text=sentences(30)),
        ]
    )
    chunks = to_chunks(fixed_drafts(doc), doc, INFO, "fixed")
    assert chunks[0].section == "SECTION 1"
    assert chunks[-1].section == "SECTION 2"


# Metadata and determinism
def test_every_chunk_has_all_its_metadata():
    doc = document(
        [
            Paragraph(text="SAFETY DATA SHEET", page=1),
            Heading(level=1, text="SECTION 1", page=1),
            Heading(level=2, text="1.1 Product identifier", page=1),
            Paragraph(text="Product name: Nitrogen", page=2),
        ]
    )
    chunks = to_chunks(structured_drafts(doc), doc, INFO, "structure")
    assert [c.chunk_id for c in chunks] == ["T001-s-0000", "T001-s-0001"]
    assert chunks[0].section == NO_SECTION
    assert chunks[1].section == "SECTION 1"
    assert (chunks[1].page_start, chunks[1].page_end) == (1, 2)
    assert chunks[1].block_kinds == ["heading", "paragraph"]
    assert chunks[1].version_or_date == "v1, 2026-01-01"
    assert chunks[1].content_hash == "0" * 64
    assert len(chunks[1].chunk_hash) == 64 and chunks[1].chars == len(chunks[1].text)


def test_a_chunk_spanning_subsections_cites_their_shared_section():
    doc = document(
        [
            Heading(level=1, text="SECTION 1"),
            Heading(level=2, text="1.1 Product identifier"),
            Paragraph(text="Nitrogen"),
            Heading(level=2, text="1.2 Uses"),
            Paragraph(text="Industrial"),
        ]
    )
    assert to_chunks(structured_drafts(doc), doc, INFO, "structure")[0].section == "SECTION 1"


def test_formats_without_pages_have_no_page_numbers():
    doc = document([Heading(level=1, text="Intro"), Paragraph(text="Text.")], fmt="html")
    chunk = to_chunks(structured_drafts(doc), doc, INFO, "structure")[0]
    assert chunk.page_start is None and chunk.page_end is None


def test_the_same_input_gives_the_same_output():
    def run() -> str:
        doc = document([Heading(level=1, text="SECTION 1"), Paragraph(text=sentences(40))])
        return "".join(
            chunk_line(chunk)
            for strategy in ("structure", "fixed")
            for chunk in to_chunks(drafts_for(doc, strategy), doc, INFO, strategy)
        )

    assert run() == run()


def test_stats_count_split_tables_and_lowercase_starts():
    doc = document([Heading(level=1, text="SECTION 3"), table(120), table(5)])
    stats = chunk_stats([(doc, structured_drafts(doc))])
    assert (stats.tables, stats.tables_whole) == (2, 1)
    assert stats.lowercase_starts == 0


# The committed corpus
@lru_cache
def cleaned() -> dict[str, CleanedDocument]:
    return {
        row["id"]: CleanedDocument.model_validate_json(
            (CLEANED / f"{row['id']}.json").read_text(encoding="utf-8")
        )
        for row in in_scope()
    }


@lru_cache
def committed(name: str) -> tuple[str, ...]:
    return tuple((CHUNKS / name).read_text(encoding="utf-8").splitlines(keepends=True))


@pytest.mark.parametrize("doc_id", ["D001", "D066", "D073", "D074"])  # PDF, HTML, DOCX, Markdown
def test_real_documents_lose_no_words(doc_id):
    doc = cleaned()[doc_id]
    for strategy in ("structure", "fixed"):
        drafts = drafts_for(doc, strategy)
        assert all(len(draft.text) <= MAX_CHARS for draft in drafts)
        assert not missing_words(doc, (draft.text for draft in drafts)), strategy


def test_d042_without_headings_is_chunked():
    chunks = [json.loads(line) for line in committed("chunks.jsonl")]
    d042 = [c for c in chunks if c["doc_id"] == "D042"]
    assert len(d042) > 1 and {c["section"] for c in d042} == {NO_SECTION}


@pytest.mark.parametrize(
    ("name", "strategy"), [("chunks.jsonl", "structure"), ("chunks_fixed.jsonl", "fixed")]
)
def test_committed_chunks_match_the_rules(name, strategy):
    rows = {row["id"]: row for row in in_scope()}
    expected = [
        chunk_line(chunk)
        for doc_id, doc in sorted(cleaned().items())
        for chunk in to_chunks(
            drafts_for(doc, strategy), doc, DocumentInfo.from_manifest(rows[doc_id]), strategy
        )
    ]
    assert list(committed(name)) == expected


@pytest.mark.parametrize("name", ["chunks.jsonl", "chunks_fixed.jsonl"])
def test_committed_chunks_are_complete(name):
    chunks = [Chunk.model_validate_json(line) for line in committed(name)]
    # section_path is empty before the first heading; `section` then says so
    optional = {"anchor", "supersedes", "table_part", "page_start", "page_end", "section_path"}
    for chunk in chunks:
        data = chunk.model_dump()
        assert all(data[key] not in (None, "", []) for key in data.keys() - optional), chunk
        assert chunk.chars == len(chunk.text) <= MAX_CHARS
        assert (chunk.page_start is not None) == (chunk.format == "pdf"), chunk.chunk_id
    assert {c.doc_id for c in chunks} == {row["id"] for row in in_scope()}
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    for doc_id, chunk_texts in by_doc.items():
        assert not missing_words(cleaned()[doc_id], chunk_texts), doc_id


def test_version_pairs_keep_their_own_version_on_every_chunk():
    chunks = [json.loads(line) for line in committed("chunks.jsonl")]
    versions = {(c["doc_id"], c["version_or_date"], c["supersedes"]) for c in chunks}
    for old, new in (("D066", "D067"), ("D071", "D072"), ("D074", "D075")):
        assert len({v for d, v, _ in versions if d == old}) == 1
        assert {s for d, _, s in versions if d == new} == {old}
