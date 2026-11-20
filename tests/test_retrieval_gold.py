from pathlib import Path

import pytest

from factory_docs.chunking import Chunk
from factory_docs.cleaning import CleanedDocument
from factory_docs.config import ROOT, load_config
from factory_docs.index import load_chunks
from factory_docs.parsing.model import ListBlock, ListItem, Paragraph, Table
from factory_docs.retrieval_gold import (
    Evidence,
    Question,
    check_evidence,
    contains_quote,
    is_hit,
    load_documents,
    load_gold,
)

# Invented documents, not corpus documents.
PDF = CleanedDocument(
    doc_id="D901",
    format="pdf",
    content_hash="x",
    blocks=[
        Paragraph(page=1, text="Product name: Example Gas"),
        Paragraph(page=4, text="General Fire Hazards: Heat may cause the\ncontainers to explode."),
        Table(page=5, rows=[["Purity, %", "O2 ≥ 99,999"]]),
    ],
)
DOCX = CleanedDocument(
    doc_id="D902",
    format="docx",
    content_hash="y",
    blocks=[ListBlock(items=[ListItem(text="Keep records of employee training.")])],
)
DOCUMENTS = {"D901": PDF, "D902": DOCX}


def chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure" if "-s-" in chunk_id else "fixed",
        doc_id=doc_id,
        index=0,
        title="t",
        section="s",
        section_path=[],
        page_start=None,
        page_end=None,
        anchor=None,
        version_or_date="v1",
        supersedes=None,
        language="en",
        doc_type="sds",
        format="pdf",
        source_url="https://example.com",
        content_hash="x",
        chunk_hash="h",
        chars=len(text),
        block_kinds=["paragraph"],
        table_part=None,
        text=text,
    )


COLLECTIONS = {
    "structure": [
        chunk(
            "D901-s-0000", "D901", "General Fire Hazards: Heat may cause the containers to explode."
        ),
        chunk("D901-s-0001", "D901", "| Purity, % | O2 ≥ 99,999 |"),
        chunk("D902-s-0000", "D902", "- Keep records of employee training."),
    ],
    "fixed": [
        chunk("D901-f-0000", "D901", "Heat may cause the containers"),  # quote cut at the boundary
        chunk("D901-f-0001", "D901", "to explode. | Purity, % | O2 ≥ 99,999 |"),
        chunk("D902-f-0000", "D902", "- Keep records of employee training."),
    ],
}


def question(**evidence) -> Question:
    fields = {"doc_id": "D901", "section": "5", "page": 4, "quote": "Heat may cause the containers"}
    return Question(id="R01", question="q?", language="en", evidence=[{**fields, **evidence}])


def problems(q: Question) -> list[str]:
    return [p for check in check_evidence(q, DOCUMENTS, COLLECTIONS) for p in check.problems]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "gold.toml"
    path.write_text(text, encoding="utf-8")
    return path


VALID_TOML = """
[[question]]
id = "R01"
question = "What happens in a fire?"
language = "en"

[[question.evidence]]
doc_id = "D901"
section = "5"
page = 4
quote = 'Heat may cause the containers to explode.'
"""


# The hit rule
def test_quote_match_ignores_line_breaks_but_not_case_or_punctuation():
    text = "Heat may cause the\ncontainers   to explode."
    assert contains_quote(text, "Heat may cause the containers to explode.")
    assert not contains_quote(text, "heat may cause the containers to explode.")
    assert not contains_quote(text, "Heat may cause the containers to explode!")


def test_hit_needs_the_evidence_document():
    evidence = Evidence(doc_id="D901", section="5", page=4, quote="Heat may cause")
    same_text = "Heat may cause the containers to explode."
    assert is_hit(chunk("D901-s-0000", "D901", same_text), evidence)
    assert not is_hit(chunk("D903-s-0000", "D903", same_text), evidence)  # e.g. argon vs nitrogen


# Checks against documents and chunks
def test_valid_evidence_reports_the_chunks_that_hold_the_quote():
    [check] = check_evidence(question(), DOCUMENTS, COLLECTIONS)
    assert check.problems == []
    assert check.hits == {"structure": ["D901-s-0000"], "fixed": ["D901-f-0000"]}


def test_quote_in_a_table_cell_and_in_a_list_item_is_found():
    assert problems(question(page=5, quote="O2 ≥ 99,999")) == []
    assert problems(question(doc_id="D902", page=None, quote="Keep records of employee")) == []


def test_document_not_in_the_index():
    assert problems(question(doc_id="D999")) == ["D999 is not in the RAG index"]


def test_quote_not_in_the_document():
    assert problems(question(quote="Heat may cause the tank to explode.")) == [
        "quote not found in D901 (it must sit in one block)"
    ]


def test_quote_across_two_blocks_is_not_found():
    quote = "Product name: Example Gas General Fire Hazards"
    assert problems(question(page=1, quote=quote)) == [
        "quote not found in D901 (it must sit in one block)"
    ]


def test_wrong_page():
    assert problems(question(page=3)) == ["quote is on page 4 of D901, not page 3"]


def test_pdf_needs_a_page():
    assert problems(question(page=None)) == ["D901 is a PDF: `page` is required (quote found on 4)"]


def test_format_without_pages_must_not_have_one():
    assert problems(question(doc_id="D902", page=2, quote="Keep records")) == [
        "D902 is docx, which has no pages: remove `page`"
    ]


def test_quote_split_across_a_chunk_boundary():
    assert problems(question(quote="Heat may cause the containers to explode.")) == [
        "quote is in no `fixed` chunk (split across a chunk boundary?)"
    ]


def test_each_alternative_is_checked_on_its_own():
    q = question()
    q.evidence.append(Evidence(doc_id="D901", section="3", page=2, quote="O2 ≥ 99,999"))
    checks = check_evidence(q, DOCUMENTS, COLLECTIONS)
    assert [c.problems for c in checks] == [[], ["quote is on page 5 of D901, not page 2"]]


# Loading the file
def test_valid_file_loads(tmp_path):
    [q] = load_gold(write(tmp_path, VALID_TOML))
    assert q.id == "R01" and q.evidence[0].page == 4 and q.notes == ""


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate question id: R01"):
        load_gold(write(tmp_path, VALID_TOML + VALID_TOML))


def test_question_without_evidence_is_rejected(tmp_path):
    text = '[[question]]\nid = "R01"\nquestion = "q?"\nlanguage = "en"\n'
    with pytest.raises(ValueError, match="R01 evidence: Field required"):
        load_gold(write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('id = "R01"', 'id = "Q1"', "R01|Q1 id: String should match pattern"),
        ('doc_id = "D901"', 'doc_id = "901"', "R01 evidence 1 doc_id: String should match"),
        ("page = 4", "page = 0", "R01 evidence 1 page: Input should be greater than"),
        ('section = "5"', 'sektion = "5"', "R01 evidence 1 sektion: Extra inputs"),
        ("quote = 'Heat may cause the containers to explode.'", "quote = ''", "quote: String"),
    ],
)
def test_schema_problems_name_the_question(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_gold(write(tmp_path, VALID_TOML.replace(old, new)))


def test_invalid_toml_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not valid TOML"):
        load_gold(write(tmp_path, VALID_TOML + "quote = 'unclosed"))


# The committed gold set (needs the committed chunk files and cleaned documents)
def test_committed_gold_set_is_valid():
    questions = load_gold()
    collections = {
        name: load_chunks(ROOT / path) for name, path in load_config().index.collections.items()
    }
    indexed = {c.doc_id for chunks in collections.values() for c in chunks}
    documents = load_documents({e.doc_id for q in questions for e in q.evidence} & indexed)
    checks = [c for q in questions for c in check_evidence(q, documents, collections)]
    assert [(c.question_id, c.problems) for c in checks if c.problems] == []
    assert 10 <= len(questions) <= 15
