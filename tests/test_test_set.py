import pytest

from factory_docs.chunking import Chunk
from factory_docs.cleaning import CleanedDocument
from factory_docs.parsing.model import Paragraph
from factory_docs.test_set import (
    TEST_SET,
    Category,
    TestQuestion,
    check_question,
    coverage_problems,
    fact_in,
    load_test_set,
    superseded_by,
)

# Invented documents, not corpus documents: D901 is new, D902 its old version, D903 a sibling.
NEW = CleanedDocument(
    doc_id="D901",
    format="pdf",
    content_hash="a",
    blocks=[Paragraph(page=2, text="Boiling point: approx. 165 °C\n(ASTM D 1120)")],
)
OLD = CleanedDocument(
    doc_id="D902",
    format="pdf",
    content_hash="b",
    blocks=[Paragraph(page=2, text="Boiling point: approx. 150 °C")],
)
SIBLING = CleanedDocument(
    doc_id="D903",
    format="pdf",
    content_hash="c",
    blocks=[Paragraph(page=1, text="Boiling point: 108 °C")],
)
DOCUMENTS = {"D901": NEW, "D902": OLD, "D903": SIBLING}
MANIFEST = [
    {"id": "D901", "supersedes": "D902"},
    {"id": "D902", "supersedes": ""},
    {"id": "D903", "supersedes": ""},
]


def chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure",
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
        chunk("D901-s-0000", "D901", "Boiling point: approx. 165 °C (ASTM D 1120)"),
        chunk("D902-s-0000", "D902", "Boiling point: approx. 150 °C"),
        chunk("D903-s-0000", "D903", "Boiling point: 108 °C"),
    ]
}
EVIDENCE = {"doc_id": "D901", "section": "9", "page": 2, "quote": "approx. 165 °C (ASTM D 1120)"}


def make(category: str = "answerable", **fields) -> TestQuestion:
    base = {
        "id": "T01",
        "question": "What is the boiling point?",
        "language": "en",
        "category": category,
        "expected_answer": "About 165 °C.",
    }
    if category != "unanswerable":
        base |= {"key_facts": [["165 °C"]], "evidence": [EVIDENCE]}
    if category in ("similar_doc", "outdated_version"):
        base["wrong_docs"] = ["D903" if category == "similar_doc" else "D902"]
    return TestQuestion.model_validate(base | fields)


def problems(question: TestQuestion) -> list[str]:
    check = check_question(question, DOCUMENTS, COLLECTIONS, superseded_by(MANIFEST))
    return check.problems + [p for e in check.evidence for p in e.problems]


# Fact matching


def test_fact_matches_any_spelling_ignoring_case_and_line_breaks():
    assert fact_in("approx. 165\n°C", ["165 °C"])
    assert fact_in("Signal word: WARNING", ["Varning", "warning"])
    assert not fact_in("approx. 165°C", ["165 °C"])  # a missing space is another spelling


# Schema: fields follow the category


@pytest.mark.parametrize("category", ["answerable", "similar_doc", "outdated_version"])
def test_answer_categories_need_evidence_and_key_facts(category):
    make(category)  # valid
    with pytest.raises(ValueError, match="needs evidence"):
        make(category, evidence=[])
    with pytest.raises(ValueError, match="needs key facts"):
        make(category, key_facts=[])


def test_unanswerable_must_have_no_evidence_or_key_facts():
    assert not make("unanswerable").answerable
    with pytest.raises(ValueError, match="no evidence or key facts"):
        make("unanswerable", evidence=[EVIDENCE])
    with pytest.raises(ValueError, match="no evidence or key facts"):
        make("unanswerable", key_facts=[["x"]])


@pytest.mark.parametrize("category", ["similar_doc", "outdated_version"])
def test_traps_need_wrong_docs_that_are_not_evidence(category):
    with pytest.raises(ValueError, match="needs wrong_docs"):
        make(category, wrong_docs=[])
    with pytest.raises(ValueError, match="also evidence: D901"):
        make(category, wrong_docs=["D901"])


@pytest.mark.parametrize("field", [{"wrong_docs": ["D903"]}, {"forbidden_facts": [["108 °C"]]}])
@pytest.mark.parametrize("category", ["answerable", "unanswerable"])
def test_trap_fields_only_on_traps(category, field):
    with pytest.raises(ValueError, match="trap questions only"):
        make(category, **field)


def test_absent_terms_only_on_unanswerable():
    make("unanswerable", absent_terms=["SRM 999"])
    with pytest.raises(ValueError, match="unanswerable questions only"):
        make("answerable", absent_terms=["SRM 999"])


@pytest.mark.parametrize("facts", [[[]], [[""]], [["165 °C", " "]]])
def test_facts_need_non_empty_spellings(facts):
    with pytest.raises(ValueError, match="non-empty spellings"):
        make(key_facts=facts)


def test_wrong_docs_must_be_document_ids():
    with pytest.raises(ValueError, match="not a document ID"):
        make("similar_doc", wrong_docs=["903"])


# Corpus checks


@pytest.mark.parametrize("category", ["answerable", "similar_doc", "outdated_version"])
def test_valid_question_has_no_problems(category):
    assert problems(make(category)) == []


def test_key_fact_must_be_in_an_evidence_quote():
    assert problems(make(key_facts=[["165 °C"], ["170 °C", "171 °C"]])) == [
        "key fact ['170 °C', '171 °C'] is in no evidence quote"
    ]


def test_key_fact_from_another_alternative_quote_counts():
    second = {"doc_id": "D903", "section": "9", "page": 1, "quote": "Boiling point: 108 °C"}
    question = make(evidence=[EVIDENCE, second], key_facts=[["108 °C"]])
    assert problems(question) == []


def test_forbidden_fact_must_be_in_a_wrong_doc():
    assert problems(make("similar_doc", forbidden_facts=[["108 °C"]])) == []
    assert problems(make("similar_doc", forbidden_facts=[["150 °C"]])) == [
        "forbidden fact ['150 °C'] is in no wrong doc"
    ]


def test_forbidden_fact_must_not_be_in_an_evidence_quote():
    question = make("similar_doc", forbidden_facts=[["108 °C", "ASTM"]])
    assert problems(question) == ["forbidden fact ['108 °C', 'ASTM'] is in an evidence quote"]


def test_forbidden_fact_must_not_be_a_key_fact():
    sibling = {"doc_id": "D901", "section": "9", "page": 2, "quote": "approx. 165 °C"}
    question = make(
        "similar_doc",
        evidence=[sibling],
        wrong_docs=["D903"],
        key_facts=[["Boiling point"]],
        forbidden_facts=[["Boiling point"]],
    )
    assert "forbidden fact ['Boiling point'] is also a key fact" in problems(question)


def test_outdated_wrong_doc_must_be_superseded_by_the_evidence():
    assert problems(make("outdated_version", wrong_docs=["D902"])) == []
    assert problems(make("outdated_version", wrong_docs=["D903"])) == [
        "wrong doc D903 is not superseded by an evidence document"
    ]


def test_wrong_doc_must_be_indexed():
    assert problems(make("similar_doc", wrong_docs=["D999"])) == [
        "wrong doc D999 is not in the RAG index"
    ]


def test_absent_term_found_in_a_chunk_is_a_problem():
    assert problems(make("unanswerable", absent_terms=["SRM 999"])) == []
    assert problems(make("unanswerable", absent_terms=["astm d 1120"])) == [
        "absent term 'astm d 1120' is in D901"
    ]


def test_evidence_checks_still_run():
    bad = EVIDENCE | {"quote": "approx. 170 °C"}
    assert problems(make(evidence=[bad], key_facts=[["170 °C"]])) == [
        "quote not found in D901 (it must sit in one block)"
    ]


def test_supersedes_chain_is_followed():
    rows = [
        {"id": "D1", "supersedes": ""},
        {"id": "D2", "supersedes": "D1"},
        {"id": "D3", "supersedes": "D2"},
    ]
    assert superseded_by(rows) == {"D1": {"D2", "D3"}, "D2": {"D3"}}


# The set as a whole


def test_coverage_needs_enough_questions_and_every_category():
    full = [make(c.value) for c in Category]
    assert coverage_problems(full, minimum=4) == []
    assert coverage_problems(full, minimum=5) == ["4 questions, at least 5 needed"]
    assert coverage_problems(full[:2], minimum=2) == [
        "no question in category: outdated_version, unanswerable"
    ]


# Loading


TEST_TOML = """
[[question]]
id = "T01"
question = "What is the boiling point of X?"
language = "en"
category = "unanswerable"
expected_answer = "Not in the documents."
"""
DEV_TOML = """
[[question]]
id = "V01"
question = "What is the boiling point of Y?"
language = "en"
kind = "near_miss"
"""
GOLD_TOML = """
[[question]]
id = "R01"
question = "What is the boiling point of Z?"
language = "en"

[[question.evidence]]
doc_id = "D901"
section = "9"
page = 2
quote = "165 °C"
"""


def load(tmp_path, test=TEST_TOML, dev=DEV_TOML, gold=GOLD_TOML):
    paths = []
    for name, text in (("test", test), ("dev", dev), ("gold", gold)):
        path = tmp_path / f"{name}.toml"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return load_test_set(*paths)


def test_valid_file_loads(tmp_path):
    assert [q.id for q in load(tmp_path)] == ["T01"]


@pytest.mark.parametrize("other", ["dev", "gold"])
def test_question_from_dev_or_gold_set_is_rejected(tmp_path, other):
    letter = "Y" if other == "dev" else "Z"
    test = TEST_TOML.replace("of X?", f"of  {letter.lower()}?")  # whitespace and case ignored
    with pytest.raises(ValueError, match=f"also in {other}.toml, so not held out: T01"):
        load(tmp_path, test=test)


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate question id: T01"):
        load(tmp_path, test=TEST_TOML + TEST_TOML.replace("of X", "of W"))


def test_schema_problems_name_the_question(tmp_path):
    with pytest.raises(ValueError, match="T01 category: Input should be"):
        load(tmp_path, test=TEST_TOML.replace('"unanswerable"', '"maybe"'))


def test_invalid_toml_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not valid TOML"):
        load(tmp_path, test="[[question]\n")


def test_committed_test_set_is_valid_and_complete():
    questions = load_test_set(TEST_SET)
    assert coverage_problems(questions) == []
