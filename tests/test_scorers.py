"""The S2-17 deterministic scorers: refusal, source, key facts, traps, and their rates.

The questions, chunks and answers are invented (documents D901-D904 are not corpus documents),
and every expected number is counted by hand in the comments. The last tests run the scorers
on the committed test set with answers built from its own labels.
"""

import pytest

from evals.scorers import (
    cited_chunk_ids,
    cited_docs,
    claim_text,
    counts,
    evidence_cited,
    facts_found,
    forbidden_found,
    refusal_correct,
    refused,
    right_source,
    score_question,
    summarize,
    wrong_source,
)
from factory_docs.chunking import Chunk
from factory_docs.config import ROOT
from factory_docs.index import load_chunks
from factory_docs.retrieval_gold import is_hit
from factory_docs.test_set import Category, TestQuestion, load_test_set


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure",
        doc_id=chunk_id.split("-")[0],
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


# D901 is the current document, D902 its old version, D903 a sibling of D904.
CHUNKS = {
    c.chunk_id: c
    for c in [
        chunk("D901-s-0000", "Boiling point: approx. 165 °C\n(ASTM D 1120)"),
        chunk("D901-s-0001", "Physical state: liquid"),
        chunk("D901-s-0002", "Manufacturers shall comply no later than May 19, 2026."),
        chunk("D902-s-0000", "Manufacturers shall comply no later than June 1, 2015."),
        chunk("D903-s-0000", "Four wafers, sliced to 1 mm thickness."),
        chunk("D904-s-0000", "Four wafers, sliced to 3 mm thickness."),
    ]
}


def evidence(doc_id: str, quote: str) -> dict:
    return {"doc_id": doc_id, "section": "s", "page": 1, "quote": quote}


ANSWERABLE = TestQuestion(
    id="T91",
    question="What is the boiling point, and by which method?",
    language="en",
    category="answerable",
    expected_answer="About 165 °C, ASTM D 1120.",
    key_facts=[["165 °C", "165°C"], ["ASTM D 1120"]],
    evidence=[evidence("D901", "Boiling point: approx. 165 °C (ASTM D 1120)")],
)
SIMILAR = TestQuestion(
    id="T92",
    question="Wie dick sind die Wafer?",
    language="de",
    category="similar_doc",
    expected_answer="3 mm.",
    key_facts=[["3 mm"]],
    forbidden_facts=[["1 mm"]],
    wrong_docs=["D903"],
    evidence=[evidence("D904", "sliced to 3 mm thickness")],
)
OUTDATED = TestQuestion(
    id="T93",
    question="By what date must manufacturers comply?",
    language="en",
    category="outdated_version",
    expected_answer="May 19, 2026.",
    key_facts=[["May 19, 2026", "19 May 2026"]],
    wrong_docs=["D902"],
    evidence=[evidence("D901", "no later than May 19, 2026")],
)
UNANSWERABLE = TestQuestion(
    id="T94",
    question="Яка температура кипіння продукту X?",
    language="uk",
    category="unanswerable",
    expected_answer="Not in the documents.",
)


def answer(status: str, *claims: tuple[str, list[tuple[str, str]]], missing=None) -> dict:
    """An Answer dict; each claim is (text, [(chunk_id, quote), ...])."""
    return {
        "status": status,
        "claims": [
            {"text": text, "citations": [{"chunk_id": c, "quote": q} for c, q in cites]}
            for text, cites in claims
        ],
        "missing": missing,
    }


def row(answer_dict: dict | None, final_status: str = "valid") -> dict:
    return {"final_status": final_status, "answer": answer_dict}


NOT_IN_DOCUMENTS = row(answer("not_in_documents", missing="Nothing about it."))
GOOD = answer(
    "answered",
    ("The boiling point is about 165 °C.", [("D901-s-0000", "approx. 165 °C")]),
    ("It is measured by ASTM D 1120.", [("D901-s-0000", "(ASTM D 1120)")]),
)

# --- parts of an answer ------------------------------------------------------------------------


def test_claim_text_leaves_out_quotes_and_missing():
    a = answer(
        "partial",
        ("About 165 °C.", [("D901-s-0000", "(ASTM D 1120)")]),
        missing="ASTM D 1120 is not stated",
    )
    assert claim_text(a) == "About 165 °C."
    assert claim_text(None) == ""
    assert claim_text(answer("not_in_documents", missing="x")) == ""


def test_cited_chunks_once_in_order_and_their_documents():
    a = answer(
        "answered",
        ("a", [("D901-s-0001", "q"), ("D903-s-0000", "q")]),
        ("b", [("D901-s-0001", "q"), ("D901-s-0000", "q")]),
    )
    assert cited_chunk_ids(a) == ["D901-s-0001", "D903-s-0000", "D901-s-0000"]
    assert cited_docs(a) == {"D901", "D903"}
    assert cited_chunk_ids(None) == []


# --- refusal -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final_status", "status", "expected"),
    [
        ("valid", "answered", False),
        ("valid", "partial", False),  # partial is an answer, as in S2-14
        ("repaired", "answered", False),
        ("valid", "not_in_documents", True),
        ("failed", None, True),  # rejected after the repair
        ("error", None, True),
        ("no_context", None, True),
        ("below_threshold", None, True),  # the score gate
    ],
)
def test_refused_means_no_answer_shown_or_not_in_documents(final_status, status, expected):
    a = answer(status, ("x", [("D901-s-0000", "q")])) if status in ("answered", "partial") else None
    if status == "partial":
        a["missing"] = "the method"
    if status == "not_in_documents":
        a = answer(status, missing="x")
    assert refused(row(a, final_status)) is expected


def test_refusal_is_correct_only_for_unanswerable_questions():
    answered = row(GOOD)
    assert refusal_correct(UNANSWERABLE, NOT_IN_DOCUMENTS)
    assert not refusal_correct(UNANSWERABLE, answered)
    assert refusal_correct(ANSWERABLE, answered)
    assert not refusal_correct(ANSWERABLE, NOT_IN_DOCUMENTS)
    assert not refusal_correct(SIMILAR, row(None, "below_threshold"))


# --- sources -----------------------------------------------------------------------------------


def test_document_level_and_quote_level_source():
    # D901-s-0001 is a chunk of the evidence document, but it does not hold the quote
    other_section = answer("answered", ("165 °C", [("D901-s-0001", "liquid")]))
    assert right_source(ANSWERABLE, other_section) is True
    assert evidence_cited(ANSWERABLE, other_section, CHUNKS) is False
    # the evidence chunk: the quote matches across the line break
    assert right_source(ANSWERABLE, GOOD) is True
    assert evidence_cited(ANSWERABLE, GOOD, CHUNKS) is True
    # a different document holds no evidence
    sibling = answer("answered", ("1 mm", [("D903-s-0000", "1 mm")]))
    assert right_source(ANSWERABLE, sibling) is False
    assert evidence_cited(ANSWERABLE, sibling, CHUNKS) is False


def test_a_cited_chunk_that_is_not_known_holds_no_evidence():
    unknown = answer("answered", ("165 °C", [("D901-s-0099", "approx. 165 °C")]))
    assert right_source(ANSWERABLE, unknown) is True  # the document is still right
    assert evidence_cited(ANSWERABLE, unknown, CHUNKS) is False


def test_sources_are_not_scored_for_unanswerable_questions():
    assert right_source(UNANSWERABLE, GOOD) is None
    assert evidence_cited(UNANSWERABLE, GOOD, CHUNKS) is None
    assert wrong_source(UNANSWERABLE, GOOD) is None


def test_wrong_source_only_for_traps():
    both = answer("answered", ("3 mm", [("D904-s-0000", "3 mm"), ("D903-s-0000", "1 mm")]))
    assert wrong_source(SIMILAR, both) is True
    assert wrong_source(SIMILAR, answer("answered", ("3 mm", [("D904-s-0000", "3 mm")]))) is False
    assert wrong_source(ANSWERABLE, both) is None  # D903 is not wrong for a non-trap


# --- facts -------------------------------------------------------------------------------------


def test_key_facts_are_searched_in_claims_only():
    # the second fact is only in a quote and in `missing`: not found
    a = answer(
        "partial",
        ("The boiling point is about 165°C.", [("D901-s-0000", "(ASTM D 1120)")]),
        missing="ASTM D 1120",
    )
    assert facts_found(ANSWERABLE, a) == [True, False]  # second spelling "165°C"
    assert facts_found(ANSWERABLE, GOOD) == [True, True]
    assert facts_found(ANSWERABLE, None) == [False, False]


def test_key_facts_ignore_case_and_whitespace():
    a = answer("answered", ("approx. 165  °C,\nastm d 1120", [("D901-s-0000", "q")]))
    assert facts_found(ANSWERABLE, a) == [True, True]


def test_forbidden_facts_are_searched_in_claims_only():
    in_quote = answer("answered", ("3 mm", [("D904-s-0000", "3 mm"), ("D903-s-0000", "1 mm")]))
    in_claim = answer("answered", ("3 mm, not 1 mm", [("D904-s-0000", "3 mm")]))
    assert forbidden_found(SIMILAR, in_quote) == [False]
    assert forbidden_found(SIMILAR, in_claim) == [True]
    assert forbidden_found(OUTDATED, in_claim) == []  # no forbidden facts


# --- one question ------------------------------------------------------------------------------


def test_a_correct_answer_passes():
    s = score_question(ANSWERABLE, row(GOOD), CHUNKS)
    assert (s.outcome, s.refused, s.refusal_correct) == ("answered", False, True)
    assert (s.right_source, s.evidence_cited, s.wrong_source) == (True, True, None)
    assert (s.facts, s.facts_present, s.facts_total, s.all_facts) == ([True, True], 2, 2, True)
    assert s.passed


def test_an_answer_missing_one_key_fact_fails():
    a = answer("partial", ("About 165 °C.", [("D901-s-0000", "approx. 165 °C")]), missing="m")
    s = score_question(ANSWERABLE, row(a), CHUNKS)
    assert (s.outcome, s.refused, s.facts_present, s.all_facts) == ("partial", False, 1, False)
    assert not s.passed


def test_the_right_facts_from_the_wrong_document_fail():
    a = answer("answered", ("165 °C, ASTM D 1120", [("D903-s-0000", "1 mm")]))
    s = score_question(ANSWERABLE, row(a), CHUNKS)
    assert (s.all_facts, s.right_source) == (True, False)
    assert not s.passed


def test_a_refused_answerable_question_misses_every_fact_and_source():
    for r in (NOT_IN_DOCUMENTS, row(None, "failed"), row(None, "below_threshold")):
        s = score_question(ANSWERABLE, r, CHUNKS)
        assert (s.refused, s.refusal_correct, s.right_source) == (True, False, False)
        assert (s.evidence_cited, s.facts, s.passed) == (False, [False, False], False)


def test_a_rejected_answer_scores_nothing_even_if_the_row_still_holds_it():
    # rejected after the repair: whatever the model wrote is never shown, so it earns nothing
    s = score_question(ANSWERABLE, row(GOOD, "failed"), CHUNKS)
    assert (s.outcome, s.refused, s.right_source, s.evidence_cited) == (
        "rejected",
        True,
        False,
        False,
    )
    assert (s.facts, s.passed) == ([False, False], False)


def test_a_trap_citing_both_documents_passes_and_reports_the_wrong_one():
    # user's choice: a wrong citation is reported, not a failure
    a = answer(
        "answered",
        ("Die Wafer sind 3 mm dick.", [("D904-s-0000", "3 mm"), ("D903-s-0000", "1 mm")]),
    )
    s = score_question(SIMILAR, row(a), CHUNKS)
    assert (s.right_source, s.wrong_source, s.forbidden, s.fell_for_forbidden) == (
        True,
        True,
        [False],
        False,
    )
    assert s.passed


def test_a_trap_with_the_forbidden_fact_fails():
    a = answer("answered", ("Die Wafer sind 3 mm oder 1 mm dick.", [("D904-s-0000", "3 mm")]))
    s = score_question(SIMILAR, row(a), CHUNKS)
    assert (s.all_facts, s.right_source, s.wrong_source, s.fell_for_forbidden) == (
        True,
        True,
        False,
        True,
    )
    assert not s.passed


def test_an_outdated_trap_answered_from_the_old_version_fails():
    a = answer("answered", ("By June 1, 2015.", [("D902-s-0000", "June 1, 2015")]))
    s = score_question(OUTDATED, row(a), CHUNKS)
    assert (s.right_source, s.wrong_source, s.facts, s.forbidden) == (False, True, [False], [])
    assert not s.passed


@pytest.mark.parametrize(
    ("r", "passed"),
    [
        (NOT_IN_DOCUMENTS, True),
        (row(None, "below_threshold"), True),
        (row(None, "failed"), True),  # nothing shown is a refusal, as in S2-14
        (row(GOOD), False),
    ],
)
def test_an_unanswerable_question_passes_when_refused(r, passed):
    s = score_question(UNANSWERABLE, r, CHUNKS)
    assert (s.refusal_correct, s.passed) == (passed, passed)
    assert (s.right_source, s.evidence_cited, s.wrong_source) == (None, None, None)
    assert (s.facts, s.all_facts) == ([], None)


# --- rates -------------------------------------------------------------------------------------


def six_scores() -> list:
    both = answer(
        "answered",
        ("Die Wafer sind 3 mm dick.", [("D904-s-0000", "3 mm"), ("D903-s-0000", "1 mm")]),
    )
    old = answer("answered", ("By June 1, 2015.", [("D902-s-0000", "June 1, 2015")]))
    return [
        score_question(ANSWERABLE, row(GOOD), CHUNKS),  # s1 en: passed, 2 of 2 facts
        score_question(ANSWERABLE, NOT_IN_DOCUMENTS, CHUNKS),  # s2 en: false refusal, 0 of 2
        score_question(SIMILAR, row(both), CHUNKS),  # s3 de: passed, wrong doc also cited
        score_question(OUTDATED, row(old), CHUNKS),  # s4 en: old version, 0 of 1, wrong doc
        score_question(UNANSWERABLE, NOT_IN_DOCUMENTS, CHUNKS),  # s5 uk: passed
        score_question(UNANSWERABLE, row(GOOD), CHUNKS),  # s6 uk: answered, failed
    ]


def f(n: int, total: int) -> dict:
    return {"n": n, "total": total, "rate": n / total if total else None}


def test_overall_rates_counted_by_hand():
    assert counts(six_scores()) == {
        "passed": f(3, 6),  # s1, s3, s5
        "unanswerable_refused": f(1, 2),  # s5 of s5, s6
        "answerable_refused": f(1, 4),  # s2 of s1-s4
        "right_source": f(2, 4),  # s1, s3
        "evidence_cited": f(2, 4),  # s1 (D901-s-0000), s3 (D904-s-0000)
        "all_facts": f(2, 4),  # s1, s3
        "facts_present": f(3, 6),  # 2 + 0 + 1 + 0 of 2 + 2 + 1 + 1
        "wrong_source": f(2, 2),  # s3 and s4 cite a wrong document
        "forbidden": f(0, 1),  # only s3's question has forbidden facts, and s3 avoids them
    }


def test_rates_per_category_and_language_counted_by_hand():
    summary = summarize(six_scores())
    assert list(summary["by_category"]) == [c.value for c in Category]
    assert summary["by_category"]["answerable"]["passed"] == f(1, 2)  # s1 of s1, s2
    assert summary["by_category"]["answerable"]["facts_present"] == f(2, 4)
    assert summary["by_category"]["answerable"]["wrong_source"] == f(0, 0)
    assert summary["by_category"]["similar_doc"]["passed"] == f(1, 1)
    assert summary["by_category"]["outdated_version"]["right_source"] == f(0, 1)
    assert summary["by_category"]["outdated_version"]["forbidden"] == f(0, 0)
    assert summary["by_category"]["unanswerable"]["unanswerable_refused"] == f(1, 2)
    assert summary["by_category"]["unanswerable"]["all_facts"] == f(0, 0)
    assert list(summary["by_language"]) == ["de", "en", "uk"]
    assert summary["by_language"]["en"]["passed"] == f(1, 3)  # s1 of s1, s2, s4
    assert summary["by_language"]["en"]["answerable_refused"] == f(1, 3)  # s2
    assert summary["by_language"]["de"]["wrong_source"] == f(1, 1)  # s3
    assert summary["by_language"]["uk"]["passed"] == f(1, 2)  # s5 of s5, s6
    assert summary["overall"] == counts(six_scores())


def test_no_scores_give_empty_rates():
    assert counts([])["passed"] == f(0, 0)
    assert summarize([])["by_category"] == {}


# --- the committed test set --------------------------------------------------------------------

STRUCTURE_CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"


@pytest.fixture(scope="module")
def corpus_chunks() -> dict[str, Chunk]:
    return {c.chunk_id: c for c in load_chunks(STRUCTURE_CHUNKS)}


@pytest.fixture(scope="module")
def test_questions() -> list[TestQuestion]:
    return load_test_set()


def ideal_answer(question: TestQuestion, chunks: dict[str, Chunk]) -> dict:
    """States the first spelling of every key fact and cites a chunk that holds the first
    evidence quote: what a correct answer looks like by the test set's own labels."""
    first = question.evidence[0]
    target = next(c for c in chunks.values() if is_hit(c, first))
    text = "; ".join(fact[0] for fact in question.key_facts)
    return answer("answered", (text, [(target.chunk_id, first.quote)]))


def test_an_answer_built_from_the_labels_passes_every_answerable_question(
    test_questions, corpus_chunks
):
    answerable = [q for q in test_questions if q.answerable]
    assert len(answerable) == 18  # 8 answerable, 5 similar_doc, 5 outdated_version
    for q in answerable:
        s = score_question(q, row(ideal_answer(q, corpus_chunks)), corpus_chunks)
        assert s.passed, q.id
        assert (s.right_source, s.evidence_cited, s.all_facts) == (True, True, True), q.id
        assert not s.fell_for_forbidden, q.id
        assert s.wrong_source in (None, False), q.id


def test_refusing_everything_passes_exactly_the_unanswerable_questions(
    test_questions, corpus_chunks
):
    scores = [score_question(q, NOT_IN_DOCUMENTS, corpus_chunks) for q in test_questions]
    passed = {s.question_id for s in scores if s.passed}
    assert passed == {q.id for q in test_questions if not q.answerable}
    assert len(passed) == 6
    rates = counts(scores)
    assert rates["unanswerable_refused"] == f(6, 6)
    assert rates["answerable_refused"] == f(18, 18)
