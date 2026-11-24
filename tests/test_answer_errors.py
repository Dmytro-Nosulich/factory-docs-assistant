"""The S2-21 error listing (`make answer-errors`): which questions it selects, where the evidence
sits in the context, the gate-off comparison, and the numbers REPORT.md quotes from the committed
run file. The unit cases are invented (D901-D903 are not corpus documents) and counted by hand."""

import json

import pytest

from evals.error_analysis import (
    collect,
    docs_before,
    evidence_ranks,
    format_cases,
    gate_off_refusals,
    language_mismatches,
    selected,
)
from factory_docs.config import ROOT
from factory_docs.index import load_chunks
from factory_docs.test_set import Category, TestQuestion, load_test_set
from tests.test_eval_answers import judgement, run_row, supports, verdict
from tests.test_scorers import chunk, evidence

CHUNKS = {
    c.chunk_id: c
    for c in [
        chunk("D901-s-0000", "Section 1: Product X"),
        chunk("D901-s-0001", "Flash point: -17 °C"),
        chunk("D902-s-0000", "Flash point: 12 °C"),
        chunk("D903-s-0000", "Boiling point: 100 °C"),
    ]
}
QUESTION = TestQuestion(
    id="T91",
    question="What is the flash point of product X?",
    language="en",
    category="answerable",
    expected_answer="-17 °C.",
    key_facts=[["-17 °C"]],
    evidence=[evidence("D901", "Flash point: -17 °C")],
)


def row(qid="T91", passed=True, quality=None, sent=None, answer=None, final="valid"):
    r = run_row(qid, "answerable", "en", passed, quality or verdict("correct"), supports(), 0.0)
    r.update(
        top_score=-2.0,
        sent=sent or ["D902-s-0000", "D901-s-0000", "D901-s-0001"],
        final_status=final,
        answer=answer,
    )
    return r


def test_selected_when_failed_or_not_judged_correct():
    assert not selected(row(passed=True, quality=verdict("correct")))
    assert selected(row(passed=False, quality=verdict("correct")))
    assert selected(row(passed=True, quality=verdict("partially_correct")))
    assert selected(row(passed=True, quality=judgement("invalid")))  # not judged


def test_evidence_rank_counts_from_one_and_needs_document_and_quote():
    # The evidence quote is in D901-s-0001 only; D902 has a flash point too, but is not evidence.
    assert evidence_ranks(QUESTION, ["D901-s-0001"], CHUNKS) == [1]
    assert evidence_ranks(QUESTION, ["D902-s-0000", "D901-s-0000", "D901-s-0001"], CHUNKS) == [3]
    assert evidence_ranks(QUESTION, ["D902-s-0000", "D903-s-0000"], CHUNKS) == []
    assert evidence_ranks(QUESTION, ["D999-s-0000"], CHUNKS) == []  # unknown chunk: no hit


def test_docs_before_stop_at_the_first_evidence_chunk():
    sent = ["D902-s-0000", "D901-s-0000", "D901-s-0001"]
    assert docs_before(sent, [3]) == ["D902", "D901"]
    assert docs_before(sent, [1]) == []
    assert docs_before(sent, []) == ["D902", "D901", "D901"]  # no evidence: every rank


def test_collect_reports_the_gate_and_the_gate_off_answer():
    run = {
        "settings": {"answer": {"min_rerank_score": -1.0}},
        "questions": [row(passed=False, quality=verdict("incorrect"), final="below_threshold")],
    }
    off_answer = {
        "status": "answered",
        "claims": [{"text": "The flash point is -17 °C.", "citations": []}],
        "missing": None,
    }
    gate_off = {"questions": [{**row(answer=off_answer), "id": "T91"}]}
    (case,) = collect(run, [QUESTION], CHUNKS, gate_off)
    assert case.gated  # -2.0 < -1.0
    assert case.evidence_ranks == [3]
    assert case.docs_before == ["D902", "D901"]
    assert case.shown.status == "below_threshold" and case.shown.language is None
    assert case.gate_off.status == "answered"
    assert case.gate_off.facts_in_claims == [True]
    assert case.gate_off.language == "en"
    # without gate-off answers the case still renders
    (alone,) = collect(run, [QUESTION], CHUNKS, None)
    assert alone.gate_off is None
    text = format_cases([alone], run, "run.json", None)
    assert "| T91 |" in text and "No gate-off answers given." in text


def test_gate_off_refusals_split_unanswerable_from_the_rest():
    unanswerable = QUESTION.model_copy(
        update={"id": "T92", "category": Category.unanswerable, "key_facts": [], "evidence": []}
    )
    refusal = {"status": "not_in_documents", "claims": [], "missing": "m"}
    answer = {"status": "answered", "claims": [{"text": "x", "citations": []}], "missing": None}
    gate_off = {
        "questions": [
            {"id": "T91", "final_status": "valid", "answer": refusal},
            {"id": "T92", "final_status": "valid", "answer": refusal},
        ]
    }
    assert gate_off_refusals(gate_off, [QUESTION, unanswerable]) == (["T92"], 1, ["T91"], 1)
    gate_off["questions"][0]["answer"] = answer
    assert gate_off_refusals(gate_off, [QUESTION, unanswerable]) == (["T92"], 1, [], 1)


def test_language_mismatches_cover_every_shown_answer():
    english = {"status": "answered", "claims": [{"text": "The value is 5.", "citations": []}]}
    german = {"status": "answered", "claims": [{"text": "Der Wert ist 5 und die", "citations": []}]}
    run = {"questions": [row("T91", answer=english), row("T92", answer=german), row("T93")]}
    assert language_mismatches(run) == (2, [("T92", "en", "de")])  # T93 shows no answer


# --- the committed run that REPORT.md's error analysis quotes ------------------------------

RUN_FILE = ROOT / "evals" / "results" / "runs" / "20261124T213807Z.json"
GATE_OFF = ROOT / "evals" / "results" / "calibration" / "answers.json"


@pytest.fixture(scope="module")
def committed():
    run = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    gate_off = json.loads(GATE_OFF.read_text(encoding="utf-8"))
    questions = load_test_set()
    chunks = {c.chunk_id: c for c in load_chunks(ROOT / "data" / "chunks" / "chunks.jsonl")}
    cases = {c.question.id: c for c in collect(run, questions, chunks, gate_off)}
    return run, gate_off, questions, cases


def test_committed_run_selects_the_seven_questions_in_the_report(committed):
    _, _, _, cases = committed
    assert sorted(cases) == ["T01", "T02", "T04", "T06", "T11", "T12", "T17"]


def test_error_1_gate_refuses_with_the_evidence_in_the_context(committed):
    _, _, _, cases = committed
    expected = {"T01": (-1.87, [1]), "T02": (-1.64, [7]), "T04": (-2.84, [3])}
    for qid, (top, ranks) in expected.items():
        c = cases[qid]
        assert round(c.top_score, 2) == top and c.threshold == -1.03 and c.gated
        assert c.evidence_ranks == ranks
        assert c.gate_off.status == "answered" and all(c.gate_off.facts_in_claims)


def test_error_2_and_3_evidence_last_and_value_only_in_missing(committed):
    _, _, _, cases = committed
    t11, t12 = cases["T11"], cases["T12"]
    assert t11.evidence_ranks == [10] and not t11.gated
    assert t11.shown.status == "not_in_documents" and t11.shown.facts_in_missing == [True]
    assert t11.forbidden_stated == [False]
    assert "D053-s-0001" not in t11.sent  # the chunk naming LZB 125
    assert t12.evidence_ranks == [10] and t12.gated and round(t12.top_score, 2) == -1.65
    assert [i for i, d in enumerate(t12.sent, 1) if d.startswith("D056-")] == [1, 3, 10]
    assert t12.gate_off.status == "not_in_documents" and t12.gate_off.facts_in_missing == [True]


def test_error_4_partial_answers_and_the_language_slip(committed):
    run, _, _, cases = committed
    for qid in ("T06", "T17"):
        c = cases[qid]
        assert c.passed and c.verdict == "partially_correct" and c.evidence_ranks == [1]
    assert cases["T17"].shown.language == "de" and cases["T17"].gate_off.language == "de"
    assert language_mismatches(run) == (15, [("T17", "en", "de")])


def test_gate_off_answers_refuse_every_unanswerable_and_two_others(committed):
    run, gate_off, questions, _ = committed
    assert gate_off_refusals(gate_off, questions) == (
        ["T19", "T20", "T21", "T22", "T23", "T24"],
        6,
        ["T11", "T12"],
        18,
    )
    gated = [r["id"] for r in run["questions"] if r["final_status"] == "below_threshold"]
    assert len(gated) == 9  # questions with no model call
