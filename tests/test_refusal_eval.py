"""The S2-14 refusal measurement: threshold rule, refusal rates, answer quality, report.

The questions and rows are invented; every expected number is counted by hand in the comments.
"""

import pytest

from factory_docs.refusal_eval import (
    END,
    START,
    calls_avoided,
    gate_refuses,
    markdown,
    outcome,
    pick_threshold,
    quality,
    rates,
    report_section,
    summarize,
)
from factory_docs.retrieval_eval import replace_section

# --- threshold rule ----------------------------------------------------------------------------


def test_threshold_is_the_midpoint_below_the_lowest_answerable_score():
    # lowest answerable -1.8; unanswerable below it: -5.0 and -3.0, the highest -3.0
    assert pick_threshold([0.8, -1.8], [-0.3, -5.0, -3.0, None]) == pytest.approx(-2.4)


def test_no_unanswerable_below_every_answerable_means_no_gate():
    assert pick_threshold([0.8, -1.8], [-0.3, 2.0]) is None
    assert pick_threshold([0.8, -1.8], [-1.8]) is None  # equal is not below
    assert pick_threshold([], [-5.0]) is None
    assert pick_threshold([None], [-5.0]) is None


def test_rounding_never_refuses_an_answerable_question_or_passes_the_one_below():
    assert pick_threshold([1.004], [0.999]) == 1.0  # midpoint 1.0015 rounds to 1.00
    t = pick_threshold([1.001], [1.0])  # 1.0005 rounds to 1.00, which passes 1.0: keep it exact
    assert 1.0 < t <= 1.001


def test_gate_refuses_below_the_threshold_and_with_no_hit():
    assert gate_refuses(-3.0, -2.4) and not gate_refuses(-2.4, -2.4)
    assert gate_refuses(None, None) and not gate_refuses(-9.0, None)


# --- rates -------------------------------------------------------------------------------------


def q(id_, kind, language, top, evidence=()):
    return {
        "id": id_,
        "kind": kind,
        "language": language,
        "top_score": top,
        "evidence_docs": list(evidence),
        "evidence_rank": None,
        "question": f"{id_}?",
    }


SENTENCES = {  # what the invented answers say, in the language under test
    "en": "The value is not stated in the documents.",
    "de": "Der Wert ist in den Dokumenten nicht angegeben.",
    "uk": "Значення в документах не вказано.",
}


def row(id_, final, status=None, cited=(), lang="en", cost=0.001, total=1.0):
    answer = None
    if status is not None:
        text = SENTENCES[lang]
        claims = [] if status == "not_in_documents" else [{"text": text, "citations": []}]
        missing = text if status != "answered" else None
        answer = {"status": status, "claims": claims, "missing": missing}
    return {
        "id": id_,
        "final_status": final,
        "answer": answer,
        "cited_docs": list(cited),
        "cost_usd": cost,
        "total_s": total,
        "errors": [],
    }


QUESTIONS = [
    q("A1", "answerable", "en", 0.8, ["D050"]),
    q("A2", "answerable", "uk", -1.8, ["D006"]),
    q("U1", "near_miss", "en", -0.3),
    q("U2", "off_topic", "en", -5.0),
    q("U3", "off_topic", "de", -3.0),
    q("U4", "near_miss", "uk", None),  # no hit
]
ROWS = {
    "A1": row("A1", "valid", "answered", ["D050"], "en", total=1.0),
    "A2": row("A2", "repaired", "not_in_documents", lang="uk", total=3.0),  # a false refusal
    "U1": row("U1", "valid", "not_in_documents", lang="de", total=2.0),  # wrong language
    "U2": row("U2", "valid", "answered", ["D001"], "en", total=1.5),  # answers off-topic
    "U3": row("U3", "failed", cost=0.002, total=2.5),  # rejected after the repair
    "U4": row("U4", "no_context", cost=0.0, total=None),
}
T = -2.4  # pick_threshold of these scores, above


def counts(result):
    return (
        (result["unanswerable_refused"]["n"], result["unanswerable_refused"]["total"]),
        (result["answerable_refused"]["n"], result["answerable_refused"]["total"]),
    )


def test_outcome_names_the_answer_status_or_why_none_is_shown():
    assert [outcome(ROWS[i]) for i in ("A1", "A2", "U3", "U4")] == [
        "answered",
        "not_in_documents",
        "rejected",
        "no_context",
    ]
    assert outcome(row("X", "valid", "partial")) == "partial"


def test_model_only():
    # model refuses A2, U1, U3 (rejected), U4 (no hit); it answers A1 and U2
    r = rates(QUESTIONS, ROWS, T, "model")
    assert counts(r) == ((3, 4), (1, 2))
    assert (r["by_kind"]["near_miss"]["n"], r["by_kind"]["off_topic"]["n"]) == (2, 1)


def test_gate_only():
    # below -2.4: U2 -5.0, U3 -3.0; U4 has no hit; A2 -1.8 passes
    r = rates(QUESTIONS, ROWS, T, "gate")
    assert counts(r) == ((3, 4), (0, 2))
    assert (r["by_kind"]["near_miss"]["n"], r["by_kind"]["off_topic"]["n"]) == (1, 2)


def test_combined_refuses_when_either_refuses():
    # U1 by the model only, U2 by the gate only: all 4 unanswerable; A2 by the model
    r = rates(QUESTIONS, ROWS, T, "combined")
    assert counts(r) == ((4, 4), (1, 2))
    assert r["unanswerable_refused"]["rate"] == 1.0
    by_language = r["by_language"]
    assert (
        by_language["en"]["unanswerable_refused"]["n"],
        by_language["en"]["answerable_refused"]["n"],
    ) == (2, 0)
    assert by_language["uk"]["answerable_refused"]["n"] == 1
    assert by_language["de"]["answerable_refused"] == {"n": 0, "total": 0, "rate": None}


def test_partial_counts_as_answered():
    rows = {**ROWS, "A2": row("A2", "valid", "partial", ["D006"], "uk")}  # claims and missing
    assert rates(QUESTIONS, rows, T, "model")["answerable_refused"]["n"] == 0


def test_quality():
    result = quality(QUESTIONS, ROWS)
    # shown answers: A1 en/en, A2 uk/uk, U1 de/en, U2 en/en -> 3 of 4
    assert (result["language_match"]["n"], result["language_match"]["total"]) == (3, 4)
    # answered answerable questions: A1 only, which cites its evidence document
    assert (result["right_document"]["n"], result["right_document"]["total"]) == (1, 1)
    assert result["outcomes"] == {
        "answered": 2,
        "no_context": 1,
        "not_in_documents": 2,
        "rejected": 1,
    }
    assert result["cost_usd"] == pytest.approx(0.006)  # 4 x 0.001 + 0.002 + 0
    assert result["latency_p50_s"] == 2.0  # nearest rank of 1.0 1.5 2.0 2.5 3.0


def test_calls_avoided():
    assert calls_avoided(QUESTIONS, T)["n"] == 3  # U2, U3 and U4
    assert calls_avoided(QUESTIONS, None)["n"] == 1  # only the question with no hit


# --- report ------------------------------------------------------------------------------------


DATA = {
    "created_at": "2026-11-21T00:00:00+00:00",
    "machine": {"platform": "test"},
    "dev": {"path": "evals/dev_questions.toml", "sha256": "ab" * 32},
    "settings": {
        "model": "haiku-5.5",
        "slug": "x",
        "collection": "structure",
        "mode": "hybrid",
        "embedding_model": "e",
        "reranker": "r",
        "rerank_candidates": 30,
        "k": 10,
        "max_context_tokens": 4000,
        "prompts": {"v1": {"name": "answer", "version": "v1", "sha256": "cd" * 32}},
    },
    "threshold": T,
    "questions": QUESTIONS,
    "runs": {"v1": list(ROWS.values())},
}


def test_summary_and_report_show_the_numbers():
    summary = summarize(DATA)
    assert summary["threshold"] == T
    assert summary["versions"]["v1"]["decisions"]["combined"]["unanswerable_refused"]["n"] == 4
    section = report_section(DATA)
    assert "**-2.40**" in section
    assert "| v1 | gate or model | 4/4 (100%) | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) |" in section
    assert "| v1 | model only | 3/4 (75%) |" in section
    assert "1/2 of an answerable rate and 1/4 of an unanswerable one" in section
    md = markdown(DATA)
    assert "| U2 | off_topic | en | U2? | -5.00 | - | answered | en |" in md
    assert markdown(DATA) == md  # deterministic: --report twice gives the same file


def test_report_section_replaces_only_its_markers():
    text = f"# R\n{START}\nSTALE TEXT\n{END}\nkept\n"
    new = replace_section(text, report_section(DATA), START, END)
    assert "STALE TEXT" not in new and new.endswith(f"{END}\nkept\n")
    assert replace_section(new, report_section(DATA), START, END) == new
