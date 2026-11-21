"""Retrieval measurement (S2-10): the metrics against hand-computed values, the default-setup
rule, the REPORT.md section, one offline end-to-end run, and the committed results."""

import json
from collections.abc import Sequence

import pytest
from index_helpers import HashEmbedder, build, make_chunk
from pydantic import ValidationError
from qdrant_client import QdrantClient

from factory_docs.chunking import Chunk
from factory_docs.config import ROOT, RetrievalEvalSettings, load_config
from factory_docs.index import load_chunks
from factory_docs.retrieval_eval import (
    END,
    START,
    Setup,
    by_language,
    evaluate,
    first_hit_rank,
    mrr,
    pick_default,
    recall_at_k,
    replace_section,
    summary,
)
from factory_docs.retrieval_gold import Evidence, Question, load_gold

RESULTS = ROOT / "evals" / "results" / "retrieval.json"
QUOTE = "Heat may cause the containers to explode."


@pytest.fixture(scope="module")
def base() -> Chunk:
    with (ROOT / "data" / "chunks" / "chunks.jsonl").open(encoding="utf-8") as f:
        return Chunk.model_validate_json(f.readline())


def question(qid: str = "R01", language: str = "en", *evidence: tuple[str, str]) -> Question:
    evidence = evidence or (("D001", QUOTE),)
    return Question(
        id=qid,
        question="What can happen to nitrogen gas cylinders in a fire?",
        language=language,
        evidence=[Evidence(doc_id=d, section="5", page=4, quote=q) for d, q in evidence],
    )


def test_the_rank_is_the_first_chunk_from_the_evidence_document_that_holds_the_quote(base):
    chunks = [
        make_chunk(base, "D039-s-0001", QUOTE, doc_id="D039"),  # same quote, other document
        make_chunk(base, "D001-s-0001", "Nitrogen, compressed.", doc_id="D001"),  # no quote
        make_chunk(base, "D001-s-0002", "5.2 ... Heat may cause\nthe containers  to explode.\n"),
        make_chunk(base, "D001-s-0003", QUOTE),
    ]
    assert first_hit_rank(chunks, question(), depth=10) == 3


def test_any_alternative_evidence_counts(base):
    q = question("R13", "de", ("D044", "Reinheit in %: 99,999"), ("D042", "O2 ≥ 99,999"))
    chunks = [
        make_chunk(base, "D001-s-0001", "unrelated"),
        make_chunk(base, "D042-s-0001", "| O2 ≥ 99,999 |", doc_id="D042"),
    ]
    assert first_hit_rank(chunks, q, depth=10) == 2


def test_a_hit_below_the_depth_is_a_miss(base):
    chunks = [make_chunk(base, f"D001-s-{i:04d}", "unrelated") for i in range(3)]
    chunks.append(make_chunk(base, "D001-s-0009", QUOTE))
    assert first_hit_rank(chunks, question(), depth=4) == 4
    assert first_hit_rank(chunks, question(), depth=3) is None


def test_recall_and_mrr_match_hand_computed_values():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 2) == 0.5
    assert recall_at_k(ranks, 3) == 0.75
    assert recall_at_k(ranks, 10) == 0.75  # a miss never counts
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert summary(ranks, [1, 3]) == {
        "questions": 4,
        "recall": {"1": 0.25, "3": 0.75},
        "hits": {"1": 1, "3": 3},
        "mrr": pytest.approx(0.458333, rel=1e-5),
    }


def test_no_questions_is_an_error_not_zero():
    with pytest.raises(ValueError):
        recall_at_k([], 5)
    with pytest.raises(ValueError):
        mrr([])


def test_metrics_split_by_question_language():
    questions = [question("R01", "en"), question("R02", "uk"), question("R03", "en")]
    result = by_language({"R01": 1, "R02": None, "R03": 4}, questions, [1, 5])
    assert list(result) == ["en", "uk"]
    assert result["en"]["hits"] == {"1": 1, "5": 2}
    assert result["en"]["mrr"] == pytest.approx((1 + 1 / 4) / 2)
    assert result["uk"] == {
        "questions": 1,
        "recall": {"1": 0, "5": 0},
        "hits": {"1": 0, "5": 0},
        "mrr": 0,
    }


def test_setup_names_parse_into_mode_and_reranking():
    assert Setup.parse("hybrid+rerank") == Setup("hybrid+rerank", "hybrid", True)
    assert Setup.parse("bm25") == Setup("bm25", "bm25", False)
    with pytest.raises(ValueError):
        Setup.parse("dense")


@pytest.mark.parametrize(
    "ks, setups",
    [
        ([5, 1], ["vector"]),  # not ascending
        ([0, 5], ["vector"]),  # not positive
        ([1, 1], ["vector"]),  # not distinct
        ([1, 5], ["vector", "vector"]),
        ([1, 5], ["dense"]),
        ([1, 5], ["vector+rerank+rerank"]),
    ],
)
def test_bad_eval_settings_are_rejected(ks, setups):
    with pytest.raises(ValidationError):
        RetrievalEvalSettings(ks=ks, setups=setups)


def test_the_committed_config_measures_the_four_agreed_setups():
    settings = load_config().retrieval_eval
    assert settings.setups == ["vector", "bm25", "hybrid", "hybrid+rerank"]
    assert settings.depth == 10


def result(mrr_value: float, hits5: int) -> dict:
    return {"mrr": mrr_value, "hits": {"5": hits5}}


def test_the_default_rule_takes_mrr_then_recall_at_5_then_no_reranker():
    ks = [1, 5, 10]
    rerank = {"hybrid": False, "hybrid+rerank": True, "vector": False}
    assert (
        pick_default(
            {"vector": result(0.5, 9), "hybrid": result(0.6, 9), "hybrid+rerank": result(0.7, 9)},
            ks,
            rerank,
        )
        == "hybrid+rerank"
    )
    assert (
        pick_default({"hybrid": result(0.6, 9), "hybrid+rerank": result(0.6, 10)}, ks, rerank)
        == "hybrid+rerank"
    )
    assert (
        pick_default({"hybrid": result(0.6, 10), "hybrid+rerank": result(0.6, 10)}, ks, rerank)
        == "hybrid"
    )


def test_the_report_section_is_replaced_and_the_rest_kept():
    text = f"# Report\nintro\n{START}\n\nold numbers\n\n{END}\nafter\n"
    new = replace_section(text, "| new | table |")
    assert new == f"# Report\nintro\n{START}\n\n| new | table |\n\n{END}\nafter\n"
    assert replace_section(new, "| new | table |") == new


def test_a_report_without_markers_is_not_written():
    with pytest.raises(ValueError):
        replace_section("# Report\nno markers\n", "x")


def test_the_committed_report_has_the_markers():
    text = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    assert text.count(START) == 1 and text.count(END) == 1


class AnswerFirstReranker:
    """Scores the chunk that holds `answer` highest, everything else equal."""

    model_name = "fake/answer-first"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        return [1.0 if self.answer in text else 0.0 for text in texts]


@pytest.fixture
def catalog(base):
    client = QdrantClient(":memory:")
    answer = "Containers under fire: heat may make them burst. " + QUOTE
    chunks = [
        # The other document says it with the same quote and more query words: a trap
        make_chunk(
            base, "D039-s-0001", "Nitrogen gas cylinders in a fire. " + QUOTE, doc_id="D039"
        ),
        *(
            make_chunk(base, f"D010-s-{i:04d}", f"nitrogen gas cylinders storage note {i}")
            for i in range(12)
        ),
        make_chunk(base, "D001-s-0005", answer),
    ]
    build(client, "structure", chunks)
    yield client, answer
    client.close()


def test_one_setup_runs_end_to_end_on_an_in_memory_index(catalog):
    client, answer = catalog
    settings = load_config().search
    q = question()
    first = evaluate(
        client, "structure", [q], Setup.parse("bm25"), [1, 5], settings, None, None, 30
    )
    record = first["questions"]["R01"]
    assert record["top"][0] == "D039-s-0001"  # the trap is first and does not count
    assert len(record["top"]) == 5
    assert record["rank"] != 1  # holding the quote is not enough: the document must match

    reranker = AnswerFirstReranker(answer)
    reranked = evaluate(
        client,
        "structure",
        [q],
        Setup.parse("hybrid+rerank"),
        [1, 5],
        settings,
        HashEmbedder(),
        reranker,
        30,
    )
    assert reranker.calls == 1
    assert reranked["questions"]["R01"]["top"][0] == "D001-s-0005"
    assert reranked["summary"]["hits"] == {"1": 1, "5": 1}
    assert reranked["summary"]["mrr"] == 1.0
    assert reranked["by_language"]["en"]["mrr"] == 1.0


def test_a_rerank_setup_without_a_reranker_is_an_error(catalog):
    client, _ = catalog
    with pytest.raises(ValueError):
        evaluate(
            client,
            "structure",
            [question()],
            Setup.parse("hybrid+rerank"),
            [5],
            load_config().search,
            HashEmbedder(),
            None,
            30,
        )


@pytest.mark.skipif(not RESULTS.exists(), reason="no committed retrieval results")
def test_the_committed_results_follow_from_their_own_rankings():
    """Every rank and summary in retrieval.json is re-derived from the stored chunk IDs, the
    chunk files and the gold set, so the reported numbers trace to the saved search output."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    config = load_config()
    questions = {q.id: q for q in load_gold()}
    assert [q["id"] for q in data["gold"]["questions"]] == list(questions)
    ks = data["settings"]["ks"]
    for collection, runs in data["runs"].items():
        chunks = {c.chunk_id: c for c in load_chunks(ROOT / config.index.collections[collection])}
        assert list(runs) == data["settings"]["setups"]
        for name, run in runs.items():
            ranks = {}
            for qid, record in run["questions"].items():
                top = [chunks[chunk_id] for chunk_id in record["top"]]
                assert len(top) == ks[-1], f"{collection} {name} {qid}"
                ranks[qid] = first_hit_rank(top, questions[qid], ks[-1])
                assert ranks[qid] == record["rank"], f"{collection} {name} {qid}"
            assert run["summary"] == summary(list(ranks.values()), ks)
            assert run["by_language"] == by_language(ranks, list(questions.values()), ks)
