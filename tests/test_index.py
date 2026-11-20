"""The index and search (S2-06 to S2-08), offline: fake embedders, a fake reranker and an
in-memory Qdrant. One test runs the real reranker, only when it is already downloaded."""

import csv
import hashlib
import math
import re
from collections.abc import Sequence
from datetime import date
from itertools import islice

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from factory_docs.chunking import Chunk
from factory_docs.config import ROOT, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import (
    BM25,
    DENSE,
    NO_FILTER,
    Hit,
    SearchFilter,
    build_collection,
    filter_fields,
    hybrid_search,
    lexical_search,
    load_chunks,
    parse_doc_date,
    point_id,
    rerank,
    rrf,
    search,
    vector_search,
)
from factory_docs.lexical import Bm25Encoder
from factory_docs.rerank import FastReranker

CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
WORD = re.compile(r"\w+")


class HashEmbedder:
    """Bag of words hashed into a small vector: texts sharing words get close vectors."""

    model_name = "fake/hash-bag-of-words"
    dimension = 64

    def __init__(self) -> None:
        self.calls: list[int] = []

    def words(self, text: str) -> list[str]:
        return WORD.findall(text.casefold())

    def vector(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in self.words(text):
            v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


def build(client, name: str, chunks: Sequence[Chunk], embedder=None) -> int:
    bm25 = Bm25Encoder.fit([chunk.text for chunk in chunks], k1=1.2, b=0.75)
    return build_collection(client, name, chunks, embedder or HashEmbedder(), bm25)


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    with CHUNKS.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in islice(f, 40)]


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def test_committed_chunk_files_load_and_validate():
    config = load_config()
    for path in config.index.collections.values():
        loaded = load_chunks(ROOT / path)
        assert loaded and len({c.chunk_id for c in loaded}) == len(loaded)


def test_a_bad_line_stops_loading(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    path.write_text(chunks[0].model_dump_json() + "\n" + '{"chunk_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)


def test_a_duplicate_chunk_id_stops_loading(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    path.write_text((chunks[0].model_dump_json() + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        load_chunks(path)


def test_point_ids_are_stable_uuids_and_distinct():
    assert point_id("D001-s-0000") == point_id("D001-s-0000")
    assert point_id("D001-s-0000") != point_id("D001-s-0001")
    assert len(point_id("D001-s-0000")) == 36


def test_every_chunk_becomes_one_point_with_all_metadata(client, chunks):
    bm25 = Bm25Encoder.fit([chunk.text for chunk in chunks], k1=1.2, b=0.75)
    assert build(client, "structure", chunks) == len(chunks)
    points, _ = client.scroll("structure", limit=1000, with_payload=True, with_vectors=True)
    by_id = {p.id: p for p in points}
    for chunk in chunks:
        point = by_id[point_id(chunk.chunk_id)]
        assert point.payload == chunk.model_dump() | filter_fields(chunk, set())
        assert len(point.vector[DENSE]) == HashEmbedder.dimension
        assert point.vector[BM25] == bm25.encode_passage(chunk.text)


def test_a_rebuild_replaces_the_collection_without_duplicates(client, chunks):
    build(client, "structure", chunks)
    first = sorted(p.id for p in client.scroll("structure", limit=1000)[0])
    assert build(client, "structure", chunks[:10]) == 10
    assert build(client, "structure", chunks) == len(chunks)
    assert sorted(p.id for p in client.scroll("structure", limit=1000)[0]) == first


def test_collections_are_separate(client, chunks):
    build(client, "structure", chunks[:5])
    build(client, "fixed", chunks[5:8])
    assert client.count("structure").count == 5
    assert client.count("fixed").count == 3


def test_vector_search_returns_the_closest_chunk_first(client, chunks):
    build(client, "structure", chunks)
    target = max(chunks, key=lambda c: len(c.text))
    hits = vector_search(client, "structure", target.text, HashEmbedder(), k=3)
    assert len(hits) == 3
    assert hits[0].chunk == target
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score >= hits[1].score >= hits[2].score


def make_chunk(base: Chunk, chunk_id: str, text: str, **fields) -> Chunk:
    update = {"chunk_id": chunk_id, "text": text, "chars": len(text), **fields}
    return base.model_copy(update=update)


def test_bm25_score_is_qdrant_idf_times_the_term_weight(client, chunks):
    tiny = [
        make_chunk(chunks[0], "T-1", "acid water"),
        make_chunk(chunks[0], "T-2", "acid"),
        make_chunk(chunks[0], "T-3", "base"),
    ]
    build(client, "structure", tiny)
    hits = lexical_search(client, "structure", "ACID?", k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["T-2", "T-1"]  # T-3 has no query term
    assert [hit.bm25_rank for hit in hits] == [1, 2]
    # n = 3 chunks, "acid" in 2: IDF = ln((3 − 2 + 0.5) / (2 + 0.5) + 1) = ln 1.6
    # T-2: 1 token, avg_len 4/3: norm = 1.2 · (0.25 + 0.75 · 0.75) = 0.975
    assert hits[0].score == pytest.approx(math.log(1.6) * 2.2 / (1 + 0.975), rel=1e-5)


def test_a_rare_term_outweighs_a_common_one(client, chunks):
    tiny = [make_chunk(chunks[0], f"T-{i}", "water") for i in range(5)]
    tiny.append(make_chunk(chunks[0], "T-rare", "water H226"))
    tiny.append(make_chunk(chunks[0], "T-common", "water water"))
    build(client, "structure", tiny)
    assert lexical_search(client, "structure", "water H226", k=1)[0].chunk.chunk_id == "T-rare"


def test_a_query_without_words_finds_nothing_lexically(client, chunks):
    build(client, "structure", chunks)
    assert lexical_search(client, "structure", "?!", k=5) == []


def test_rrf_matches_hand_computed_scores():
    fused = dict(rrf([["a", "b", "c"], ["c", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)  # 0.032522
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)  # 0.032266
    assert fused["b"] == pytest.approx(1 / 62)  # only one list
    assert [key for key, _ in rrf([["a", "b", "c"], ["c", "a"]], k=60)] == ["a", "c", "b"]


def test_rrf_breaks_ties_by_key_whatever_the_list_order():
    assert rrf([["b"], ["a"]], k=60) == rrf([["a"], ["b"]], k=60)
    assert [key for key, _ in rrf([["b"], ["a"]], k=60)] == ["a", "b"]
    assert rrf([[], []], k=60) == []


class ConceptEmbedder(HashEmbedder):
    """Behaves like a real embedding in two ways that matter here: every number looks the
    same (embeddings blur codes), and a few words share a meaning across wording."""

    dimension = 512
    CONCEPTS = {"swallowing": "<ingestion>", "swallowed": "<ingestion>", "ingestion": "<ingestion>"}

    def words(self, text: str) -> list[str]:
        return [
            "<number>" if any(ch.isdigit() for ch in word) else self.CONCEPTS.get(word, word)
            for word in super().words(text)
        ]


@pytest.fixture
def sds_like(client, chunks):
    texts = {
        "X-01": "Section 3 composition. Solvent naphtha (petroleum), heavy aromatic, "
        "CAS 64742-94-5, content 10 to 25 percent of the mixture by weight.",
        "X-02": "CAS 7727-37-9 EC 231-783-9",
        "X-03": "UN 1066 class 2.2",
        "X-04": "CAS 7440-37-1 EC 231-147-0",
        "X-05": "Ingestion: rinse mouth with water. Never induce vomiting.",
        "X-06": "After use, close the container tightly.",
        "X-07": "Store in a cool, well-ventilated place.",
        "X-08": "Wear protective gloves and eye protection.",
    }
    build(
        client,
        "structure",
        [make_chunk(chunks[0], cid, text) for cid, text in texts.items()],
        ConceptEmbedder(),
    )
    return client


def ids(hits) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_a_term_found_only_lexically_reaches_the_hybrid_results(sds_like):
    query, target, k = "64742-94-5", "X-01", 3
    assert target not in ids(vector_search(sds_like, "structure", query, ConceptEmbedder(), k))
    assert ids(lexical_search(sds_like, "structure", query, k)) == [target]
    hybrid = hybrid_search(sds_like, "structure", query, ConceptEmbedder(), k, 50, 60)
    assert ids(hybrid)[0] == target
    assert (hybrid[0].vector_rank, hybrid[0].bm25_rank) == (4, 1)  # vector rank 4 is outside k


def test_a_paraphrase_found_only_by_vectors_reaches_the_hybrid_results(sds_like):
    query, target, k = "after swallowing", "X-05", 3
    assert target in ids(vector_search(sds_like, "structure", query, ConceptEmbedder(), k))
    assert target not in ids(lexical_search(sds_like, "structure", query, 50))  # no shared word
    hybrid = hybrid_search(sds_like, "structure", query, ConceptEmbedder(), k, 50, 60)
    assert target in ids(hybrid)
    hit = next(hit for hit in hybrid if hit.chunk.chunk_id == target)
    assert hit.bm25_rank is None and hit.vector_rank is not None


def test_hybrid_scores_are_the_rrf_of_both_ranks(sds_like):
    for hit in hybrid_search(
        sds_like, "structure", "after swallowing", ConceptEmbedder(), 8, 50, 60
    ):
        expected = sum(1 / (60 + r) for r in (hit.vector_rank, hit.bm25_rank) if r is not None)
        assert hit.score == pytest.approx(expected)


def test_search_dispatches_by_mode(sds_like):
    settings = load_config().search
    lexical = search(sds_like, "structure", "64742-94-5", None, 3, settings, "bm25")
    assert ids(lexical) == ["X-01"]
    vector = search(sds_like, "structure", "64742-94-5", ConceptEmbedder(), 3, settings, "vector")
    assert "X-01" not in ids(vector)
    default = search(sds_like, "structure", "64742-94-5", ConceptEmbedder(), 3, settings)
    assert settings.mode == "hybrid" and ids(default)[0] == "X-01"
    with pytest.raises(ValueError, match="needs an embedder"):
        search(sds_like, "structure", "x", None, 3, settings, "hybrid")


class RecordingModel:
    """Stands in for fastembed.TextEmbedding and records what it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts, batch_size: int = 256):
        import numpy as np

        for text in texts:
            self.seen.append(text)
            yield np.array([1.0, 0.0])


def fake_fastembedder(query_prefix: str, passage_prefix: str) -> FastEmbedder:
    embedder = object.__new__(FastEmbedder)  # skips __init__, which would download a model
    embedder.model_name = "fake"
    embedder.query_prefix = query_prefix
    embedder.passage_prefix = passage_prefix
    embedder.batch_size = 8
    embedder.dimension = 2
    embedder.model = RecordingModel()
    return embedder


def test_fastembedder_adds_the_configured_prefixes():
    embedder = fake_fastembedder("query: ", "passage: ")
    assert embedder.embed_passages(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.embed_query("q") == [1.0, 0.0]
    assert embedder.model.seen == ["passage: a", "passage: b", "query: q"]


def test_fastembedder_without_prefixes_sends_the_text_unchanged():
    embedder = fake_fastembedder("", "")
    embedder.embed_passages(["Flammpunkt"])
    embedder.embed_query("Flammpunkt?")
    assert embedder.model.seen == ["Flammpunkt", "Flammpunkt?"]


def test_every_compared_model_has_a_table(tmp_path):
    config = load_config()
    assert set(config.embedding.compare) <= set(config.embedding_models)
    e5 = config.embedding_models["e5-large"]
    assert (e5.query_prefix, e5.passage_prefix) == ("query: ", "passage: ")


def config_with(tmp_path, old: str, new: str):
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_comparing_a_missing_embedding_model_is_rejected(tmp_path):
    path = config_with(tmp_path, 'compare = ["minilm-l12"', 'compare = ["nope"')
    with pytest.raises(ValidationError, match="no \\[embedding_models.'nope'\\] table"):
        load_config(path)


def test_an_unknown_embedding_key_is_rejected(tmp_path):
    path = config_with(tmp_path, "batch_size = 32", "batch_size = 32\nbatchsize = 8")
    with pytest.raises(ValidationError, match="batchsize"):
        load_config(path)


def test_search_settings_are_validated(tmp_path):
    path = config_with(tmp_path, "bm25_b = 0.75", "bm25_b = 1.5")
    with pytest.raises(ValidationError, match="bm25_b"):
        load_config(path)
    path = config_with(tmp_path, 'mode = "hybrid"', 'mode = "fuzzy"')
    with pytest.raises(ValidationError, match="mode"):
        load_config(path)


# --- S2-08: filter fields --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v1.3, 2020-02-20", "2020-02-20"),
        ("as of 2024-05-01", "2024-05-01"),
        ("KA01385D/06/EN/02.19, 2019-02-01", "2019-02-01"),
        ("publication 98476041 0813; declaration 2013-07-15", "2013-07-15"),
        ("2015 MSGP, file posted 2015-11", "2015-11-01"),
        ("2021 MSGP, February 2021", "2021-02-01"),
        ("2014 (from file name)", "2014-01-01"),
        ("no date", None),
        ("v1.40.3", None),
    ],
)
def test_the_document_date_comes_from_version_or_date(text, expected):
    assert parse_doc_date(text) == expected


def test_every_document_in_scope_has_a_date_except_the_three_undated_ones():
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["in_rag_index"] == "yes"]
    undated = [row["id"] for row in rows if parse_doc_date(row["version_or_date"]) is None]
    assert undated == ["D016", "D017", "D018"]  # version_or_date is "no date"


def test_only_the_three_old_versions_are_not_latest_in_the_committed_chunks():
    for chunks_file in load_config().index.collections.values():
        chunks = load_chunks(ROOT / chunks_file)
        superseded = {c.supersedes for c in chunks if c.supersedes}
        not_latest = {c.doc_id for c in chunks if not filter_fields(c, superseded)["is_latest"]}
        assert not_latest == {"D066", "D071", "D074"}


def test_a_multilingual_document_counts_for_each_of_its_languages(chunks):
    assert filter_fields(chunks[0].model_copy(update={"language": "de/en"}), set())[
        "languages"
    ] == ["de", "en"]
    assert filter_fields(chunks[0], set())["languages"] == [chunks[0].language]


# --- S2-08: filters exclude documents ----------------------------------------------------------

# Every text shares the query words, so without a filter every document is a hit, and a missing
# document below can only be the filter's doing.
CATALOG = {
    "D901": dict(doc_type="sds", language="en", version_or_date="v1, 2020-02-20"),
    "D902": dict(doc_type="sds", language="uk", version_or_date="v2, 2024-07-26"),
    "D903": dict(doc_type="manual", language="de/en", version_or_date="2019-02-01"),
    "D904": dict(
        doc_type="regulation_guideline", language="en", version_or_date="as of 2024-05-01"
    ),
    "D905": dict(
        doc_type="regulation_guideline",
        language="en",
        version_or_date="as of 2026-11-02",
        supersedes="D904",
    ),
    "D906": dict(doc_type="sds", language="en", version_or_date="no date"),
}
QUERY = "flash point solvent"
MODES = ["vector", "bm25", "hybrid"]


@pytest.fixture
def catalog(client, chunks):
    docs = [
        make_chunk(
            chunks[0],
            f"{doc_id}-s-0000",
            f"{QUERY} {doc_id}",
            doc_id=doc_id,
            **{"supersedes": None, **fields},
        )
        for doc_id, fields in CATALOG.items()
    ]
    build(client, "structure", docs)
    return client


def found(client, search_filter: SearchFilter, mode: str, k: int = 10) -> set[str]:
    settings = load_config().search
    hits = search(client, "structure", QUERY, HashEmbedder(), k, settings, mode, search_filter)
    return {hit.chunk.doc_id for hit in hits}


@pytest.mark.parametrize("mode", MODES)
def test_without_a_filter_every_document_is_found(catalog, mode):
    assert found(catalog, NO_FILTER, mode) == set(CATALOG)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("search_filter", "expected"),
    [
        (SearchFilter(doc_types=("sds",)), {"D901", "D902", "D906"}),
        (SearchFilter(doc_types=("manual", "sds")), {"D901", "D902", "D903", "D906"}),
        (SearchFilter(languages=("uk",)), {"D902"}),
        (SearchFilter(languages=("de",)), {"D903"}),  # the de/en manual
        (SearchFilter(languages=("en",)), {"D901", "D903", "D904", "D905", "D906"}),
        (SearchFilter(latest_only=True), set(CATALOG) - {"D904"}),
        (SearchFilter(date_from=date(2024, 1, 1)), {"D902", "D904", "D905"}),  # D906 undated
        (SearchFilter(date_to=date(2020, 2, 20)), {"D901", "D903"}),  # the bound is inclusive
        (SearchFilter(date_from=date(2024, 5, 1), date_to=date(2024, 5, 1)), {"D904"}),
        (SearchFilter(doc_types=("sds",), date_from=date(2021, 1, 1)), {"D902"}),  # AND
        (SearchFilter(doc_types=("regulation_guideline",), latest_only=True), {"D905"}),
    ],
)
def test_filters_exclude_documents(catalog, mode, search_filter, expected):
    assert found(catalog, search_filter, mode) == expected


def test_a_filtered_search_still_returns_k_hits(catalog):
    # Qdrant filters before ranking, not after: k = 2 gives 2 SDS, not "2 hits minus non-SDS"
    for mode in MODES:
        assert len(found(catalog, SearchFilter(doc_types=("sds",)), mode, k=2)) == 2


def test_an_empty_filter_is_no_filter():
    assert SearchFilter().to_qdrant() is None
    assert SearchFilter().describe() == "none"


def test_a_filter_matching_nothing_returns_nothing(catalog):
    for mode in MODES:
        assert found(catalog, SearchFilter(languages=("ja",)), mode) == set()


# --- S2-08: reranking changes the order --------------------------------------------------------


class FakeReranker:
    """Scores each text by a table lookup; records what it was asked to score."""

    model_name = "fake/table-reranker"

    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.0) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[list[str]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(list(texts))
        return [next((v for k, v in self.scores.items() if k in t), self.default) for t in texts]


def test_reranking_moves_a_low_candidate_to_the_top(sds_like):
    settings = load_config().search
    query = "after swallowing"
    first = search(sds_like, "structure", query, ConceptEmbedder(), 8, settings)
    last = first[-1]
    reranker = FakeReranker({last.chunk.text: 9.0})
    reranked = search(
        sds_like,
        "structure",
        query,
        ConceptEmbedder(),
        3,
        settings,
        reranker=reranker,
        candidates=8,
    )
    assert ids(reranked)[0] == last.chunk.chunk_id != ids(first)[0]
    assert reranked[0].first_stage_rank == len(first)
    assert reranked[0].score == 9.0
    assert len(reranker.calls) == 1 and len(reranker.calls[0]) == 8  # all candidates scored
    assert len(reranked) == 3


def test_reranking_returns_only_candidates_and_keeps_order_on_ties(chunks):
    hits = [Hit(1.0 / i, make_chunk(chunks[0], f"T-{i}", f"text {i}")) for i in range(1, 6)]
    reranked = rerank("q", hits, FakeReranker(), k=5)  # every score 0.0: a tie
    assert ids(reranked) == ids(hits)
    assert [hit.first_stage_rank for hit in reranked] == [1, 2, 3, 4, 5]
    reranked = rerank("q", hits, FakeReranker({"text 4": 2.0, "text 2": 1.0}), k=3)
    assert ids(reranked) == ["T-4", "T-2", "T-1"]
    assert set(ids(rerank("q", hits, FakeReranker(), k=50))) == set(ids(hits))


def test_reranking_keeps_the_first_stage_ranks_of_each_retriever(sds_like):
    settings = load_config().search
    first = search(sds_like, "structure", "64742-94-5", ConceptEmbedder(), 5, settings)
    reranked = search(
        sds_like,
        "structure",
        "64742-94-5",
        ConceptEmbedder(),
        5,
        settings,
        reranker=FakeReranker(),
        candidates=5,
    )
    assert [(h.vector_rank, h.bm25_rank) for h in reranked] == [
        (h.vector_rank, h.bm25_rank) for h in first
    ]


def test_reranking_applies_after_the_filter(catalog):
    settings = load_config().search
    old = f"{QUERY} D904"  # the text of the superseded regulation
    reranker = FakeReranker({old: 9.0})
    hits = search(
        catalog,
        "structure",
        QUERY,
        HashEmbedder(),
        3,
        settings,
        search_filter=SearchFilter(latest_only=True),
        reranker=reranker,
        candidates=10,
    )
    assert "D904" not in {hit.chunk.doc_id for hit in hits}
    assert not any(old in text for text in reranker.calls[0])  # never even a candidate


def test_a_reranker_returning_the_wrong_number_of_scores_is_an_error(chunks):
    class Broken(FakeReranker):
        def score(self, query, texts):
            return [0.0]

    hits = [Hit(1.0, make_chunk(chunks[0], f"T-{i}", "x")) for i in range(2)]
    with pytest.raises(ValueError, match="1 scores for 2 hits"):
        rerank("q", hits, Broken(), k=2)


def test_an_unknown_reranker_key_is_rejected(tmp_path):
    path = config_with(tmp_path, 'compare = ["bge-m3"', 'compare = ["nope"')
    with pytest.raises(ValidationError, match="no \\[reranker_models.'nope'\\] table"):
        load_config(path)


def test_the_real_reranker_prefers_the_answer_in_german():
    """The chosen cross-encoder, offline: skipped unless `make compare-rerankers` or a reranked
    search has already downloaded it."""
    config = load_config()
    try:
        reranker = FastReranker(
            config.reranker_models[config.reranking.model],
            config.reranking.batch_size,
            ROOT / config.embedding.cache_dir,
            local_files_only=True,
        )
    except Exception as error:  # not downloaded: FastEmbed raises its own error types
        pytest.skip(f"reranker not in the cache: {error}")
    answer = "9.1 Flammpunkt: 23 °C (geschlossener Tiegel)"
    other = "1.1 Produktidentifikator: Handelsname RM 110"
    scores = reranker.score("Wie hoch ist der Flammpunkt von RM 110?", [other, answer])
    assert scores[1] > scores[0]
