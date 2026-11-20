"""The index and search (S2-06, S2-07), offline: fake embedders and an in-memory Qdrant."""

import hashlib
import math
import re
from collections.abc import Sequence
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
    build_collection,
    hybrid_search,
    lexical_search,
    load_chunks,
    point_id,
    rrf,
    search,
    vector_search,
)
from factory_docs.lexical import Bm25Encoder

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
        assert point.payload == chunk.model_dump()
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


def make_chunk(base: Chunk, chunk_id: str, text: str) -> Chunk:
    return base.model_copy(update={"chunk_id": chunk_id, "text": text, "chars": len(text)})


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
