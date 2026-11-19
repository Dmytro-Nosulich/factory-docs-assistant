"""The vector index (S2-06), offline: a fake embedder and an in-memory Qdrant, no downloads."""

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
from factory_docs.index import DENSE, build_collection, load_chunks, point_id, search

CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
WORD = re.compile(r"\w+")


class HashEmbedder:
    """Bag of words hashed into a small vector: texts sharing words get close vectors."""

    model_name = "fake/hash-bag-of-words"
    dimension = 64

    def __init__(self) -> None:
        self.calls: list[int] = []

    def vector(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in WORD.findall(text.casefold()):
            v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


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
    assert build_collection(client, "structure", chunks, HashEmbedder()) == len(chunks)
    points, _ = client.scroll("structure", limit=1000, with_payload=True, with_vectors=True)
    by_id = {p.id: p for p in points}
    for chunk in chunks:
        point = by_id[point_id(chunk.chunk_id)]
        assert point.payload == chunk.model_dump()
        assert len(point.vector[DENSE]) == HashEmbedder.dimension


def test_a_rebuild_replaces_the_collection_without_duplicates(client, chunks):
    build_collection(client, "structure", chunks, HashEmbedder())
    first = sorted(p.id for p in client.scroll("structure", limit=1000)[0])
    assert build_collection(client, "structure", chunks[:10], HashEmbedder()) == 10
    assert build_collection(client, "structure", chunks, HashEmbedder()) == len(chunks)
    assert sorted(p.id for p in client.scroll("structure", limit=1000)[0]) == first


def test_collections_are_separate(client, chunks):
    build_collection(client, "structure", chunks[:5], HashEmbedder())
    build_collection(client, "fixed", chunks[5:8], HashEmbedder())
    assert client.count("structure").count == 5
    assert client.count("fixed").count == 3


def test_search_returns_the_closest_chunk_first(client, chunks):
    build_collection(client, "structure", chunks, HashEmbedder())
    target = max(chunks, key=lambda c: len(c.text))
    hits = search(client, "structure", target.text, HashEmbedder(), k=3)
    assert len(hits) == 3
    assert hits[0].chunk == target
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score >= hits[1].score >= hits[2].score


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
