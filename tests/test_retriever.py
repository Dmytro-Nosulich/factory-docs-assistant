"""The shared retriever (S2-14) searches like `make search` and reports how, on an in-memory
index with the fake embedder and reranker of the index tests."""

from itertools import islice

import pytest
from index_helpers import HashEmbedder, build, make_chunk
from qdrant_client import QdrantClient, models
from test_index import FakeReranker

from factory_docs.chunking import Chunk
from factory_docs.config import ROOT, load_config
from factory_docs.index import SearchFilter, search
from factory_docs.retriever import IndexNotReady, Retriever, check_collection

TEXTS = {
    "X-01": "Flash point 243 degrees, open cup.",
    "X-02": "UN 1006 argon compressed.",
    "X-03": "Store in a cool place.",
}


@pytest.fixture
def client():
    with (ROOT / "data" / "chunks" / "chunks.jsonl").open(encoding="utf-8") as f:
        base = Chunk.model_validate_json(next(islice(f, 1)))
    client = QdrantClient(":memory:")
    chunks = [
        make_chunk(base, f"{i}-s-0000", text, doc_id=i[:1] + "9" + i[-2:])
        for i, text in TEXTS.items()
    ]
    build(client, "structure", chunks)
    yield client
    client.close()


def test_retriever_matches_search_with_the_answer_settings(client):
    config = load_config()
    embedder, reranker = HashEmbedder(), FakeReranker({"UN 1006": 5.0})
    retriever = Retriever(config, client, "structure", embedder, reranker)
    hits = retriever.search("UN number of argon")
    expected = search(
        client,
        "structure",
        "UN number of argon",
        embedder,
        config.answer.k,
        config.search,
        config.search.mode,
        reranker=reranker,
        candidates=config.reranking.candidates,
    )
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in expected]
    assert hits[0].chunk.text.startswith("UN 1006") and hits[0].score == 5.0
    info = retriever.info(SearchFilter(languages=("de",)))
    assert info == {
        "collection": "structure",
        "mode": config.search.mode,
        "embedding_model": HashEmbedder.model_name,
        "reranker": FakeReranker.model_name,
        "rerank_candidates": config.reranking.candidates,
        "filter": "language de",
    }


def test_index_without_bm25_is_not_ready():
    client = QdrantClient(":memory:")
    client.create_collection(
        "structure",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )
    with pytest.raises(IndexNotReady, match="no BM25 vectors"):
        check_collection(client, "structure")
    client.close()
