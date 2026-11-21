"""Search the index the way the answer pipeline does, S2-14 (moved out of scripts/ask.py).

`open_retriever` checks that the index is current enough (BM25 vectors from S2-07, filter
fields from S2-08), loads the embedding model and, with `[search] rerank`, the reranker once,
and keeps the Qdrant store open. `make ask` and `make eval-refusal` use it, so the eval sees
the same hits a user gets.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, Config
from factory_docs.embedding import Embedder
from factory_docs.index import BM25, NO_FILTER, Hit, SearchFilter, search
from factory_docs.rerank import Reranker

FILTER_FIELDS = "is_latest"  # a payload field every index built since S2-08 has


class IndexNotReady(Exception):
    """The index is missing or was built by an older `make index`."""


@dataclass
class Retriever:
    config: Config
    client: QdrantClient
    collection: str
    embedder: Embedder | None  # None for BM25 alone
    reranker: Reranker | None

    def search(self, question: str, search_filter: SearchFilter = NO_FILTER) -> list[Hit]:
        """The best `[answer] k` hits, best first, with the [search] mode and reranking."""
        return search(
            self.client,
            self.collection,
            question,
            self.embedder,
            self.config.answer.k,
            self.config.search,
            self.config.search.mode,
            search_filter,
            self.reranker,
            self.config.reranking.candidates,
        )

    def info(self, search_filter: SearchFilter = NO_FILTER) -> dict:
        """How hits were found, for the answer record."""
        return {
            "collection": self.collection,
            "mode": self.config.search.mode,
            "embedding_model": self.embedder.model_name if self.embedder else None,
            "reranker": self.reranker.model_name if self.reranker else None,
            "rerank_candidates": self.config.reranking.candidates if self.reranker else None,
            "filter": search_filter.describe(),
        }


def check_collection(client: QdrantClient, collection: str) -> None:
    if BM25 not in (client.get_collection(collection).config.params.sparse_vectors or {}):
        raise IndexNotReady("The index has no BM25 vectors (built before S2-07); run `make index`")
    sample = client.scroll(collection, limit=1, with_payload=True)[0]
    if sample and FILTER_FIELDS not in (sample[0].payload or {}):
        raise IndexNotReady("The index has no filter fields (built before S2-08); run `make index`")


@contextmanager
def open_retriever(config: Config, collection: str | None = None) -> Iterator[Retriever]:
    """A retriever over `collection` (default `[answer] collection`); models load once."""
    from factory_docs.embedding import FastEmbedder  # slow imports
    from factory_docs.rerank import FastReranker

    collection = collection or config.answer.collection
    store = ROOT / config.index.path
    if not store.exists():
        raise IndexNotReady(f"No index at {config.index.path}; run `make index` first")
    embedder = None if config.search.mode == "bm25" else FastEmbedder.from_config(config)
    reranker = FastReranker.from_config(config) if config.search.rerank else None
    client = QdrantClient(path=str(store))
    try:
        check_collection(client, collection)
        yield Retriever(config, client, collection, embedder, reranker)
    finally:
        client.close()
