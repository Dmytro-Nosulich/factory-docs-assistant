"""Test helpers for an in-memory index: a fake embedder, a build shortcut and a chunk maker.
Shared by test_index.py and test_retrieval_eval.py."""

import hashlib
import math
import re
from collections.abc import Sequence

from factory_docs.chunking import Chunk
from factory_docs.index import build_collection
from factory_docs.lexical import Bm25Encoder

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


def make_chunk(base: Chunk, chunk_id: str, text: str, **fields) -> Chunk:
    update = {"chunk_id": chunk_id, "text": text, "chars": len(text), **fields}
    return base.model_copy(update=update)
