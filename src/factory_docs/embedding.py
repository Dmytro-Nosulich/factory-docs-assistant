"""Text embeddings, S2-06.

An embedding model turns a text into a vector so that texts with similar meaning get nearby
vectors. Models run in-process with FastEmbed (ONNX Runtime, decision 13): no server, no
per-call cost. FastEmbed adds no query/passage prefixes itself, so `FastEmbedder` adds the ones
in config.toml. Code that embeds takes the `Embedder` protocol, so tests use a fake.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from factory_docs.config import ROOT, Config, EmbeddingModel

Vector = list[float]


class Embedder(Protocol):
    """Like a Swift protocol: anything with these members can embed."""

    model_name: str
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class FastEmbedder:
    """A FastEmbed model with the prefixes its model card asks for."""

    def __init__(self, model: EmbeddingModel, batch_size: int, cache_dir: Path) -> None:
        from fastembed import TextEmbedding  # imported here: loading it takes a second

        self.model_name = model.name
        self.query_prefix = model.query_prefix
        self.passage_prefix = model.passage_prefix
        self.batch_size = batch_size
        self.model = TextEmbedding(model.name, cache_dir=str(cache_dir))
        self.dimension = self.model.embedding_size

    @classmethod
    def from_config(cls, config: Config, key: str | None = None) -> "FastEmbedder":
        key = key or config.embedding.model
        return cls(
            config.embedding_models[key],
            config.embedding.batch_size,
            ROOT / config.embedding.cache_dir,
        )

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        prefixed = [self.passage_prefix + text for text in texts]
        return [v.tolist() for v in self.model.embed(prefixed, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> Vector:
        return next(iter(self.model.embed([self.query_prefix + text]))).tolist()

    @property
    def token_limit(self) -> int | None:
        """The length FastEmbed truncates every input to, in tokens."""
        truncation = self.model.model.tokenizer.truncation
        return truncation["max_length"] if truncation else None

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Each passage's length in this model's tokens (with prefix), before truncation.

        FastEmbed cuts anything longer than `token_limit` without a warning, so this shows how
        much of a chunk the model never sees. Counts include special tokens; a static model
        adds none, so its counts are 2 too high.
        """
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        encodings = tokenizer.encode_batch([self.passage_prefix + text for text in texts])
        return [len(encoding.ids) for encoding in encodings]
