"""Cross-encoder reranking, S2-08.

Vector search and BM25 score the query and a chunk separately and compare the results. A
cross-encoder reads the pair together, query and chunk in one input, and returns one relevance
score. That is more precise and far slower, so it only rescores the first-stage candidates
(`factory_docs.index.rerank`). Models run in-process with FastEmbed (decision 13). Code that
reranks takes the `Reranker` protocol, so tests use a fake.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from factory_docs.config import ROOT, Config, RerankerModel


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score per text, higher is more relevant; scores compare only within a query."""
        ...


class FastReranker:
    """A FastEmbed cross-encoder."""

    def __init__(
        self,
        model: RerankerModel,
        batch_size: int,
        cache_dir: Path,
        local_files_only: bool = False,  # True: fail instead of downloading
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # slow import

        self.model_name = model.name
        self.batch_size = batch_size
        self.model = TextCrossEncoder(
            model.name, cache_dir=str(cache_dir), local_files_only=local_files_only
        )

    @classmethod
    def from_config(cls, config: Config, key: str | None = None) -> "FastReranker":
        key = key or config.reranking.model
        return cls(
            config.reranker_models[key],
            config.reranking.batch_size,
            ROOT / config.embedding.cache_dir,
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.rerank(query, texts, batch_size=self.batch_size)]

    @property
    def token_limit(self) -> int | None:
        """The length FastEmbed truncates every query + chunk pair to, in tokens."""
        truncation = self.model.model.tokenizer.truncation
        return truncation["max_length"] if truncation else None

    def token_lengths(self, query: str, texts: Sequence[str]) -> list[int]:
        """Each query + text pair's length in this model's tokens, before truncation."""
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return [len(e.ids) for e in tokenizer.encode_batch([(query, t) for t in texts])]
