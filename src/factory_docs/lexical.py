"""Keyword search, S2-07: a language-neutral tokenizer and BM25 term weights as sparse vectors.

BM25 scores a chunk by the query words it contains. A rare word counts more than a common one
(IDF), and repeating a word helps less each time (saturation, k1), with a correction for chunk
length (b). Vector search finds paraphrases; BM25 finds exact strings such as a CAS number,
an H-code or a product code, which embeddings blur.

How it is stored: each chunk becomes a sparse vector {term id: saturated term frequency}, and
Qdrant multiplies by IDF at query time (Modifier.IDF), so IDF always matches the collection.
A query is {term id: 1} for each distinct word. This is the scheme FastEmbed's Bm25 model uses
with Qdrant; only the tokenizer is ours, because the corpus has nine languages and FastEmbed's
needs one stemmer language per model.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from qdrant_client import models

# A run of letters or digits (no underscore); runs joined by - . / stay one token, so
# 64742-48-9, H226, EUH066 and 1.1 survive whole. Trailing punctuation is not part of a token.
TOKEN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*")
# Unicode dashes (hyphen, non-breaking hyphen, figure dash, en dash, em dash, minus) to "-"
DASHES = str.maketrans(dict.fromkeys("‐‑‒–—−", "-"))


def tokenize(text: str) -> list[str]:
    """The same rules for every language: NFKC, dashes unified, casefold, word runs.

    No stemming and no stopwords: IDF already gives common words little weight. The cost is
    that inflected forms and German compound parts do not match (Flammpunkt is not Punkt).
    """
    text = unicodedata.normalize("NFKC", text).translate(DASHES).casefold()
    return TOKEN.findall(text)


def term_id(token: str) -> int:
    """A stable 32-bit ID for a token: Qdrant sparse vectors take integer indices.

    A hash, not a vocabulary table, so a query needs no state from the build. Two tokens can
    share an ID; `make index` records how many do.
    """
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")


def sparse(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])


def encode_query(text: str) -> models.SparseVector:
    """Weight 1 for each distinct query term; Qdrant adds IDF."""
    return sparse(dict.fromkeys((term_id(token) for token in tokenize(text)), 1.0))


class Bm25Encoder:
    """BM25 document weights for one collection; `avg_len` is that collection's mean length."""

    def __init__(self, k1: float, b: float, avg_len: float) -> None:
        self.k1 = k1
        self.b = b
        self.avg_len = avg_len

    @classmethod
    def fit(cls, texts: Sequence[str], k1: float, b: float) -> "Bm25Encoder":
        lengths = [len(tokenize(text)) for text in texts]
        return cls(k1, b, sum(lengths) / len(lengths) if lengths else 0.0)

    def encode_passage(self, text: str) -> models.SparseVector:
        """tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len)) for each term of `text`."""
        tokens = tokenize(text)
        counts = Counter(term_id(token) for token in tokens)  # colliding tokens add up
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / (self.avg_len or 1.0))
        return sparse({i: tf * (self.k1 + 1) / (tf + norm) for i, tf in counts.items()})


def vocabulary(texts: Sequence[str]) -> tuple[int, int]:
    """(distinct tokens, tokens whose term ID another token also has) over `texts`."""
    tokens = {token for text in texts for token in tokenize(text)}
    ids = Counter(term_id(token) for token in tokens)
    return len(tokens), sum(n for n in ids.values() if n > 1)
