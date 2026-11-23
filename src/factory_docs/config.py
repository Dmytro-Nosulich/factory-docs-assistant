"""Run configuration (config.toml) and secrets (.env), S1-07.

config.toml holds everything that changes a run: prompt version, model slugs, sampling,
timeouts and retries. Keys are never in it; they come from the environment or from .env.
"""

import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from factory_docs.chunking import MAX_CHARS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.toml"
ENV = ROOT / ".env"


class Settings(BaseModel):
    """Base: an unknown key in config.toml is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ExtractionSettings(Settings):
    prompt: str
    prompt_version: str
    max_input_tokens: int = Field(gt=0)
    repair_prompt: str  # one repair attempt for an invalid reply, S1-15
    repair_prompt_version: str


class BudgetSettings(Settings):
    stop_at_usd: float = Field(gt=0)


class ComparisonSettings(Settings):
    """What plain `make run-extraction` runs, S1-17; sampling applies to models that take it."""

    models: list[str] = Field(min_length=1)
    prompt_versions: list[str] = Field(min_length=1)
    sampling: list[str] = Field(min_length=1)


class OpenRouterSettings(Settings):
    base_url: str
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)


class OllamaSettings(Settings):
    base_url: str
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)


class OpenRouterModel(Settings):
    client: Literal["openrouter"]
    slug: str
    reasoning_effort: str | None = None
    max_tokens: int = Field(gt=0)
    # Most cloud models take no sampling (docs/ISSUES.md I-003). For one that does (S1-22),
    # `sampling` names the [sampling] table used when --sampling is not given and
    # `sampling_params` the values of it that are sent: Claude 4+ models reject temperature
    # and top_p in one request.
    sampling: str | None = None
    sampling_params: list[Literal["temperature", "top_p"]] = Field(default_factory=list)
    # USD per 1M tokens and the date they were checked; for metrics only, never sent (S1-11)
    price_in_usd_per_mtok: float = Field(ge=0)
    price_out_usd_per_mtok: float = Field(ge=0)
    price_checked: date

    @model_validator(mode="after")
    def sampling_with_params(self) -> "OpenRouterModel":
        if (self.sampling is None) != (not self.sampling_params):
            raise ValueError("set sampling and sampling_params together, or neither")
        return self


PRICE_FIELDS = {"price_in_usd_per_mtok", "price_out_usd_per_mtok", "price_checked"}


class OllamaModel(Settings):
    client: Literal["ollama"]
    slug: str  # Ollama tag, e.g. gemma4:e4b
    num_ctx: int = Field(gt=0)  # context window; required, the server default is too small
    num_predict: int = Field(gt=0)  # max output tokens
    think: bool = False
    sampling: str  # [sampling] table used when no --sampling is given, S1-10


class SamplingConfig(Settings):
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)


# The `client` field picks the model type, like an enum with associated values.
ModelConfig = Annotated[OpenRouterModel | OllamaModel, Field(discriminator="client")]


def accepts_sampling(model: OpenRouterModel | OllamaModel) -> bool:
    """Ollama models always; a cloud model only if config.toml names its sampling (I-003)."""
    return isinstance(model, OllamaModel) or model.sampling is not None


def sent_sampling(model: OpenRouterModel | OllamaModel, values: SamplingConfig) -> dict:
    """The sampling values sent to the model: all of them to Ollama, `sampling_params` to a
    cloud model."""
    sent = values.model_dump()
    if isinstance(model, OpenRouterModel):
        sent = {name: value for name, value in sent.items() if name in model.sampling_params}
    return sent


class EmbeddingModel(Settings):
    """A FastEmbed model, S2-06. FastEmbed adds no prefixes itself, so the ones the model card
    asks for are set here and added to every query or passage."""

    name: str  # the FastEmbed model name, e.g. intfloat/multilingual-e5-large
    query_prefix: str = ""
    passage_prefix: str = ""


class EmbeddingSettings(Settings):
    model: str  # key in [embedding_models]: the model the index uses, picked in S2-06
    compare: list[str] = Field(min_length=1)  # what `make compare-embeddings` measures
    batch_size: int = Field(gt=0)
    cache_dir: str  # model downloads, relative to the repo root (gitignored)


class RerankerModel(Settings):
    """A FastEmbed cross-encoder, S2-08."""

    name: str  # the FastEmbed model name, e.g. BAAI/bge-reranker-v2-m3


class RerankingSettings(Settings):
    model: str  # key in [reranker_models]: the model search uses, picked in S2-08
    compare: list[str] = Field(min_length=1)  # what `make compare-rerankers` measures
    candidates: int = Field(gt=0)  # first-stage hits the reranker rescores
    batch_size: int = Field(gt=0)


class IndexSettings(Settings):
    path: str  # Qdrant local storage, relative to the repo root (gitignored)
    collections: dict[str, str] = Field(min_length=1)  # collection name: chunks file


SearchMode = Literal["vector", "bm25", "hybrid"]


class SearchSettings(Settings):
    """Retrieval, S2-07. bm25_k1 and bm25_b are baked into the index by `make index`."""

    mode: SearchMode  # what `make search` runs when no MODE is given
    rerank: bool  # whether `make search` reranks when RERANK is not given, S2-08
    prefetch_k: int = Field(gt=0)  # candidates each retriever hands to fusion
    rrf_k: int = Field(gt=0)  # RRF constant: score = sum of 1 / (rrf_k + rank)
    bm25_k1: float = Field(gt=0)  # term-frequency saturation
    bm25_b: float = Field(ge=0, le=1)  # length normalization: 0 none, 1 full


RERANK_SUFFIX = "+rerank"


class RetrievalEvalSettings(Settings):
    """What `make eval-retrieval` measures, S2-10. A setup is a search mode, optionally with
    `+rerank` (the [reranking] model rescores [reranking] candidates first-stage hits)."""

    ks: list[int] = Field(min_length=1)  # recall@k for each k; the largest k is the search depth
    setups: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self) -> "RetrievalEvalSettings":
        if any(k <= 0 for k in self.ks) or self.ks != sorted(set(self.ks)):
            raise ValueError("[retrieval_eval] ks must be positive, distinct and ascending")
        if len(set(self.setups)) != len(self.setups):
            raise ValueError("[retrieval_eval] setups must be distinct")
        for name in self.setups:
            if name.removesuffix(RERANK_SUFFIX) not in ("vector", "bm25", "hybrid"):
                raise ValueError(
                    f"[retrieval_eval] unknown setup {name!r}: vector, bm25 or hybrid, "
                    "with +rerank optional"
                )
        return self

    @property
    def depth(self) -> int:
        return self.ks[-1]


# The rendered context must hold the largest chunk: MAX_CHARS of text plus its `<chunk ...>` tag
# line (467 characters at most in the corpus, 2026-11-21), at 4 characters per token.
CHUNK_TAG_CHARS = 500
MIN_CONTEXT_TOKENS = (MAX_CHARS + CHUNK_TAG_CHARS) // 4


class AnswerSettings(Settings):
    """The answer pipeline, S2-13: retrieve, fit a token budget, answer, check citations."""

    model: str  # key in [models], decision 15
    prompt: str
    prompt_version: str
    repair_prompt: str  # one repair attempt for an answer that fails validation
    repair_prompt_version: str
    collection: str  # key in [index] collections, decision 19
    k: int = Field(gt=0)  # search hits offered to the budget, best first
    max_context_tokens: int = Field(ge=MIN_CONTEXT_TOKENS)  # estimate of the rendered chunks
    stop_at_usd: float = Field(gt=0)  # stage 2 budget guard over results/answers/, decision 18
    # S2-14: refuse without a model call when the best hit's reranker score is below this; the
    # scale is the cross-encoder's raw score, so it needs [search] rerank. None: no gate.
    min_rerank_score: float | None = None


class RefusalEvalSettings(Settings):
    """What `make eval-refusal` measures, S2-14: these versions of the [answer] prompt, each on
    every development question, with the score gate off so the model's own status is seen."""

    prompt_versions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct(self) -> "RefusalEvalSettings":
        if len(set(self.prompt_versions)) != len(self.prompt_versions):
            raise ValueError("[refusal_eval] prompt_versions must be distinct")
        return self


class JudgeSettings(Settings):
    """The LLM judges of `make eval`, S2-18: answer quality against the test set's reference
    answer, and whether each claim is supported by the chunks that were sent. One versioned
    prompt each; the model is a key in [models]."""

    model: str
    quality_prompt: str
    quality_prompt_version: str
    faithfulness_prompt: str
    faithfulness_prompt_version: str


class RunSettings(Settings):
    """`make run`, S2-15: the development questions answered and printed as samples."""

    sample_questions: list[str] = Field(min_length=1)  # IDs in evals/dev_questions.toml

    @model_validator(mode="after")
    def distinct(self) -> "RunSettings":
        if len(set(self.sample_questions)) != len(self.sample_questions):
            raise ValueError("[run] sample_questions must be distinct")
        return self


class Config(Settings):
    extraction: ExtractionSettings
    budget: BudgetSettings
    comparison: ComparisonSettings
    openrouter: OpenRouterSettings
    ollama: OllamaSettings
    models: dict[str, ModelConfig]
    sampling: dict[str, SamplingConfig]
    embedding: EmbeddingSettings
    embedding_models: dict[str, EmbeddingModel]
    index: IndexSettings
    search: SearchSettings
    reranking: RerankingSettings
    reranker_models: dict[str, RerankerModel]
    retrieval_eval: RetrievalEvalSettings
    answer: AnswerSettings
    refusal_eval: RefusalEvalSettings
    judge: JudgeSettings
    run: RunSettings

    @model_validator(mode="after")
    def answer_names_exist(self) -> "Config":
        if self.answer.model not in self.models:
            raise ValueError(
                f"[answer] model {self.answer.model!r}: no [models.{self.answer.model!r}] table"
            )
        if self.answer.collection not in self.index.collections:
            raise ValueError(
                f"[answer] collection {self.answer.collection!r}: not in [index] collections"
            )
        if self.judge.model not in self.models:
            raise ValueError(
                f"[judge] model {self.judge.model!r}: no [models.{self.judge.model!r}] table"
            )
        if self.answer.min_rerank_score is not None and not self.search.rerank:
            raise ValueError(
                "[answer] min_rerank_score is a reranker score: it needs [search] rerank = true"
            )
        return self

    @model_validator(mode="after")
    def embedding_names_exist(self) -> "Config":
        for key in [self.embedding.model, *self.embedding.compare]:
            if key not in self.embedding_models:
                raise ValueError(f"[embedding] model {key!r}: no [embedding_models.{key!r}] table")
        return self

    @model_validator(mode="after")
    def reranker_names_exist(self) -> "Config":
        for key in [self.reranking.model, *self.reranking.compare]:
            if key not in self.reranker_models:
                raise ValueError(f"[reranking] model {key!r}: no [reranker_models.{key!r}] table")
        return self

    @model_validator(mode="after")
    def default_sampling_exists(self) -> "Config":
        for key, model in self.models.items():
            if accepts_sampling(model) and model.sampling not in self.sampling:
                raise ValueError(f"model {key!r}: no [sampling.{model.sampling!r}] table")
        return self

    @model_validator(mode="after")
    def comparison_names_exist(self) -> "Config":
        for key in self.comparison.models:
            if key not in self.models:
                raise ValueError(f"[comparison] model {key!r}: no [models.{key!r}] table")
        for name in self.comparison.sampling:
            if name not in self.sampling:
                raise ValueError(f"[comparison] sampling {name!r}: no [sampling.{name!r}] table")
        return self


def load_config(path: Path = CONFIG) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))


def resolve_sampling(
    config: Config, model_key: str, name: str | None = None
) -> tuple[str, SamplingConfig] | None:
    """The sampling configuration a run uses: `name`, else the model's default.

    None for a model that takes no temperature/top_p (most cloud models, docs/ISSUES.md I-003).
    """
    model = config.models[model_key]
    if not accepts_sampling(model):
        if name is not None:
            raise ValueError(
                f"model {model_key!r} does not accept sampling parameters (docs/ISSUES.md I-003)"
            )
        return None
    name = name or model.sampling
    if name not in config.sampling:
        raise ValueError(f"unknown sampling {name!r}; config.toml has {', '.join(config.sampling)}")
    return name, config.sampling[name]


def run_params(config: Config, model_key: str, sampling: str | None = None) -> dict:
    """The parameters a run sends, as logged in every result record and run.json.

    `sampling` is None for a model that takes none: no temperature/top_p is sent, so the
    provider's default applies (docs/ISSUES.md I-003). Otherwise it holds the table name and
    the values really sent (for Haiku 4.5 only temperature, S1-22).
    """
    model = config.models[model_key]
    params = model.model_dump(
        exclude={"client", "slug", "sampling", "sampling_params", *PRICE_FIELDS}
    )
    resolved = resolve_sampling(config, model_key, sampling)
    params["sampling"] = None
    if resolved:
        name, values = resolved
        params["sampling"] = {"name": name, **sent_sampling(model, values)}
    return params


def params_match(saved: dict | None, current: dict) -> bool:
    """Whether a saved run's params are `current`; runs before S1-16 had no cloud sampling key."""
    return {"sampling": None, **(saved or {})} == current


def load_env(path: Path = ENV) -> None:
    """Read KEY=VALUE lines from .env into os.environ; variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set: copy .env.example to .env and fill it in")
    return value
