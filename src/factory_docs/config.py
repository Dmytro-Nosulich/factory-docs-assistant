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
    """The matrix plain `make run` runs, S1-17; sampling applies to models that accept it."""

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


class Config(Settings):
    extraction: ExtractionSettings
    budget: BudgetSettings
    comparison: ComparisonSettings
    openrouter: OpenRouterSettings
    ollama: OllamaSettings
    models: dict[str, ModelConfig]
    sampling: dict[str, SamplingConfig]

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
