"""Run configuration (config.toml) and secrets (.env), S1-07.

config.toml holds everything that changes a run: prompt version, model slugs, timeouts and
retries. Keys are never in it; they come from the environment or from .env.
"""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class OpenRouterSettings(Settings):
    base_url: str
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)


class ModelConfig(Settings):
    client: Literal["openrouter"]
    slug: str
    reasoning_effort: str | None = None
    max_tokens: int = Field(gt=0)


class Config(Settings):
    extraction: ExtractionSettings
    openrouter: OpenRouterSettings
    models: dict[str, ModelConfig]


def load_config(path: Path = CONFIG) -> Config:
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))


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
