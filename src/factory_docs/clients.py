"""Build the client for a model key from config.toml, S1-09.

The only place that knows which client class serves which model; the runner and scripts
call `make_client` and work with any `ChatClient`.
"""

from factory_docs.config import (
    Config,
    OllamaModel,
    OpenRouterModel,
    require_env,
    resolve_sampling,
    sent_sampling,
)
from factory_docs.ollama import OllamaClient
from factory_docs.openrouter import OpenRouterClient


def make_client(
    model_key: str, config: Config, sampling: str | None = None
) -> OpenRouterClient | OllamaClient:
    """sampling: a [sampling] table name; None means the model's default (if it takes any)."""
    model = config.models[model_key]
    resolved = resolve_sampling(config, model_key, sampling)
    if isinstance(model, OpenRouterModel):
        return OpenRouterClient(
            config.openrouter,
            model,
            require_env("OPENROUTER_API_KEY"),
            sampling=sent_sampling(model, resolved[1]) if resolved else None,
        )
    if isinstance(model, OllamaModel):
        return OllamaClient(config.ollama, model, resolved[1])
    raise ValueError(f"no client for model {model_key!r}")
