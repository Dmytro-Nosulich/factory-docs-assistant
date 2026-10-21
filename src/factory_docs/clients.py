"""Build the client for a model key from config.toml, S1-09.

The only place that knows which client class serves which model; the runner and scripts
call `make_client` and work with any `ChatClient`.
"""

from factory_docs.config import Config, OllamaModel, OpenRouterModel, require_env
from factory_docs.ollama import OllamaClient
from factory_docs.openrouter import OpenRouterClient


def make_client(model_key: str, config: Config) -> OpenRouterClient | OllamaClient:
    model = config.models[model_key]
    if isinstance(model, OpenRouterModel):
        return OpenRouterClient(config.openrouter, model, require_env("OPENROUTER_API_KEY"))
    if isinstance(model, OllamaModel):
        return OllamaClient(config.ollama, model)
    raise ValueError(f"no client for model {model_key!r}")
