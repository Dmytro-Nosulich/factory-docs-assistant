from factory_docs.clients import make_client
from factory_docs.config import load_config
from factory_docs.ollama import OllamaClient
from factory_docs.openrouter import OpenRouterClient


def test_each_model_gets_its_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = load_config()
    for key, expected in [("haiku-5.5", OpenRouterClient), ("gemma4-e4b", OllamaClient)]:
        client = make_client(key, config)
        try:
            assert isinstance(client, expected)
        finally:
            client.close()


def test_local_model_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    make_client("gemma4-e4b", load_config()).close()
