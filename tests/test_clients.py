import pytest

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


def test_local_model_gets_the_chosen_sampling():
    client = make_client("qwen3.5-9b", load_config(), "greedy")
    try:
        assert (client.sampling.temperature, client.sampling.top_p) == (0.0, 1.0)
    finally:
        client.close()


def test_haiku_4_5_gets_only_the_temperature_of_its_sampling(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = make_client("haiku-4.5", load_config(), "greedy")
    try:
        assert client.sampling == {"temperature": 0.0}
    finally:
        client.close()
    client = make_client("haiku-5.5", load_config())
    try:
        assert client.sampling is None
    finally:
        client.close()


def test_cloud_model_with_sampling_is_refused(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="I-003"):
        make_client("haiku-5.5", load_config(), "greedy")


def test_local_model_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    make_client("gemma4-e4b", load_config()).close()
