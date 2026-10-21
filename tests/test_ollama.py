import json

import httpx
import pytest

from factory_docs.config import OllamaModel, OllamaSettings
from factory_docs.models import ModelCallError
from factory_docs.ollama import OllamaClient

SETTINGS = OllamaSettings(base_url="http://ollama.test", timeout_s=5, max_retries=1, backoff_s=5.0)
MODEL = OllamaModel(client="ollama", slug="vendor:small", num_ctx=1000, num_predict=100)
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
SCHEMA = {"type": "object"}


def reply(prompt_tokens=500, cached=0, **extra):
    return {
        "model": "vendor:small",
        "message": {"role": "assistant", "content": "{}"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_cached_count": cached,
        "eval_count": 20,
        "load_duration": 2_000_000_000,
        "prompt_eval_duration": 1_500_000_000,
        "eval_duration": 500_000_000,
        **extra,
    }


def make_client(*steps, model=MODEL):
    """Client whose transport plays `steps` in order: an exception to raise or a response."""
    requests, waits = [], []
    queue = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    client = OllamaClient(
        SETTINGS, model, transport=httpx.MockTransport(handler), sleep=waits.append
    )
    return client, requests, waits


def ok(**kwargs):
    return httpx.Response(200, json=reply(**kwargs))


def test_request_has_schema_context_and_no_thinking():
    client, requests, _ = make_client(ok())
    client.complete(MESSAGES, SCHEMA)
    [request] = requests
    assert request.url == "http://ollama.test/api/chat"
    body = json.loads(request.content)
    assert body == {
        "model": "vendor:small",
        "messages": MESSAGES,
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"num_ctx": 1000, "num_predict": 100},
    }


def test_response_fields_are_parsed():
    client, _, _ = make_client(ok(cached=480))
    response = client.complete(MESSAGES, SCHEMA)
    assert response.content == "{}"
    assert response.finish_reason == "stop"
    assert response.provider == "ollama"
    assert (response.prompt_tokens, response.completion_tokens) == (500, 20)
    assert response.cost_usd == 0.0
    assert response.details == {
        "prompt_cached_tokens": 480,
        "load_s": 2.0,
        "prompt_eval_s": 1.5,
        "eval_s": 0.5,
        "thinking_chars": 0,
    }


def test_prompt_that_fills_the_context_is_an_error():
    client, requests, _ = make_client(ok(prompt_tokens=901))  # 901 + 100 > 1000
    with pytest.raises(ModelCallError, match="did not fit in num_ctx=1000") as e:
        client.complete(MESSAGES, SCHEMA)
    assert len(requests) == 1  # not retried
    assert e.value.attempts == 1


def test_prompt_that_just_fits_is_accepted():
    client, _, _ = make_client(ok(prompt_tokens=900))  # 900 + 100 == 1000
    assert client.complete(MESSAGES, SCHEMA).prompt_tokens == 900


def test_missing_prompt_count_is_an_error():
    client, _, _ = make_client(httpx.Response(200, json=reply(prompt_tokens=None)))
    with pytest.raises(ModelCallError, match="cannot check context"):
        client.complete(MESSAGES, SCHEMA)


def test_unreachable_server_is_retried_then_explained():
    refused = httpx.ConnectError("connection refused")
    client, requests, waits = make_client(refused, refused)
    with pytest.raises(ModelCallError, match="is `ollama serve` running") as e:
        client.complete(MESSAGES, SCHEMA)
    assert len(requests) == 2
    assert waits == [5.0]
    assert e.value.attempts == 2


def test_timeout_is_retried():
    client, _, waits = make_client(httpx.ReadTimeout("slow"), ok())
    assert client.complete(MESSAGES, SCHEMA).attempts == 2
    assert waits == [5.0]


def test_missing_model_is_not_retried():
    missing = httpx.Response(404, json={"error": "model 'vendor:small' not found"})
    client, requests, _ = make_client(missing)
    with pytest.raises(ModelCallError, match="ollama pull vendor:small"):
        client.complete(MESSAGES, SCHEMA)
    assert len(requests) == 1
