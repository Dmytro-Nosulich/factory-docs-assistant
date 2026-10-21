import json

import httpx
import pytest

from factory_docs.config import OpenRouterModel, OpenRouterSettings
from factory_docs.models import ModelCallError
from factory_docs.openrouter import OpenRouterClient

SETTINGS = OpenRouterSettings(
    base_url="https://openrouter.test/api/v1", timeout_s=5, max_retries=3, backoff_s=2.0
)
MODEL = OpenRouterModel(
    client="openrouter", slug="vendor/model", reasoning_effort="none", max_tokens=100
)
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
SCHEMA = {"type": "object"}
REPLY = {
    "id": "gen-1",
    "model": "vendor/model",
    "provider": "Vendor",
    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "cost": 0.000012,
    },
}


def make_client(*steps):
    """Client whose transport plays `steps` in order: an exception to raise or a response."""
    requests, waits = [], []
    queue = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    client = OpenRouterClient(
        SETTINGS, MODEL, "test-key", transport=httpx.MockTransport(handler), sleep=waits.append
    )
    return client, requests, waits


def ok():
    return httpx.Response(200, json=REPLY)


def test_request_has_structured_output_and_routing():
    client, requests, _ = make_client(ok())
    client.complete(MESSAGES, SCHEMA)
    [request] = requests
    assert request.url == "https://openrouter.test/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "vendor/model"
    assert body["messages"] == MESSAGES
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "safety_data_sheet", "strict": True, "schema": SCHEMA},
    }
    assert body["provider"] == {"require_parameters": True}
    assert body["reasoning"] == {"effort": "none"}
    assert body["max_tokens"] == 100


def test_response_fields_are_parsed():
    client, _, _ = make_client(ok())
    r = client.complete(MESSAGES, SCHEMA)
    assert (r.content, r.finish_reason, r.provider, r.generation_id) == (
        "{}",
        "stop",
        "Vendor",
        "gen-1",
    )
    assert (r.prompt_tokens, r.completion_tokens, r.reasoning_tokens) == (10, 4, 0)
    assert r.cost_usd == 0.000012
    assert r.attempts == 1


def test_timeout_is_retried_after_backoff():
    client, requests, waits = make_client(httpx.ReadTimeout("slow"), ok())
    r = client.complete(MESSAGES, SCHEMA)
    assert r.attempts == 2
    assert len(requests) == 2
    assert waits == [2.0]


def test_gives_up_after_max_retries():
    client, requests, waits = make_client(*[httpx.ReadTimeout("slow")] * 4)
    with pytest.raises(ModelCallError, match="gave up after 4 attempts") as e:
        client.complete(MESSAGES, SCHEMA)
    assert e.value.attempts == 4
    assert len(requests) == 4
    assert waits == [2.0, 4.0, 8.0]


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_server_errors_are_retried(status):
    client, _, waits = make_client(httpx.Response(status, text="busy"), ok())
    assert client.complete(MESSAGES, SCHEMA).attempts == 2
    assert waits == [2.0]


@pytest.mark.parametrize("status", [400, 401])
def test_client_errors_are_not_retried(status):
    client, requests, waits = make_client(httpx.Response(status, text="bad"))
    with pytest.raises(ModelCallError, match=f"HTTP {status}") as e:
        client.complete(MESSAGES, SCHEMA)
    assert (e.value.status, e.value.attempts) == (status, 1)
    assert len(requests) == 1
    assert waits == []


def test_error_inside_200_response_is_retried_if_retryable():
    busy = httpx.Response(200, json={"error": {"code": 502, "message": "upstream down"}})
    client, _, waits = make_client(busy, ok())
    assert client.complete(MESSAGES, SCHEMA).attempts == 2
    assert waits == [2.0]
