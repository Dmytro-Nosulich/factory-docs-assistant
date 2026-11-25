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
    client="openrouter",
    slug="vendor/model",
    reasoning_effort="none",
    max_tokens=100,
    price_in_usd_per_mtok=1.0,
    price_out_usd_per_mtok=2.0,
    price_checked="2026-01-01",
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


def make_client(*steps, sampling=None):
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
        SETTINGS,
        MODEL,
        "test-key",
        sampling=sampling,
        transport=httpx.MockTransport(handler),
        sleep=waits.append,
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
    assert "temperature" not in body and "top_p" not in body  # none given: none sent (I-003)


def test_schema_name_is_passed_through():
    client, requests, _ = make_client(ok())
    client.complete(MESSAGES, SCHEMA, schema_name="rag_answer")
    body = json.loads(requests[0].content)
    assert body["response_format"]["json_schema"]["name"] == "rag_answer"


def test_sampling_values_are_sent_at_the_top_level():
    client, requests, _ = make_client(ok(), sampling={"temperature": 0.0})
    client.complete(MESSAGES, SCHEMA)
    body = json.loads(requests[0].content)
    assert body["temperature"] == 0.0
    assert "top_p" not in body


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


# Exact bytes the stage 1 and stage 2 calls sent before S3-02 (captured from the S3-01 code).
# The tool-calling client must not change a single byte of them.
STAGE1_BODY = (
    b'{"model":"vendor/model","messages":[{"role":"system","content":"s"},'
    b'{"role":"user","content":"u"}],"response_format":{"type":"json_schema","json_schema":'
    b'{"name":"safety_data_sheet","strict":true,"schema":{"type":"object"}}},'
    b'"provider":{"require_parameters":true},"max_tokens":100,"reasoning":{"effort":"none"}}'
)
STAGE2_BODY = (
    b'{"model":"vendor/model","messages":[{"role":"system","content":"s"},'
    b'{"role":"user","content":"u"}],"response_format":{"type":"json_schema","json_schema":'
    b'{"name":"rag_answer","strict":true,"schema":{"type":"object"}}},'
    b'"provider":{"require_parameters":true},"max_tokens":100,"reasoning":{"effort":"none"},'
    b'"temperature":0.0}'
)


def test_stage_1_request_bytes_are_unchanged():
    client, requests, _ = make_client(ok())
    client.complete(MESSAGES, SCHEMA)
    assert requests[0].content == STAGE1_BODY


def test_stage_2_request_bytes_are_unchanged():
    client, requests, _ = make_client(ok(), sampling={"temperature": 0.0})
    client.complete(MESSAGES, SCHEMA, schema_name="rag_answer")
    assert requests[0].content == STAGE2_BODY


# Tool calling (S3-02)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a product.",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
            },
        },
    }
]


def tool_reply(*calls, content=None, finish_reason="tool_calls"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return httpx.Response(
        200, json={**REPLY, "choices": [{"finish_reason": finish_reason, "message": message}]}
    )


def raw_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_tool_request_has_tools_and_no_response_format():
    client, requests, _ = make_client(tool_reply(raw_call("c1", "lookup", "{}")))
    client.chat_with_tools(MESSAGES, TOOLS)
    body = json.loads(requests[0].content)
    assert body["tools"] == TOOLS
    assert body["tool_choice"] == "auto"
    assert "response_format" not in body
    assert body["provider"] == {"require_parameters": True}
    assert body["reasoning"] == {"effort": "none"}
    assert body["max_tokens"] == 100
    assert body["messages"] == MESSAGES


def test_tool_request_passes_tool_choice_and_sampling():
    forced = {"type": "function", "function": {"name": "lookup"}}
    client, requests, _ = make_client(
        tool_reply(raw_call("c1", "lookup", "{}")), sampling={"temperature": 0.0}
    )
    client.chat_with_tools(MESSAGES, TOOLS, tool_choice=forced)
    body = json.loads(requests[0].content)
    assert body["tool_choice"] == forced
    assert body["temperature"] == 0.0


def test_one_tool_call_is_parsed_with_usage_and_cost():
    client, _, _ = make_client(tool_reply(raw_call("c1", "lookup", '{"product": "RM 110"}')))
    turn = client.chat_with_tools(MESSAGES, TOOLS)
    [call] = turn.tool_calls
    assert (call.id, call.name, call.arguments) == ("c1", "lookup", '{"product": "RM 110"}')
    r = turn.response
    assert r.finish_reason == "tool_calls"
    assert (r.prompt_tokens, r.completion_tokens, r.cost_usd) == (10, 4, 0.000012)


def test_parallel_tool_calls_keep_their_order():
    client, _, _ = make_client(
        tool_reply(raw_call("c1", "lookup", '{"product": "a"}'), raw_call("c2", "lookup", "{}"))
    )
    turn = client.chat_with_tools(MESSAGES, TOOLS)
    assert [c.id for c in turn.tool_calls] == ["c1", "c2"]


def test_text_answer_has_no_tool_calls():
    client, _, _ = make_client(tool_reply(content="Done.", finish_reason="stop"))
    turn = client.chat_with_tools(MESSAGES, TOOLS)
    assert turn.tool_calls == ()
    assert turn.response.content == "Done."
    assert turn.message == {"role": "assistant", "content": "Done."}


def test_assistant_message_and_tool_result_go_back_unchanged():
    client, requests, _ = make_client(
        tool_reply(raw_call("c1", "lookup", '{"product": "a"}'), content="Checking."),
        tool_reply(content="It is 12 °C.", finish_reason="stop"),
    )
    turn = client.chat_with_tools(MESSAGES, TOOLS)
    assert turn.message == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [raw_call("c1", "lookup", '{"product": "a"}')],
    }
    result = {"role": "tool", "tool_call_id": "c1", "content": "flash point 12 °C"}
    history = [*MESSAGES, turn.message, result]
    client.chat_with_tools(history, TOOLS)
    assert json.loads(requests[1].content)["messages"] == history


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "c1", "type": "function", "function": {"arguments": "{}"}},  # no name
        {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},  # no id
        {"id": "c1", "type": "function", "function": {"name": "lookup"}},  # no arguments
    ],
)
def test_malformed_tool_call_fails_loudly(bad):
    client, requests, _ = make_client(tool_reply(bad))
    with pytest.raises(ModelCallError, match="malformed tool call"):
        client.chat_with_tools(MESSAGES, TOOLS)
    assert len(requests) == 1  # not retried: the same reply would come back


def test_tool_call_is_retried_on_503():
    client, _, waits = make_client(
        httpx.Response(503, text="busy"), tool_reply(raw_call("c1", "lookup", "{}"))
    )
    assert client.chat_with_tools(MESSAGES, TOOLS).response.attempts == 2
    assert waits == [2.0]
