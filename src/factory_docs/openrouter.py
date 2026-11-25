"""OpenRouter chat client with structured output, timeout and retry, S1-07.

HTTP only: it sends chat messages with a JSON schema and returns the raw reply with usage,
cost and timing. Parsing and validating the reply is `extract.py`'s job. Since S3-02 it can
also send OpenAI-format tools and return the model's tool calls (`chat_with_tools`); the
structured-output request bodies of stages 1 and 2 are unchanged. No response-healing
plugin is used, so the share of valid first responses stays an honest measurement.
"""

import time
from collections.abc import Callable

import httpx

from factory_docs.config import OpenRouterModel, OpenRouterSettings
from factory_docs.models import (
    AttemptFailed,
    ModelCallError,
    ModelResponse,
    ToolCall,
    ToolTurn,
    call_with_retry,
)

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
SCHEMA_NAME = "safety_data_sheet"  # stage 1 default, so stage 1 request bodies stay the same


class OpenRouterClient:
    def __init__(
        self,
        settings: OpenRouterSettings,
        model: OpenRouterModel,
        api_key: str,
        sampling: dict[str, float] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """sampling: the values to send (`config.sent_sampling`); None sends none (I-003)."""
        self.settings = settings
        self.model = model
        self.sampling = sampling
        self.sleep = sleep
        self.http = httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self.http.close()

    def request_body(
        self, messages: list[dict[str, str]], schema: dict, schema_name: str = SCHEMA_NAME
    ) -> dict:
        body = {
            "model": self.model.slug,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            # Route only to upstream providers that support every parameter we send.
            "provider": {"require_parameters": True},
            "max_tokens": self.model.max_tokens,
        }
        return self._add_options(body)

    def tool_request_body(
        self, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto"
    ) -> dict:
        """A chat request with OpenAI-format tools and no response_format (S3-02).

        require_parameters makes OpenRouter route only to providers that support tools.
        """
        body = {
            "model": self.model.slug,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "provider": {"require_parameters": True},
            "max_tokens": self.model.max_tokens,
        }
        return self._add_options(body)

    def _add_options(self, body: dict) -> dict:
        """Reasoning effort and sampling, appended last (the order is part of the bytes)."""
        if self.model.reasoning_effort:
            body["reasoning"] = {"effort": self.model.reasoning_effort}
        if self.sampling:
            body.update(self.sampling)
        return body

    def complete(
        self, messages: list[dict[str, str]], schema: dict, schema_name: str = SCHEMA_NAME
    ) -> ModelResponse:
        """One chat completion; retries timeouts, network errors, 408/429/5xx with backoff.

        schema_name is the name OpenRouter passes on with the schema: the stage 1 default, or
        e.g. `rag_answer` for the answer pipeline (S2-13).
        """
        body = self.request_body(messages, schema, schema_name)
        data, latency_s, total_s, attempts = call_with_retry(
            lambda: self._post(body), self.settings.max_retries, self.settings.backoff_s, self.sleep
        )
        return parse_response(data, latency_s, total_s, attempts)

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice: str | dict = "auto"
    ) -> ToolTurn:
        """One chat completion that may ask for tools; retried like `complete`.

        Retrying is safe here: a model call has no side effects (the tools run elsewhere).
        """
        body = self.tool_request_body(messages, tools, tool_choice)
        data, latency_s, total_s, attempts = call_with_retry(
            lambda: self._post(body), self.settings.max_retries, self.settings.backoff_s, self.sleep
        )
        return parse_tool_turn(data, latency_s, total_s, attempts)

    def _post(self, body: dict) -> dict:
        try:
            r = self.http.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise AttemptFailed(f"timeout: {e!r}") from e
        except httpx.TransportError as e:
            raise AttemptFailed(f"network error: {e!r}") from e
        if r.status_code != 200:
            retryable = r.status_code in RETRY_STATUS
            raise AttemptFailed(f"HTTP {r.status_code}: {r.text[:500]}", r.status_code, retryable)
        data = r.json()
        # OpenRouter can also report an error inside a 200 response, at the top or per choice.
        choices = data.get("choices") or [{}]
        error = data.get("error") or choices[0].get("error")
        if error:
            code = error.get("code")
            retryable = code in RETRY_STATUS
            raise AttemptFailed(f"API error {code}: {error.get('message')}", code, retryable)
        if not data.get("choices"):
            raise AttemptFailed(f"no choices in response: {str(data)[:500]}", retryable=False)
        return data


def parse_response(data: dict, latency_s: float, total_s: float, attempts: int) -> ModelResponse:
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return ModelResponse(
        content=(choice.get("message") or {}).get("content"),
        finish_reason=choice.get("finish_reason"),
        response_model=data.get("model"),
        provider=data.get("provider"),
        generation_id=data.get("id"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cost_usd=usage.get("cost"),
        latency_s=round(latency_s, 3),
        total_s=round(total_s, 3),
        attempts=attempts,
    )


def parse_tool_turn(data: dict, latency_s: float, total_s: float, attempts: int) -> ToolTurn:
    """The reply plus its tool calls and the assistant message for the history (S3-02).

    A tool call without an id, a function name or arguments is a format change, so it fails
    loudly (ModelCallError) instead of becoming a silent None.
    """
    response = parse_response(data, latency_s, total_s, attempts)
    raw_calls = (data["choices"][0].get("message") or {}).get("tool_calls") or []
    calls = []
    for raw in raw_calls:
        function = raw.get("function") or {}
        call_id, name, arguments = raw.get("id"), function.get("name"), function.get("arguments")
        if not call_id or not name or arguments is None:
            raise ModelCallError(f"malformed tool call: {str(raw)[:500]}", None, attempts)
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    message: dict = {"role": "assistant", "content": response.content}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    return ToolTurn(response=response, tool_calls=tuple(calls), message=message)
