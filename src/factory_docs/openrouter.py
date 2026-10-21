"""OpenRouter chat client with structured output, timeout and retry, S1-07.

HTTP only: it sends chat messages with a JSON schema and returns the raw reply with usage,
cost and timing. Parsing and validating the reply is `extract.py`'s job. No response-healing
plugin is used, so the share of valid first responses stays an honest measurement.
"""

import time
from collections.abc import Callable

import httpx

from factory_docs.config import OpenRouterModel, OpenRouterSettings
from factory_docs.models import AttemptFailed, ModelResponse, call_with_retry

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class OpenRouterClient:
    def __init__(
        self,
        settings: OpenRouterSettings,
        model: OpenRouterModel,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.model = model
        self.sleep = sleep
        self.http = httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self.http.close()

    def request_body(self, messages: list[dict[str, str]], schema: dict) -> dict:
        body = {
            "model": self.model.slug,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "safety_data_sheet", "strict": True, "schema": schema},
            },
            # Route only to upstream providers that support every parameter we send.
            "provider": {"require_parameters": True},
            "max_tokens": self.model.max_tokens,
        }
        if self.model.reasoning_effort:
            body["reasoning"] = {"effort": self.model.reasoning_effort}
        return body

    def complete(self, messages: list[dict[str, str]], schema: dict) -> ModelResponse:
        """One chat completion; retries timeouts, network errors, 408/429/5xx with backoff."""
        body = self.request_body(messages, schema)
        data, latency_s, total_s, attempts = call_with_retry(
            lambda: self._post(body), self.settings.max_retries, self.settings.backoff_s, self.sleep
        )
        return parse_response(data, latency_s, total_s, attempts)

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
