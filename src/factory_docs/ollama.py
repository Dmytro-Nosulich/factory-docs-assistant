"""Ollama chat client (local models) with structured output, timeout and retry, S1-09.

Same contract as `openrouter.py`: chat messages plus a JSON schema in, the raw reply with
tokens and timing out. Thinking is switched off and the context window (`num_ctx`) is set
explicitly. Some Ollama runners cut a too-long prompt silently, so a prompt that does not fit
`num_ctx` together with `num_predict` is a hard error, never a reply (docs/ISSUES.md I-004).
"""

import time
from collections.abc import Callable

import httpx

from factory_docs.config import OllamaModel, OllamaSettings
from factory_docs.models import AttemptFailed, ModelCallError, ModelResponse, call_with_retry


class OllamaClient:
    def __init__(
        self,
        settings: OllamaSettings,
        model: OllamaModel,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.model = model
        self.sleep = sleep
        self.http = httpx.Client(
            base_url=settings.base_url, timeout=settings.timeout_s, transport=transport
        )

    def close(self) -> None:
        self.http.close()

    def request_body(self, messages: list[dict[str, str]], schema: dict) -> dict:
        options = {"num_ctx": self.model.num_ctx, "num_predict": self.model.num_predict}
        return {
            "model": self.model.slug,
            "messages": messages,
            "format": schema,  # Ollama constrains the output to this JSON schema
            "stream": False,
            "think": self.model.think,
            "options": options,
        }

    def complete(self, messages: list[dict[str, str]], schema: dict) -> ModelResponse:
        """One chat completion; retries timeouts and network errors with backoff."""
        body = self.request_body(messages, schema)
        data, latency_s, total_s, attempts = call_with_retry(
            lambda: self._post(body), self.settings.max_retries, self.settings.backoff_s, self.sleep
        )
        check_context(data, self.model, attempts)
        return parse_response(data, latency_s, total_s, attempts)

    def _post(self, body: dict) -> dict:
        try:
            r = self.http.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise AttemptFailed(f"timeout: {e!r}") from e
        except httpx.ConnectError as e:
            message = (
                f"cannot reach Ollama at {self.settings.base_url} (is `ollama serve` running?)"
            )
            raise AttemptFailed(f"{message}: {e!r}") from e
        except httpx.TransportError as e:
            raise AttemptFailed(f"network error: {e!r}") from e
        if r.status_code == 404:
            message = f"HTTP 404: {r.text[:300]} (run `ollama pull {self.model.slug}`)"
            raise AttemptFailed(message, 404, retryable=False)
        if r.status_code != 200:
            retryable = r.status_code >= 500
            raise AttemptFailed(f"HTTP {r.status_code}: {r.text[:500]}", r.status_code, retryable)
        data = r.json()
        if data.get("error"):
            raise AttemptFailed(f"Ollama error: {data['error']}", retryable=False)
        return data


def check_context(data: dict, model: OllamaModel, attempts: int) -> None:
    """Fail unless the whole prompt plus the output budget fit in num_ctx.

    prompt_eval_count is the full prompt, cached tokens included (measured, I-004).
    """
    prompt_tokens = data.get("prompt_eval_count")
    if prompt_tokens is None:
        raise ModelCallError("no prompt_eval_count in reply: cannot check context", None, attempts)
    if prompt_tokens + model.num_predict > model.num_ctx:
        message = (
            f"prompt did not fit in num_ctx={model.num_ctx}: {prompt_tokens} prompt tokens "
            f"+ num_predict {model.num_predict} (may be cut silently, docs/ISSUES.md I-004)"
        )
        raise ModelCallError(message, None, attempts)


def seconds(ns: int | None) -> float | None:
    return None if ns is None else round(ns / 1e9, 3)


def parse_response(data: dict, latency_s: float, total_s: float, attempts: int) -> ModelResponse:
    message = data.get("message") or {}
    return ModelResponse(
        content=message.get("content"),
        finish_reason=data.get("done_reason"),
        response_model=data.get("model"),
        provider="ollama",
        generation_id=None,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
        reasoning_tokens=None,
        cost_usd=0.0,  # no API cost; hardware and time are reported separately
        latency_s=round(latency_s, 3),
        total_s=round(total_s, 3),
        attempts=attempts,
        details={
            "prompt_cached_tokens": data.get("prompt_eval_cached_count"),
            "load_s": seconds(data.get("load_duration")),
            "prompt_eval_s": seconds(data.get("prompt_eval_duration")),
            "eval_s": seconds(data.get("eval_duration")),
            "thinking_chars": len(message.get("thinking") or ""),  # 0 when thinking is off
        },
    )
