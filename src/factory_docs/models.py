"""Shared model interface: what every client returns and how calls are retried, S1-09.

A client turns chat messages plus a JSON schema into a `ModelResponse` (the raw reply with
tokens, cost and timing). Prompt rendering and validation live in `extract.py`, so a new
model needs only a new client, never changes in the runner.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ModelCallError(Exception):
    """The call failed for good: a non-retryable error, or retries used up."""

    def __init__(self, message: str, status: int | None, attempts: int):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    finish_reason: str | None
    response_model: str | None
    provider: str | None
    generation_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    latency_s: float  # the successful attempt
    total_s: float  # all attempts, including backoff waits
    attempts: int
    details: dict | None = None  # client-specific extras, e.g. Ollama's timing breakdown
    model_digest: str | None = None  # exact local build behind the tag (Ollama), S1-16


class ChatClient(Protocol):
    """schema_name: what the structured-output schema is called, where the API takes a name
    (OpenRouter); the client's default when not given (S2-13)."""

    def complete(
        self, messages: list[dict[str, str]], schema: dict, schema_name: str = ...
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for (OpenAI format), S3-02.

    arguments is the raw JSON text the model wrote: parsing and validating it is the tool
    registry's job (S3-03), so a bad argument can go back to the model as text.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolTurn:
    """One model call with tools, S3-02.

    response: content, finish reason, tokens, cost and timing, as for any call.
    tool_calls: what the model asked to run; empty when it answered in text.
    message: the assistant message to append to the history unchanged before the tool results.
    """

    response: ModelResponse
    tool_calls: tuple[ToolCall, ...]
    message: dict


class ToolChatClient(Protocol):
    """A client the agent loop can use (S3-04): chat messages plus OpenAI-format tools."""

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice: str | dict = ...
    ) -> ToolTurn: ...


class AttemptFailed(Exception):
    """One attempt failed; `retryable` says whether trying again can help."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def call_with_retry(
    attempt: Callable[[], dict],
    max_retries: int,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict, float, float, int]:
    """Run attempt() until it succeeds; return (data, latency_s, total_s, attempts).

    A retryable failure waits backoff_s, 2 x backoff_s, 4 x backoff_s, ... between tries.
    """
    start = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        attempt_start = time.perf_counter()
        try:
            data = attempt()
        except AttemptFailed as e:
            if not e.retryable:
                raise ModelCallError(str(e), e.status, attempts) from e
            if attempts > max_retries:
                message = f"{e} (gave up after {attempts} attempts)"
                raise ModelCallError(message, e.status, attempts) from e
            sleep(backoff_s * 2 ** (attempts - 1))
            continue
        now = time.perf_counter()
        return data, now - attempt_start, now - start, attempts
