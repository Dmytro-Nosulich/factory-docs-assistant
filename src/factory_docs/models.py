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


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]], schema: dict) -> ModelResponse: ...


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
