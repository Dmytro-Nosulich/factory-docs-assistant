"""Answer a question from the index: search hits -> token budget -> model -> checked citations,
S2-13.

The steps:
1. Budget: the search hits are added to the context best first, and filling stops at the first
   hit that would take the rendered context over `[answer] max_context_tokens` (estimated at 4
   characters per token). A lower-ranked chunk never replaces a higher-ranked one, and the
   real prompt tokens the API reports are logged next to the estimate.
2. Answer: the answer prompt with the context and the `Answer` JSON schema, sent through any
   `ChatClient` (decision 15: Haiku 5.5).
3. Checks: the reply must be JSON, pass the `Answer` model (status rules) and pass the
   citation checks, which need the context: each cited chunk ID must be one that was sent (a
   chunk retrieved but dropped by the budget does not count), and each quote must be in its
   chunk, whitespace ignored (the gold-set rule, `retrieval_gold.contains_quote`).
4. Repair (user's choice): a reply that fails gets one repair attempt with the errors listed,
   as in stage 1 (S1-15). If that fails too, the answer is rejected and nothing is shown. Both
   replies are logged, so the share of valid first answers stays visible.

Before the call, a score gate (S2-14, `[answer] min_rerank_score`): when the best hit's
reranker score is below the threshold, the question is refused without a model call.

The record follows the stage 1 record: `status` is the first reply's (`valid`, `invalid`,
`error`, or, with no model call, `no_context` when no chunk was retrieved and
`below_threshold` when the gate refused), and `final_status` is `valid`, `repaired`,
`failed`, `error`, `no_context` or `below_threshold`. It names the model,
prompt version and hash, parameters, retrieval settings, every hit, the chunks sent and
dropped, and the cost. Records go to results/answers/, which has its own budget guard
(decision 18; since S2-18 it also counts the judge calls in results/judgements/).
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from factory_docs.answer_schema import Answer, answer_schema_text, render_context
from factory_docs.chunking import Chunk
from factory_docs.config import Config, run_params
from factory_docs.extract import prompt_info, set_final_status
from factory_docs.golden import format_loc
from factory_docs.index import Hit
from factory_docs.models import ChatClient, ModelCallError
from factory_docs.pdf_text import estimate_tokens
from factory_docs.prompts import load_prompt
from factory_docs.retrieval_gold import contains_quote
from factory_docs.run import RESULTS, record_cost, spent_usd

ANSWERS = RESULTS / "answers"
JUDGEMENTS = RESULTS / "judgements"  # judge calls of `make eval` (S2-18), same guard
SCHEMA_NAME = "rag_answer"  # the name OpenRouter passes on with the JSON schema
SHOWN = ("valid", "repaired")  # final statuses whose answer may be shown
NOT_CALLED = ("no_context", "below_threshold")  # refused before any model call


class BudgetExceeded(Exception):
    """The stage 2 answer logs already cost `[answer] stop_at_usd` or more."""


def context_tokens(chunks: Sequence[Chunk]) -> int:
    """Estimated tokens of the rendered context, as the model receives it."""
    return estimate_tokens(render_context(chunks)) if chunks else 0


def fit_context(hits: Sequence[Hit], max_tokens: int) -> tuple[list[Hit], list[Hit]]:
    """(sent, dropped): the best hits whose rendered context fits max_tokens, and the rest.

    Filling stops at the first hit that does not fit, so every sent hit ranks above every
    dropped one.
    """
    sent: list[Hit] = []
    for position, hit in enumerate(hits):
        if context_tokens([h.chunk for h in [*sent, hit]]) > max_tokens:
            return sent, list(hits[position:])
        sent.append(hit)
    return sent, []


def check_citations(answer: Answer, chunks: Sequence[Chunk]) -> list[str]:
    """One error per citation whose chunk was not sent, or whose quote is not in its chunk."""
    sent = {chunk.chunk_id: chunk for chunk in chunks}
    errors = []
    for i, claim in enumerate(answer.claims):
        for j, citation in enumerate(claim.citations):
            where = f"claims[{i}].citations[{j}]"
            chunk = sent.get(citation.chunk_id)
            if chunk is None:
                errors.append(f"{where}: chunk {citation.chunk_id} is not in the context")
            elif not contains_quote(chunk.text, citation.quote):
                errors.append(
                    f"{where}: the quote is not in chunk {citation.chunk_id}: {citation.quote!r}"
                )
    return errors


def validate_answer(content: str | None, chunks: Sequence[Chunk]) -> tuple[dict | None, list[str]]:
    """Parsed answer and problems; the answer is None unless every check passes."""
    if not content:
        return None, ["empty reply"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}"]
    try:
        answer = Answer.model_validate(data)
    except ValidationError as e:
        return None, [
            f"{format_loc(err['loc']) or '(top level)'}: {err['msg']}" for err in e.errors()
        ]
    errors = check_citations(answer, chunks)
    if errors:
        return None, errors
    return answer.model_dump(mode="json"), []


def call_and_validate(
    client: ChatClient, messages: list[dict[str, str]], chunks: Sequence[Chunk]
) -> dict:
    """One model call (retries included), validated: status, call, raw, output, errors."""
    try:
        response = client.complete(messages, Answer.model_json_schema(), schema_name=SCHEMA_NAME)
    except ModelCallError as e:
        return {
            "status": "error",
            "call": {"attempts": e.attempts, "http_status": e.status},
            "raw": None,
            "output": None,
            "errors": [str(e)],
        }
    call = asdict(response)
    raw = call.pop("content")
    output, errors = validate_answer(raw, chunks)
    return {
        "status": "valid" if output is not None else "invalid",
        "call": call,
        "raw": raw,
        "output": output,
        "errors": errors,
    }


def hit_info(hit: Hit) -> dict:
    return {
        "chunk_id": hit.chunk.chunk_id,
        "score": hit.score,
        "vector_rank": hit.vector_rank,
        "bm25_rank": hit.bm25_rank,
        "first_stage_rank": hit.first_stage_rank,
    }


def pages(chunk: Chunk) -> str | None:
    if chunk.page_start is None:
        return None
    end = chunk.page_end or chunk.page_start
    return str(chunk.page_start) if end == chunk.page_start else f"{chunk.page_start}-{end}"


def source_info(chunk: Chunk) -> dict:
    """Where a sent chunk comes from: what a citation of it points to."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "pages": pages(chunk),
        "version_or_date": chunk.version_or_date,
        "source_url": chunk.source_url,
    }


def below_threshold(hits: Sequence[Hit], threshold: float | None) -> bool:
    """The score gate: the best hit scores below the threshold (None: no gate)."""
    return threshold is not None and bool(hits) and hits[0].score < threshold


def check_budget(spent: float, config: Config) -> None:
    limit = config.answer.stop_at_usd
    if spent >= limit:
        raise BudgetExceeded(
            f"stage 2 answers and judge calls have cost ${spent:.4f}, at or above [answer] "
            f"stop_at_usd ${limit:.2f} (decision 18)"
        )


def answer_question(
    question: str,
    hits: Sequence[Hit],
    client: ChatClient,
    config: Config,
    spent_before: float,
    retrieval: dict | None = None,
    prompt_version: str | None = None,
    gate: bool = True,
) -> dict:
    """The record of one question. hits: search results, best first. retrieval: how they were
    found (collection, mode, reranker, filter), logged as given. spent_before: the cost of the
    stage 2 answer logs so far; at or above the guard, BudgetExceeded is raised before a call.
    prompt_version: instead of `[answer] prompt_version`. gate: False skips the score gate, so
    the eval sees the model's own status for every question (S2-14).
    """
    settings = config.answer
    model = config.models[settings.model]
    prompt = load_prompt(settings.prompt, prompt_version or settings.prompt_version)
    threshold = settings.min_rerank_score if gate else None
    sent, dropped = fit_context(hits, settings.max_context_tokens)
    chunks = [hit.chunk for hit in sent]
    record = {
        "question": question,
        "status": None,
        "model": {"key": settings.model, "client": model.client, "slug": model.slug},
        "prompt": prompt_info(prompt),
        "params": run_params(config, settings.model),
        "retrieval": {
            **(retrieval or {}),
            "k": settings.k,
            "hits": [hit_info(hit) for hit in hits],
            "top_score": hits[0].score if hits else None,
            "min_rerank_score": threshold,
            "max_context_tokens": settings.max_context_tokens,
            "est_context_tokens": context_tokens(chunks),
            "sent": [source_info(chunk) for chunk in chunks],
            "dropped": [hit.chunk.chunk_id for hit in dropped],
        },
        "call": None,
        "raw": None,
        "output": None,
        "errors": [],
        "repair": None,
        "final_status": None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not chunks:
        record["status"] = "no_context"
        record["errors"] = ["no chunk was retrieved, so no model was called"]
        set_final_status(record)
        return record
    if below_threshold(hits, threshold):
        record["status"] = "below_threshold"
        record["errors"] = [
            f"the best hit scores {hits[0].score:.3f}, below [answer] min_rerank_score "
            f"{threshold}, so no model was called"
        ]
        set_final_status(record)
        return record

    check_budget(spent_before, config)
    messages = prompt.render(
        question=question, context=render_context(chunks), schema=answer_schema_text()
    )
    first = call_and_validate(client, messages, chunks)
    if first["status"] != "error":
        record["model"]["response_model"] = first["call"].pop("response_model")
        record["model"]["provider"] = first["call"].pop("provider")
        record["model"]["digest"] = first["call"].pop("model_digest")
    record.update(first)
    if record["status"] == "invalid":
        record["repair"] = repair_answer(
            messages, record, chunks, client, config, spent_before + record_cost(record)
        )
    set_final_status(record)
    return record


def repair_answer(
    messages: list[dict[str, str]],
    record: dict,
    chunks: Sequence[Chunk],
    client: ChatClient,
    config: Config,
    spent: float,
) -> dict:
    """The one repair attempt for an invalid answer; the `repair` part of a record."""
    settings = config.answer
    prompt = load_prompt(settings.repair_prompt, settings.repair_prompt_version)
    try:
        check_budget(spent, config)
    except BudgetExceeded as e:
        return {"prompt": prompt_info(prompt), "status": "not_run", "errors": [str(e)]}
    conversation = [
        *messages,
        {"role": "assistant", "content": record["raw"] or ""},
        *prompt.render(errors="\n".join(f"- {error}" for error in record["errors"])),
    ]
    return {
        "prompt": prompt_info(prompt),
        **call_and_validate(client, conversation, chunks),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def shown_answer(record: dict) -> dict | None:
    """The answer to show: the repaired one if there was a repair, None if it was rejected."""
    if record["final_status"] == "repaired":
        return record["repair"]["output"]
    if record["final_status"] == "valid":
        return record["output"]
    return None


def spent_on_answers(*dirs: Path) -> float:
    """Reported cost of every stage 2 log: the stage 2 budget guard total (decision 18).

    dirs: the log folders; by default the answer logs and, since S2-18, the judge calls.
    """
    return sum(
        spent_usd(folder, exclude=()) for folder in dirs or (ANSWERS, JUDGEMENTS) if folder.exists()
    )


def save_record(record: dict, answers_dir: Path = ANSWERS) -> Path:
    """Write one record as results/answers/<UTC time>.json; never overwrites a file."""
    answers_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    path = answers_dir / f"{stamp}.json"
    n = 1
    while path.exists():
        n += 1
        path = answers_dir / f"{stamp}_{n}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
