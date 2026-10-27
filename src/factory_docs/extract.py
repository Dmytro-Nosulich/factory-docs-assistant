"""Extract one document: text -> prompt -> model -> validated SafetyDataSheet, S1-07.

The result record says what was sent (prompt version, parameters, pages), what came back
(raw reply, tokens, cost, timing) and whether it is schema-valid. It also records which
configuration produced it (S1-16): the model (with the Ollama digest of a local build), the
prompt version and file hash, every parameter sent (`sampling` is None for a cloud model) and
the input cap. `status` is the first
response's: `valid`; `invalid` (reply is not JSON or not schema-valid); `error` (the call
itself failed).

Repair (S1-15): an `invalid` reply gets one repair attempt. The model sees the same
conversation plus its own reply and a repair prompt listing the validation errors. The attempt
is stored under `repair`, next to the first response, which stays as it was, so first-response
metrics are not hidden. `final_status` is `valid` (first reply valid), `repaired`, `failed`
(the repair reply was invalid too, or its call failed) or `error` (the first call failed: there
is no reply to repair, and retries already ran).
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from factory_docs.config import ROOT, Config, run_params
from factory_docs.golden import format_loc
from factory_docs.models import ChatClient, ModelCallError
from factory_docs.pdf_text import CappedText, cap_pages, estimate_tokens
from factory_docs.prompts import Prompt, load_prompt
from factory_docs.schema import SafetyDataSheet, schema_text

TEXTS = ROOT / "data" / "texts"


def validate_reply(content: str | None) -> tuple[dict | None, list[str]]:
    """Parsed output and problems; output is None unless the reply is schema-valid."""
    if not content:
        return None, ["empty reply"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}"]
    try:
        sds = SafetyDataSheet.model_validate(data)
    except ValidationError as e:
        return None, [
            f"{format_loc(err['loc']) or '(top level)'}: {err['msg']}" for err in e.errors()
        ]
    return sds.model_dump(mode="json"), []


def prompt_info(prompt: Prompt) -> dict:
    return {"name": prompt.name, "version": prompt.version, "sha256": prompt.sha256}


def build_messages(
    doc_id: str, prompt: Prompt, config: Config, texts_dir: Path = TEXTS
) -> tuple[list[dict[str, str]], CappedText]:
    """The extraction conversation for one document, and the capped text it contains."""
    capped = cap_pages(
        (texts_dir / f"{doc_id}.txt").read_text(encoding="utf-8"),
        config.extraction.max_input_tokens,
    )
    return prompt.render(document=capped.text, schema=schema_text()), capped


def call_and_validate(client: ChatClient, messages: list[dict[str, str]]) -> dict:
    """One model call (retries included), validated: status, call, raw, output, errors."""
    try:
        response = client.complete(messages, SafetyDataSheet.model_json_schema())
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
    output, errors = validate_reply(raw)
    return {
        "status": "valid" if output is not None else "invalid",
        "call": call,
        "raw": raw,
        "output": output,
        "errors": errors,
    }


def repair_reply(
    messages: list[dict[str, str]],
    raw: str | None,
    errors: list[str],
    client: ChatClient,
    config: Config,
) -> dict:
    """The one repair attempt for an invalid reply to `messages`; the `repair` part of a record."""
    settings = config.extraction
    prompt = load_prompt(settings.repair_prompt, settings.repair_prompt_version)
    conversation = [
        *messages,
        {"role": "assistant", "content": raw or ""},
        *prompt.render(errors="\n".join(f"- {error}" for error in errors)),
    ]
    return {
        "prompt": prompt_info(prompt),
        **call_and_validate(client, conversation),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def set_final_status(record: dict) -> None:
    """final_status from the first response and, for an invalid one, its repair attempt."""
    if record["status"] != "invalid":
        record["final_status"] = record["status"]
    elif (record.get("repair") or {}).get("status") == "valid":
        record["final_status"] = "repaired"
    else:
        record["final_status"] = "failed"


def extract_document(
    doc_id: str,
    client: ChatClient,
    model_key: str,
    config: Config,
    texts_dir: Path = TEXTS,
    sampling: str | None = None,
) -> dict:
    """sampling names the [sampling] table the client was built with (None: model default)."""
    model = config.models[model_key]
    settings = config.extraction
    prompt = load_prompt(settings.prompt, settings.prompt_version)
    messages, capped = build_messages(doc_id, prompt, config, texts_dir)

    record = {
        "doc_id": doc_id,
        "status": None,
        "model": {"key": model_key, "client": model.client, "slug": model.slug},
        "prompt": prompt_info(prompt),
        "params": run_params(config, model_key, sampling),
        "input": {
            "pages_sent": capped.pages_sent,
            "pages_total": capped.pages_total,
            "truncated": capped.truncated,
            "est_tokens": estimate_tokens(capped.text),
            "max_input_tokens": settings.max_input_tokens,
        },
        "call": None,
        "raw": None,
        "output": None,
        "errors": [],
        "repair": None,
        "final_status": None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    first = call_and_validate(client, messages)
    if first["status"] != "error":
        record["model"]["response_model"] = first["call"].pop("response_model")
        record["model"]["provider"] = first["call"].pop("provider")
        record["model"]["digest"] = first["call"].pop("model_digest")
    record.update(first)
    if record["status"] == "invalid":
        record["repair"] = repair_reply(messages, record["raw"], record["errors"], client, config)
    set_final_status(record)
    return record
