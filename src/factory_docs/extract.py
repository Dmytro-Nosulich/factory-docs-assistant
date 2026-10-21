"""Extract one document: text -> prompt -> model -> validated SafetyDataSheet, S1-07.

The result record says what was sent (prompt version, parameters, pages), what came back
(raw reply, tokens, cost, timing) and whether it is schema-valid. Status is one of:
`valid`; `invalid` (reply is not JSON or not schema-valid); `error` (the call itself failed).
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from factory_docs.config import ROOT, Config
from factory_docs.golden import format_loc
from factory_docs.models import ChatClient, ModelCallError
from factory_docs.pdf_text import cap_pages, estimate_tokens
from factory_docs.prompts import load_prompt
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


def extract_document(
    doc_id: str, client: ChatClient, model_key: str, config: Config, texts_dir: Path = TEXTS
) -> dict:
    model = config.models[model_key]
    settings = config.extraction
    prompt = load_prompt(settings.prompt, settings.prompt_version)
    capped = cap_pages(
        (texts_dir / f"{doc_id}.txt").read_text(encoding="utf-8"), settings.max_input_tokens
    )
    messages = prompt.render(document=capped.text, schema=schema_text())

    record = {
        "doc_id": doc_id,
        "status": None,
        "model": {"key": model_key, "client": model.client, "slug": model.slug},
        "prompt": {"name": prompt.name, "version": prompt.version, "sha256": prompt.sha256},
        "params": model.model_dump(exclude={"client", "slug"}),
        "input": {
            "pages_sent": capped.pages_sent,
            "pages_total": capped.pages_total,
            "truncated": capped.truncated,
            "est_tokens": estimate_tokens(capped.text),
        },
        "call": None,
        "raw": None,
        "output": None,
        "errors": [],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        response = client.complete(messages, SafetyDataSheet.model_json_schema())
    except ModelCallError as e:
        record["status"] = "error"
        record["call"] = {"attempts": e.attempts, "http_status": e.status}
        record["errors"] = [str(e)]
        return record

    call = asdict(response)
    record["raw"] = call.pop("content")
    record["model"]["response_model"] = call.pop("response_model")
    record["model"]["provider"] = call.pop("provider")
    record["call"] = call
    record["output"], record["errors"] = validate_reply(record["raw"])
    record["status"] = "valid" if record["output"] is not None else "invalid"
    return record
