"""Run every stage 1 document through one model and save the results, S1-08.

One run is one directory, results/runs/<run_id>/: a `<doc_id>.json` record per document
(the record from `extract_document`) and `run.json`, a summary rebuilt from those files
after every document, so an interrupted run still has an honest summary. Documents run one
at a time, which keeps latency numbers free of concurrency effects.

Budget guard: before each call, the cost of every result under results/ (earlier runs and
smoke files included) is compared with `budget.stop_at_usd`; at or above it, the run stops
and the rest are listed as `not_run`. Failed documents are saved like any other result and
never re-run on resume, so failures stay visible.
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from factory_docs.config import ROOT, Config
from factory_docs.extract import TEXTS, extract_document
from factory_docs.models import ChatClient
from factory_docs.prompts import load_prompt

MANIFEST = ROOT / "data" / "manifest.csv"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
SUMMARY = "run.json"
STATUSES = ("valid", "invalid", "error")


def stage1_doc_ids(manifest: Path = MANIFEST) -> list[str]:
    with manifest.open(encoding="utf-8", newline="") as f:
        return [r["id"] for r in csv.DictReader(f) if "1" in r["stage"].split(",")]


def new_run_id(model_key: str, prompt_version: str) -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{model_key}_{prompt_version}"


def record_cost(record: dict) -> float:
    return (record.get("call") or {}).get("cost_usd") or 0.0


def spent_usd(results_dir: Path = RESULTS) -> float:
    """Total reported cost of every result record under results_dir."""
    total = 0.0
    for path in results_dir.rglob("*.json"):
        if path.name == SUMMARY:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            total += record_cost(record)
    return total


def failed_record(doc_id: str, model_key: str, error: Exception) -> dict:
    """Record for a document that failed outside the model call (e.g. a missing text file)."""
    return {
        "doc_id": doc_id,
        "status": "error",
        "model": {"key": model_key},
        "prompt": None,
        "params": None,
        "input": None,
        "call": None,
        "raw": None,
        "output": None,
        "errors": [f"{type(error).__name__}: {error}"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_all(
    doc_ids: list[str],
    client: ChatClient,
    model_key: str,
    config: Config,
    run_dir: Path,
    spent_before: float,
    texts_dir: Path = TEXTS,
    on_result: Callable[[dict], None] = lambda record: None,
) -> dict:
    """Extract every document that has no result in run_dir yet; return the summary.

    spent_before is the cost already recorded under results/, including this run_dir.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / SUMMARY
    started_at = None
    if summary_path.exists():
        started_at = json.loads(summary_path.read_text(encoding="utf-8"))["started_at"]
    started_at = started_at or datetime.now(UTC).isoformat(timespec="seconds")

    spent = spent_before
    stopped_by_budget = False
    for doc_id in doc_ids:
        path = run_dir / f"{doc_id}.json"
        if path.exists():
            continue
        if spent >= config.budget.stop_at_usd:
            stopped_by_budget = True
            break
        try:
            record = extract_document(doc_id, client, model_key, config, texts_dir)
        except Exception as e:  # one bad document must not stop the run
            record = failed_record(doc_id, model_key, e)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spent += record_cost(record)
        on_result(record)
        write_summary(run_dir, doc_ids, model_key, config, started_at, None, spent, False)

    finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    return write_summary(
        run_dir, doc_ids, model_key, config, started_at, finished_at, spent, stopped_by_budget
    )


def write_summary(
    run_dir: Path,
    doc_ids: list[str],
    model_key: str,
    config: Config,
    started_at: str,
    finished_at: str | None,
    spent_total: float,
    stopped_by_budget: bool,
) -> dict:
    """Rebuild run.json from the result files; finished_at is None while the run is going."""
    model = config.models[model_key]
    settings = config.extraction
    prompt = load_prompt(settings.prompt, settings.prompt_version)
    records = {}
    for doc_id in doc_ids:
        path = run_dir / f"{doc_id}.json"
        if path.exists():
            records[doc_id] = json.loads(path.read_text(encoding="utf-8"))
    calls = [r["call"] for r in records.values() if r["call"] and r["status"] != "error"]

    summary = {
        "run_id": run_dir.name,
        "model": {"key": model_key, "client": model.client, "slug": model.slug},
        "prompt": {"name": prompt.name, "version": prompt.version, "sha256": prompt.sha256},
        "params": model.model_dump(exclude={"client", "slug"}),
        "max_input_tokens": settings.max_input_tokens,
        "docs": doc_ids,
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": {
            **{s: sum(r["status"] == s for r in records.values()) for s in STATUSES},
            "not_run": len(doc_ids) - len(records),
        },
        "cost_usd": round(sum(record_cost(r) for r in records.values()), 6),
        "spent_total_usd": round(spent_total, 6),
        "budget_stop_at_usd": config.budget.stop_at_usd,
        "tokens": {
            "prompt": sum(c["prompt_tokens"] or 0 for c in calls),
            "completion": sum(c["completion_tokens"] or 0 for c in calls),
        },
        "truncated": [d for d, r in records.items() if (r["input"] or {}).get("truncated")],
        "failures": [
            {"doc_id": d, "status": r["status"], "errors": r["errors"]}
            for d, r in records.items()
            if r["status"] != "valid"
        ],
        "not_run": [d for d in doc_ids if d not in records],
        "stopped_by_budget": stopped_by_budget,
    }
    (run_dir / SUMMARY).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
