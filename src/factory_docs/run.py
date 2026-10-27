"""Run every stage 1 document through one model and save the results, S1-08.

One run is one directory, results/runs/<run_id>/: a `<doc_id>.json` record per document
(the record from `extract_document`) and `run.json`, a summary rebuilt from those files
after every document, so an interrupted run still has an honest summary. Documents run one
at a time, which keeps latency numbers free of concurrency effects.

Budget guard: before each document (its first call and any repair call), the cost of every
result under results/ (earlier runs and smoke files included) is compared with
`budget.stop_at_usd`; at or above it, the run stops and the rest are listed as `not_run`.
Failed documents are saved like any other result and never re-run on resume, so failures
stay visible.

Repair (S1-15): `extract_document` gives an invalid reply one repair attempt. `repair_run`
does the same for an existing run whose records predate the repair step: it repairs each
invalid record in place and sets `final_status` on every record, without re-running the
first calls.
"""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from factory_docs.config import ROOT, Config, run_params
from factory_docs.extract import (
    TEXTS,
    build_messages,
    extract_document,
    prompt_info,
    repair_reply,
    set_final_status,
)
from factory_docs.models import ChatClient
from factory_docs.prompts import load_prompt

MANIFEST = ROOT / "data" / "manifest.csv"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
SUMMARY = "run.json"
STATUSES = ("valid", "invalid", "error")
FINAL_STATUSES = ("valid", "repaired", "failed", "error")


def stage1_doc_ids(manifest: Path = MANIFEST) -> list[str]:
    with manifest.open(encoding="utf-8", newline="") as f:
        return [r["id"] for r in csv.DictReader(f) if "1" in r["stage"].split(",")]


def new_run_id(model_key: str, prompt_version: str, sampling: str | None = None) -> str:
    parts = [f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}", model_key, sampling, prompt_version]
    return "_".join(p for p in parts if p)


def record_cost(record: dict) -> float:
    """Reported cost of the first call plus the repair call, if any."""
    calls = [record.get("call"), (record.get("repair") or {}).get("call")]
    return sum((call or {}).get("cost_usd") or 0.0 for call in calls)


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
        "repair": None,
        "final_status": "error",
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
    sampling: str | None = None,
) -> dict:
    """Extract every document that has no result in run_dir yet; return the summary.

    spent_before is the cost already recorded under results/, including this run_dir.
    sampling names the [sampling] table the client was built with (None: model default).
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
            record = extract_document(doc_id, client, model_key, config, texts_dir, sampling)
        except Exception as e:  # one bad document must not stop the run
            record = failed_record(doc_id, model_key, e)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        spent += record_cost(record)
        on_result(record)
        write_summary(run_dir, doc_ids, model_key, config, started_at, None, spent, False, sampling)

    finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    return write_summary(
        run_dir,
        doc_ids,
        model_key,
        config,
        started_at,
        finished_at,
        spent,
        stopped_by_budget,
        sampling,
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
    sampling: str | None = None,
) -> dict:
    """Rebuild run.json from the result files; finished_at is None while the run is going."""
    model = config.models[model_key]
    settings = config.extraction
    prompt = load_prompt(settings.prompt, settings.prompt_version)
    repair_prompt = load_prompt(settings.repair_prompt, settings.repair_prompt_version)
    records = {}
    for doc_id in doc_ids:
        path = run_dir / f"{doc_id}.json"
        if path.exists():
            records[doc_id] = json.loads(path.read_text(encoding="utf-8"))
    calls = [r["call"] for r in records.values() if r["call"] and r["status"] != "error"]
    repairs = [r["repair"] for r in records.values() if r.get("repair")]
    calls += [p["call"] for p in repairs if p["status"] != "error"]  # paid tokens too

    summary = {
        "run_id": run_dir.name,
        "model": {"key": model_key, "client": model.client, "slug": model.slug},
        "prompt": prompt_info(prompt),
        "repair_prompt": prompt_info(repair_prompt),
        "params": run_params(config, model_key, sampling),
        "max_input_tokens": settings.max_input_tokens,
        "docs": doc_ids,
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": {
            **{s: sum(r["status"] == s for r in records.values()) for s in STATUSES},
            "not_run": len(doc_ids) - len(records),
        },
        # after one repair attempt; None while a record predates the repair step (S1-15)
        "final_counts": (
            {s: sum(r["final_status"] == s for r in records.values()) for s in FINAL_STATUSES}
            if all(r.get("final_status") for r in records.values())
            else None
        ),
        "cost_usd": round(sum(record_cost(r) for r in records.values()), 6),
        "spent_total_usd": round(spent_total, 6),
        "budget_stop_at_usd": config.budget.stop_at_usd,
        "tokens": {
            "prompt": sum(c["prompt_tokens"] or 0 for c in calls),
            "completion": sum(c["completion_tokens"] or 0 for c in calls),
        },
        "truncated": [d for d, r in records.items() if (r["input"] or {}).get("truncated")],
        "failures": [
            {
                "doc_id": d,
                "status": r["status"],
                "final_status": r.get("final_status"),
                "errors": r["errors"],
                "repair_errors": (r.get("repair") or {}).get("errors"),
            }
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


class RunMismatch(Exception):
    """The prompt file or the parameters differ from the saved run's, so it cannot be repaired."""


def repair_run(
    run_dir: Path,
    client: ChatClient,
    config: Config,
    spent_before: float,
    texts_dir: Path = TEXTS,
    on_result: Callable[[dict], None] = lambda record: None,
) -> dict:
    """Give each invalid record of a saved run its one repair attempt; set every final_status.

    config.extraction.prompt_version must be the run's, and client must be built with the run's
    model and sampling. spent_before is the cost already recorded under results/ (budget
    guard); the summary keeps the run's own spent_total_usd plus the repair calls. Records that
    already have a final_status are left alone, so running this twice makes no new calls.
    on_result gets each record that had a repair attempt. Returns the rebuilt summary.
    """
    previous = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    model_key = previous["model"]["key"]
    sampling = ((previous["params"] or {}).get("sampling") or {}).get("name")
    settings = config.extraction
    prompt = load_prompt(settings.prompt, settings.prompt_version)
    if prompt_info(prompt) != previous["prompt"]:
        raise RunMismatch(f"prompt is now {prompt_info(prompt)}, the run used {previous['prompt']}")
    if run_params(config, model_key, sampling) != previous["params"]:
        raise RunMismatch("parameters in config.toml differ from the run's")
    repair_cost = 0.0
    for doc_id in previous["docs"]:
        path = run_dir / f"{doc_id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("final_status"):
            continue
        if record["status"] == "invalid":
            if spent_before + repair_cost >= config.budget.stop_at_usd:
                break
            messages, _ = build_messages(doc_id, prompt, config, texts_dir)
            record["repair"] = repair_reply(
                messages, record["raw"], record["errors"], client, config
            )
            repair_cost += record_cost(record["repair"])
        else:
            record["repair"] = None
        set_final_status(record)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if record["repair"]:
            on_result(record)

    return write_summary(
        run_dir,
        previous["docs"],
        model_key,
        config,
        previous["started_at"],
        previous["finished_at"],
        previous["spent_total_usd"] + repair_cost,
        previous["stopped_by_budget"],
        sampling,
    )
