"""Entry point for `python -m factory_docs` (used by `make run`): the full extraction run, S1-08.

Usage: uv run python -m factory_docs [--model haiku-5.5] [--docs D001 D003] [--resume RUN_ID]
One model call per document (retries included), paid for cloud models and free for local
(Ollama) ones; see factory_docs.run for the layout.
"""

import argparse
import json

from factory_docs.clients import make_client
from factory_docs.config import load_config, load_env
from factory_docs.prompts import load_prompt
from factory_docs.run import RUNS, SUMMARY, new_run_id, run_all, spent_usd, stage1_doc_ids


def print_result(record: dict) -> None:
    call, inp = record["call"] or {}, record["input"] or {}
    line = f"{record['doc_id']} {record['status']:<7}"
    if record["status"] != "error":
        line += f" {call['latency_s']:6.1f} s  ${call['cost_usd'] or 0:.4f}"
    if inp.get("truncated"):
        line += f"  truncated {inp['pages_sent']}/{inp['pages_total']} pages"
    if record["errors"]:
        line += f"  {record['errors'][0][:120]}"
    print(line, flush=True)


def print_summary(summary: dict) -> None:
    counts = summary["counts"]
    print(f"\nrun {summary['run_id']}")
    print("  " + ", ".join(f"{status}: {n}" for status, n in counts.items()))
    print(
        f"  cost: ${summary['cost_usd']:.4f} this run, ${summary['spent_total_usd']:.4f} "
        f"in all results/ (stop at ${summary['budget_stop_at_usd']:.2f})"
    )
    if summary["stopped_by_budget"]:
        print(f"  STOPPED BY BUDGET: {len(summary['not_run'])} documents not run")
    if summary["truncated"]:
        print(f"  truncated: {' '.join(summary['truncated'])}")
    for failure in summary["failures"]:
        print(f"  {failure['status']}: {failure['doc_id']}: {failure['errors'][0][:200]}")
    print(f"  saved: results/runs/{summary['run_id']}/{SUMMARY}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract every stage 1 document with one model.")
    parser.add_argument("--model", default="haiku-5.5", help="model key from config.toml")
    parser.add_argument("--docs", nargs="+", help="document IDs (default: all of stage 1)")
    parser.add_argument("--resume", metavar="RUN_ID", help="fill in documents missing from a run")
    args = parser.parse_args(argv)

    config = load_config()
    if args.model not in config.models:
        parser.error(f"unknown model {args.model!r}; config.toml has {', '.join(config.models)}")
    if args.resume:
        run_dir = RUNS / args.resume
        if not (run_dir / SUMMARY).exists():
            parser.error(f"no {SUMMARY} in results/runs/{args.resume}")
        previous = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
        settings = config.extraction
        prompt = load_prompt(settings.prompt, settings.prompt_version)
        if previous["model"]["key"] != args.model or previous["prompt"]["sha256"] != prompt.sha256:
            parser.error("model or prompt differs from the run being resumed; start a new run")
        doc_ids = previous["docs"]
    else:
        run_dir = RUNS / new_run_id(args.model, config.extraction.prompt_version)
        doc_ids = args.docs or stage1_doc_ids()

    load_env()
    client = make_client(args.model, config)
    print(f"{len(doc_ids)} documents, model {args.model}, results in results/runs/{run_dir.name}")
    try:
        summary = run_all(
            doc_ids, client, args.model, config, run_dir, spent_usd(), on_result=print_result
        )
    finally:
        client.close()
    print_summary(summary)


if __name__ == "__main__":
    main()
