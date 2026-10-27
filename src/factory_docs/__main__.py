"""Entry point for `python -m factory_docs` (used by `make run`): the full extraction run, S1-08.

Usage: uv run python -m factory_docs --all [--docs D001 D003]
       uv run python -m factory_docs [--model haiku-5.5] [--sampling greedy] [--docs D001 D003]
                                    [--prompt-version v2] [--resume RUN_ID]
       uv run python -m factory_docs --repair RUN_ID
One model call per document (retries included), plus one repair call for an invalid reply;
paid for cloud models and free for local (Ollama) ones; see factory_docs.run for the layout.
--all (plain `make run`, S1-17) runs every configuration in [comparison] in config.toml, one
new run each, then rewrites the comparison tables in REPORT.md (factory_docs.report).
--repair gives the invalid replies of a saved run their repair attempt (S1-15), with the
run's model, sampling and prompt version.
"""

import argparse
import json
from pathlib import Path

from factory_docs.clients import make_client
from factory_docs.config import (
    Config,
    OllamaModel,
    OpenRouterModel,
    load_config,
    load_env,
    params_match,
    require_env,
    resolve_sampling,
    run_params,
)
from factory_docs.ollama import check_ollama
from factory_docs.prompts import load_prompt
from factory_docs.report import Cell, cell_label, matrix_cells, update_report
from factory_docs.run import (
    RUNS,
    SUMMARY,
    RunMismatch,
    new_run_id,
    repair_run,
    run_all,
    spent_usd,
    stage1_doc_ids,
)


def print_result(record: dict) -> None:
    call, inp = record["call"] or {}, record["input"] or {}
    line = f"{record['doc_id']} {record['status']:<7}"
    repair = record.get("repair")
    if repair:
        line += f" -> {record['final_status']}"
    if record["status"] != "error":
        line += f" {call['latency_s']:6.1f} s  ${call['cost_usd'] or 0:.4f}"
    if repair and repair["status"] != "error":
        fixed = repair["call"]
        line += f"  (repair {fixed['latency_s']:.1f} s  ${fixed['cost_usd'] or 0:.4f})"
    if inp.get("truncated"):
        line += f"  truncated {inp['pages_sent']}/{inp['pages_total']} pages"
    if record["errors"]:
        line += f"  {record['errors'][0][:120]}"
    if repair and repair["errors"]:
        line += f"  repair: {repair['errors'][0][:120]}"
    print(line, flush=True)


def print_summary(summary: dict) -> None:
    counts = summary["counts"]
    print(f"\nrun {summary['run_id']}")
    print("  first response: " + ", ".join(f"{status}: {n}" for status, n in counts.items()))
    if summary["final_counts"]:
        final = summary["final_counts"]
        print("  after repair: " + ", ".join(f"{status}: {n}" for status, n in final.items()))
    print(
        f"  cost: ${summary['cost_usd']:.4f} this run, ${summary['spent_total_usd']:.4f} "
        f"in all results/ (stop at ${summary['budget_stop_at_usd']:.2f})"
    )
    if summary["stopped_by_budget"]:
        print(f"  STOPPED BY BUDGET: {len(summary['not_run'])} documents not run")
    if summary["truncated"]:
        print(f"  truncated: {' '.join(summary['truncated'])}")
    for failure in summary["failures"]:
        status = failure["status"]
        if failure.get("final_status") and failure["final_status"] != status:
            status += f" -> {failure['final_status']}"
        print(f"  {status}: {failure['doc_id']}: {failure['errors'][0][:200]}")
    print(f"  saved: results/runs/{summary['run_id']}/{SUMMARY}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract every stage 1 document with one model.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every configuration in [comparison] in config.toml, then the REPORT.md tables",
    )
    parser.add_argument("--model", help="model key from config.toml (default: haiku-5.5)")
    parser.add_argument(
        "--sampling", help="[sampling] table from config.toml (local models; default: the model's)"
    )
    parser.add_argument("--docs", nargs="+", help="document IDs (default: all of stage 1)")
    parser.add_argument(
        "--prompt-version", help="prompts/<prompt>_<version>.md (default: from config.toml)"
    )
    once = parser.add_mutually_exclusive_group()
    once.add_argument("--resume", metavar="RUN_ID", help="fill in documents missing from a run")
    once.add_argument(
        "--repair", metavar="RUN_ID", help="one repair attempt for each invalid reply of a run"
    )
    args = parser.parse_args(argv)

    if args.all:
        if args.model or args.sampling or args.prompt_version or args.resume or args.repair:
            parser.error("--all runs every configuration in [comparison]; it takes only --docs")
        run_matrix(args.docs or stage1_doc_ids(), parser)
        return
    config = load_config()
    if args.repair:
        repair_saved_run(args.repair, config, parser)
        return
    args.model = args.model or "haiku-5.5"
    if args.model not in config.models:
        parser.error(f"unknown model {args.model!r}; config.toml has {', '.join(config.models)}")
    previous = None
    if args.resume:
        run_dir = RUNS / args.resume
        if not (run_dir / SUMMARY).exists():
            parser.error(f"no {SUMMARY} in results/runs/{args.resume}")
        previous = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
        if args.sampling is None:  # resume with the sampling the run started with
            args.sampling = (previous["params"].get("sampling") or {}).get("name")
        if args.prompt_version is None:  # and with its prompt version
            args.prompt_version = previous["prompt"]["version"]
    if args.prompt_version:
        config.extraction.prompt_version = args.prompt_version
    try:
        resolved = resolve_sampling(config, args.model, args.sampling)
        prompt = load_prompt(config.extraction.prompt, config.extraction.prompt_version)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    sampling = resolved[0] if resolved else None
    if previous:
        if (
            previous["model"]["key"] != args.model
            or previous["prompt"]["sha256"] != prompt.sha256
            or not params_match(previous["params"], run_params(config, args.model, sampling))
        ):
            parser.error(
                "model, prompt or parameters differ from the run being resumed; start a new run"
            )
        doc_ids = previous["docs"]
    else:
        run_dir = RUNS / new_run_id(args.model, config.extraction.prompt_version, sampling)
        doc_ids = args.docs or stage1_doc_ids()

    load_env()
    run_one(config, args.model, sampling, run_dir, doc_ids)


def run_one(
    config: Config, model_key: str, sampling: str | None, run_dir: Path, doc_ids: list[str]
) -> dict:
    """One configuration over doc_ids into run_dir (new or resumed); prints and returns the summary.

    config.extraction.prompt_version must already be the run's.
    """
    client = make_client(model_key, config, sampling)
    model_line = model_key + (f", sampling {sampling}" if sampling else "")
    print(f"{len(doc_ids)} documents, model {model_line}, results in results/runs/{run_dir.name}")
    try:
        summary = run_all(
            doc_ids,
            client,
            model_key,
            config,
            run_dir,
            spent_usd(),
            on_result=print_result,
            sampling=sampling,
        )
    finally:
        client.close()
    print_summary(summary)
    return summary


def run_matrix(doc_ids: list[str], parser: argparse.ArgumentParser) -> None:
    """Plain `make run` (S1-17): one new run per [comparison] cell, then the REPORT.md tables."""
    config = load_config()
    cells = matrix_cells(config)
    try:
        preflight(config, cells)
    except (RuntimeError, FileNotFoundError) as e:
        parser.error(str(e))
    print(f"{len(cells)} configurations x {len(doc_ids)} documents")
    made = []
    for k, cell in enumerate(cells, 1):
        model_key, sampling, version = cell
        print(f"\n[{k}/{len(cells)}] {cell_label(cell)}")
        config = load_config()  # fresh per cell, so one cell's prompt version cannot leak
        config.extraction.prompt_version = version
        run_dir = RUNS / new_run_id(model_key, version, sampling)
        run_one(config, model_key, sampling, run_dir, doc_ids)
        made.append(run_dir.name)

    run_ids, missing = update_report(load_config())
    print(f"\nREPORT.md: comparison tables rebuilt from {len(run_ids)} runs")
    left_out = [run_id for run_id in made if run_id not in run_ids]
    if left_out:
        print(
            f"  {len(left_out)} of the {len(made)} runs just made are not complete (a DOCS subset "
            "or a budget stop), so the tables use earlier runs or leave the cell out: "
            + ", ".join(left_out)
        )
    if missing:
        print("  no complete run: " + ", ".join(cell_label(c) for c in missing))


def preflight(config: Config, cells: list[Cell]) -> None:
    """Everything the matrix needs, checked before the first model call."""
    settings = config.extraction
    for _, _, version in cells:
        load_prompt(settings.prompt, version)
    load_prompt(settings.repair_prompt, settings.repair_prompt_version)
    models = [config.models[key] for key in dict.fromkeys(key for key, _, _ in cells)]
    load_env()
    if any(isinstance(m, OpenRouterModel) for m in models):
        require_env("OPENROUTER_API_KEY")
    local = [m.slug for m in models if isinstance(m, OllamaModel)]
    if local:
        check_ollama(config.ollama, local)


def repair_saved_run(run_id: str, config: Config, parser: argparse.ArgumentParser) -> None:
    run_dir = RUNS / run_id
    if not (run_dir / SUMMARY).exists():
        parser.error(f"no {SUMMARY} in results/runs/{run_id}")
    previous = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    model_key = previous["model"]["key"]
    sampling = ((previous["params"] or {}).get("sampling") or {}).get("name")
    config.extraction.prompt_version = previous["prompt"]["version"]

    load_env()
    client = make_client(model_key, config, sampling)
    print(f"repairing invalid replies in results/runs/{run_id}")
    try:
        summary = repair_run(run_dir, client, config, spent_usd(), on_result=print_result)
    except RunMismatch as e:
        parser.error(str(e))
    finally:
        client.close()
    print_summary(summary)


if __name__ == "__main__":
    main()
