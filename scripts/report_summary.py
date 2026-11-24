"""Write the "Results at a glance" section of REPORT.md from saved results (S2-21).

Reads evals/results/retrieval.json, the latest run file in evals/results/runs/ (as
`make eval-gate` does), evals/results/calibration.json and evals/results/refusal.json (for the
`make cost` figure). No model calls; only the section between the summary markers changes.
Usage: PYTHONPATH=. uv run python scripts/report_summary.py
"""

import argparse
import json
from pathlib import Path

from evals.gate import GateError, evaluate
from evals.summary import END, START, section
from factory_docs.config import ROOT, load_config
from factory_docs.cost import summarize as cost_summary
from factory_docs.retrieval_eval import update_section

RESULTS = ROOT / "evals" / "results"
RETRIEVAL = RESULTS / "retrieval.json"
RUNS = RESULTS / "runs"
CALIBRATION = RESULTS / "calibration.json"
REFUSAL = RESULTS / "refusal.json"
REPORT = ROOT / "REPORT.md"


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    config = load_config()
    retrieval = read(RETRIEVAL)
    runs = sorted(RUNS.glob("*.json"))
    run_path = runs[-1] if runs else None
    run = read(run_path) if run_path else None
    refusal = read(REFUSAL)
    cost = None
    if refusal is not None:
        a = config.answer
        model = config.models[a.model]
        try:
            cost = cost_summary(
                refusal,
                a.prompt_version,
                a.min_rerank_score,
                a.model,
                model.price_in_usd_per_mtok,
                model.price_out_usd_per_mtok,
                model.price_checked.isoformat(),
            )
        except ValueError:
            cost = None  # refusal.json is for another prompt or model: shown as not run yet
    checks = None
    if retrieval is not None and run is not None:
        try:
            checks = evaluate(retrieval, run, config.eval_gate)
        except GateError:
            checks = None
    text = section(
        config.eval_gate,
        retrieval,
        run,
        str(run_path.relative_to(ROOT)) if run_path else None,
        read(CALIBRATION),
        cost,
        checks,
    )
    update_section(REPORT, text, START, END)
    print(f"Wrote the summary section of {REPORT.name}")


if __name__ == "__main__":
    main()
