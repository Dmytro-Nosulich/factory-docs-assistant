"""Cost per 1000 questions from the saved refusal eval, no model calls (S2-15).

Reads evals/results/refusal.json, applies `[answer] min_rerank_score` and prices the
`[answer] prompt_version` run with the `[answer] model` prices in config.toml, by the rule in
docs/METRICS.md (`factory_docs.cost`). Writes evals/results/cost.md and the generated cost
section of REPORT.md, and prints a one-line summary.
Usage: uv run python scripts/cost_report.py
"""

import json

from factory_docs.config import ROOT, load_config
from factory_docs.cost import END, START, markdown, one_line, report_section, summarize
from factory_docs.retrieval_eval import update_section

OUT = ROOT / "evals" / "results"
SOURCE = OUT / "refusal.json"
MD_OUT = OUT / "cost.md"
REPORT = ROOT / "REPORT.md"


def main() -> None:
    config = load_config()
    settings = config.answer
    model = config.models[settings.model]
    if not SOURCE.exists():
        raise SystemExit(f"No {SOURCE.relative_to(ROOT)}; run `make eval-refusal` first")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    try:
        summary = summarize(
            data,
            settings.prompt_version,
            settings.min_rerank_score,
            settings.model,
            model.price_in_usd_per_mtok,
            model.price_out_usd_per_mtok,
            model.price_checked.isoformat(),
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    MD_OUT.write_text(markdown(summary), encoding="utf-8")
    update_section(REPORT, report_section(summary), START, END)
    for path in (MD_OUT, REPORT):
        print(f"Wrote {path.relative_to(ROOT)}")
    print(one_line(summary))


if __name__ == "__main__":
    main()
