"""Check that chunk section labels name the section of their text (S2-12, decision 19).

Reads the gold set, the cleaned documents, both chunk files and evals/results/retrieval.json.
Calls no model. Writes evals/results/section_labels.md and the generated chunking table in
README.md, whose other numbers come from the same chunker (`make ingest`) and from
`make eval-retrieval`. Rules in docs/METRICS.md.

Usage: uv run python scripts/section_labels.py
"""

from factory_docs.config import ROOT
from factory_docs.retrieval_eval import update_section
from factory_docs.section_labels import (
    END,
    MARKDOWN,
    README,
    START,
    STRATEGIES,
    build_outputs,
    passed,
)


def main() -> None:
    try:
        outputs = build_outputs()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    MARKDOWN.write_text(outputs.markdown, encoding="utf-8")
    update_section(README, outputs.readme_block, START, END)
    for strategy in STRATEGIES:
        c = outputs.corpus[strategy]
        print(
            f"{strategy}: gold {passed(outputs.labels, strategy)}/{len(outputs.labels)}, "
            f"pairs {c.pairs_correct}/{c.pairs}, chunks {c.chunks_correct}/{c.chunks}"
        )
    print(f"Wrote {MARKDOWN.relative_to(ROOT)} and the chunking table in README.md")


if __name__ == "__main__":
    main()
