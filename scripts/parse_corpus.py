"""Parse every in-scope document into data/parsed/<id>.json and write parse_report.csv (S2-03).

Usage: uv run python scripts/parse_corpus.py [--ids D001 D067]
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from factory_docs.corpus import in_scope
from factory_docs.parsing import parse_document
from factory_docs.parsing.model import ListBlock, ParsedDocument, to_review_json

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
PARSED = DATA / "parsed"
REPORT = PARSED / "parse_report.csv"
COLUMNS = [
    "id",
    "format",
    "pages",
    "pages_without_text",
    "headings",
    "paragraphs",
    "lists",
    "list_items",
    "tables",
    "code_blocks",
    "non_main_blocks",
    "warnings",
]


def report_row(document: ParsedDocument) -> dict[str, str | int]:
    main = [block for block in document.blocks if block.zone == "main"]
    kinds = Counter(block.kind for block in main)
    return {
        "id": document.doc_id,
        "format": document.format,
        "pages": document.page_count if document.page_count is not None else "",
        "pages_without_text": " ".join(str(page) for page in document.pages_without_text),
        "headings": kinds["heading"],
        "paragraphs": kinds["paragraph"],
        "lists": kinds["list"],
        "list_items": sum(len(b.items) for b in main if isinstance(b, ListBlock)),
        "tables": kinds["table"],
        "code_blocks": kinds["code"],
        "non_main_blocks": len(document.blocks) - len(main),
        "warnings": " | ".join(document.warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="parse only these IDs and keep the old report")
    args = parser.parse_args()
    # D001 makes pypdf warn about a font that only holds the space character (docs/ISSUES.md).
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    rows = [r for r in in_scope() if not args.ids or r["id"] in args.ids]
    PARSED.mkdir(parents=True, exist_ok=True)
    report = []
    for row in rows:
        document = parse_document(row["id"], RAW / f"{row['id']}.{row['format']}")
        (PARSED / f"{row['id']}.json").write_text(to_review_json(document), encoding="utf-8")
        report.append(report_row(document))
    print(f"{len(report)} documents parsed: {dict(Counter(r['format'] for r in report))}")
    if args.ids:
        return
    in_scope_ids = {row["id"] for row in rows}
    for path in PARSED.glob("D*.json"):
        if path.stem not in in_scope_ids:
            path.unlink()  # a document that left the scope must not keep an old parsed file
    with REPORT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report)
    print(f"Parsed files and report: {PARSED.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
