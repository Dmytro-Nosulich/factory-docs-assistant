"""Clean every parsed document into data/cleaned/<id>.json and report what changed (S2-04).

Reads data/parsed/, so run `make parse` first when the parsers change.
Usage: uv run python scripts/clean_corpus.py
"""

import csv
from pathlib import Path

from factory_docs.cleaning import (
    CleanResult,
    character_count,
    clean_corpus,
    find_duplicates,
    removal_rows,
)
from factory_docs.corpus import in_scope
from factory_docs.parsing.model import ParsedDocument, to_review_json

DATA = Path(__file__).resolve().parent.parent / "data"
PARSED = DATA / "parsed"
CLEANED = DATA / "cleaned"
REPORT = CLEANED / "clean_report.csv"
REMOVED = CLEANED / "removed.csv"
RULES = [
    "zone_header",
    "zone_footer",
    "zone_nav",
    "zone_aside",
    "zone_toc",
    "header_kept",
    "footer_restored",
    "repeated_line",
    "contents_title",
    "toc_line",
    "back_to_top",
    "nfc",
    "ligature",
    "soft_hyphen",
    "broken_glyph",
    "symbol_glyph",
    "icon_glyph",
    "control_char",
    "hyphen_join",
    "hyphen_keep",
    "hyphen_suspended",
    "markdown_syntax",
]
COLUMNS = [
    "id",
    "format",
    "blocks_before",
    "blocks_after",
    "chars_before",
    "chars_after",
    *RULES,
    "content_hash",
    "duplicate_of",
]
REMOVED_COLUMNS = ["id", "rule", "pages", "times", "text"]


def report_row(result: CleanResult, duplicate_of: str) -> dict[str, str | int]:
    document = result.document
    return {
        "id": document.doc_id,
        "format": document.format,
        "blocks_before": result.blocks_before,
        "blocks_after": len(document.blocks),
        "chars_before": result.chars_before,
        "chars_after": character_count(document.blocks),
        **{rule: result.counts[rule] for rule in RULES},
        "content_hash": document.content_hash,
        "duplicate_of": duplicate_of,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = in_scope()
    documents = [
        (
            ParsedDocument.model_validate_json(
                (PARSED / f"{row['id']}.json").read_text(encoding="utf-8")
            ),
            row["language"],
        )
        for row in rows
    ]
    results = clean_corpus(documents)
    hashes = {result.document.doc_id: result.document.content_hash for result in results}
    supersedes = {row["id"]: row["supersedes"] for row in rows if row["supersedes"]}
    duplicates = find_duplicates(hashes, supersedes)

    CLEANED.mkdir(parents=True, exist_ok=True)
    for path in CLEANED.glob("D*.json"):
        path.unlink()  # a document that left the scope or became a duplicate must not stay
    for result in results:
        if result.document.doc_id not in duplicates:
            path = CLEANED / f"{result.document.doc_id}.json"
            path.write_text(to_review_json(result.document), encoding="utf-8")

    write_csv(
        REPORT,
        COLUMNS,
        [report_row(r, duplicates.get(r.document.doc_id, "")) for r in results],
    )
    write_csv(
        REMOVED,
        REMOVED_COLUMNS,
        [row for r in results for row in removal_rows(r.document.doc_id, r.removals)],
    )
    print(f"{len(results) - len(duplicates)} documents cleaned, {len(duplicates)} duplicates")
    print(f"Cleaned files and reports: {CLEANED.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
