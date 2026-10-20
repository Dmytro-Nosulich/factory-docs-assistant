"""Extract text from stage-1 PDFs into data/texts/<id>.txt and write data/texts/index.csv.

Usage: uv run python scripts/extract_texts.py [--ids D001 D003]
"""

import argparse
import csv
import logging
from pathlib import Path

from factory_docs.pdf_text import estimate_tokens, extract_pages, to_marked_text

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA / "manifest.csv"
RAW = DATA / "raw"
TEXTS = DATA / "texts"
INDEX = TEXTS / "index.csv"
INDEX_COLUMNS = ["id", "pages", "empty_pages", "chars", "est_tokens"]


def stage1_ids() -> list[str]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r["id"] for r in csv.DictReader(f) if "1" in r["stage"].split(",")]


def extract_one(doc_id: str) -> dict[str, str | int]:
    pages = extract_pages(RAW / f"{doc_id}.pdf")
    text = to_marked_text(pages)
    (TEXTS / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    empty = [str(n) for n, p in enumerate(pages, start=1) if not p]
    return {
        "id": doc_id,
        "pages": len(pages),
        "empty_pages": " ".join(empty),
        "chars": len(text),
        "est_tokens": estimate_tokens(text),
    }


def read_index() -> dict[str, dict[str, str]]:
    if not INDEX.exists():
        return {}
    with INDEX.open(encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_index(rows: dict[str, dict]) -> None:
    with INDEX.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for doc_id in sorted(rows):
            writer.writerow(rows[doc_id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="re-extract only these document IDs")
    args = parser.parse_args()
    # D001 makes pypdf warn ~200 times about a font that only holds the space character;
    # the text is unaffected (see docs/ISSUES.md), so keep only errors.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    ids = args.ids or stage1_ids()
    TEXTS.mkdir(parents=True, exist_ok=True)
    # Keep index rows of other text files that still exist (e.g. non-SDS texts kept for stage 2).
    rows: dict[str, dict] = {
        doc_id: r for doc_id, r in read_index().items() if (TEXTS / f"{doc_id}.txt").exists()
    }
    for doc_id in ids:
        rows[doc_id] = extract_one(doc_id)
        print(f"{doc_id}: {rows[doc_id]['pages']} pages, ~{rows[doc_id]['est_tokens']} tokens")
    write_index(rows)

    total = sum(int(r["est_tokens"]) for r in rows.values())
    print(f"\n{len(rows)} text files in {TEXTS.relative_to(DATA.parent)}, ~{total} tokens total")
    for r in rows.values():
        if r["empty_pages"]:
            print(f"  {r['id']}: no text on pages {r['empty_pages']}")


if __name__ == "__main__":
    main()
