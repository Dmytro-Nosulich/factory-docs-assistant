"""Chunk every cleaned document two ways and write data/chunks/ (S2-05).

Reads data/cleaned/, so run `make clean-corpus` first when the cleaning rules change.
Writes chunks.jsonl (structure-aware, the chunks S2-06 indexes), chunks_fixed.jsonl (the
fixed-size baseline), ingest_manifest.csv (one row per document) and chunking_comparison.md.
Usage: uv run python scripts/ingest.py
"""

import csv
from collections.abc import Sequence
from pathlib import Path

from factory_docs.chunking import (
    MAX_CHARS,
    MIN_CHARS,
    OVERLAP,
    SHORT_CHARS,
    DocumentInfo,
    Draft,
    Stats,
    Strategy,
    chunk_line,
    chunk_stats,
    drafts_for,
    render_block,
    render_document,
    section_label,
    to_chunks,
)
from factory_docs.cleaning import CleanedDocument
from factory_docs.corpus import in_scope
from factory_docs.metrics import percentile
from factory_docs.parsing.model import Table

DATA = Path(__file__).resolve().parent.parent / "data"
CLEANED = DATA / "cleaned"
CLEAN_REPORT = CLEANED / "clean_report.csv"
CHUNKS = DATA / "chunks"
OUTPUT: dict[Strategy, Path] = {
    "structure": CHUNKS / "chunks.jsonl",
    "fixed": CHUNKS / "chunks_fixed.jsonl",
}
MANIFEST_OUT = CHUNKS / "ingest_manifest.csv"
COMPARISON = CHUNKS / "chunking_comparison.md"
STRATEGIES: tuple[Strategy, ...] = ("structure", "fixed")
# Every format, three languages, short and long documents (user's choice, 2026-11-19)
EXAMPLES = ["D001", "D007", "D033", "D029", "D066", "D073", "D074"]
MANIFEST_COLUMNS = [
    "id",
    "format",
    "language",
    "content_hash",
    "blocks",
    "chars",
    *(f"{s}_{c}" for s in STRATEGIES for c in ("chunks", "median_chars", "max_chars")),
]

Chunked = dict[Strategy, list[Draft]]


def load_documents() -> list[tuple[dict[str, str], CleanedDocument]]:
    """Every in-scope cleaned document, checked against the hash S2-04 reported."""
    with CLEAN_REPORT.open(encoding="utf-8", newline="") as f:
        report = {row["id"]: row for row in csv.DictReader(f)}
    documents = []
    for row in sorted(in_scope(), key=lambda row: row["id"]):
        if report[row["id"]]["duplicate_of"]:
            continue  # a duplicate has no cleaned file and is not indexed
        path = CLEANED / f"{row['id']}.json"
        document = CleanedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if document.content_hash != report[row["id"]]["content_hash"]:
            raise SystemExit(f"{row['id']}: content hash differs from {CLEAN_REPORT.name}")
        documents.append((row, document))
    return documents


def median(sizes: Sequence[int]) -> int | str:
    return percentile(list(sizes), 50) if sizes else ""


def manifest_row(row: dict[str, str], document: CleanedDocument, chunked: Chunked) -> dict:
    out: dict[str, str | int] = {
        "id": document.doc_id,
        "format": document.format,
        "language": row["language"],
        "content_hash": document.content_hash,
        "blocks": len(document.blocks),
        "chars": len(render_document(document)[0]),
    }
    for strategy in STRATEGIES:
        sizes = [len(draft.text) for draft in chunked[strategy]]
        out[f"{strategy}_chunks"] = len(sizes)
        out[f"{strategy}_median_chars"] = median(sizes)
        out[f"{strategy}_max_chars"] = max(sizes, default="")
    return out


# The comparison report
def share(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.0f} %)" if whole else "0"


def stats_cells(stats: Stats) -> list[str]:
    sizes = list(stats.sizes)
    return [
        str(stats.chunks),
        f"{stats.chars:,}",
        str(percentile(sizes, 50)),
        str(percentile(sizes, 95)),
        str(max(sizes)),
        share(stats.one_section, stats.chunks),
        f"{stats.tables_whole} of {stats.tables}" if stats.tables else "no tables",
        share(stats.lowercase_starts, stats.chunks),
        share(stats.short, stats.chunks),
    ]


STATS_HEADER = (
    "| Chunks | Characters | p50 | p95 | Max | In one top-level section | Tables kept whole "
    f"| Start lowercase | Under {SHORT_CHARS} chars |"
)


def example(document: CleanedDocument, chunked: Chunked) -> list[str]:
    """The first table that fits in a chunk (or the second chunk), as each strategy cuts it."""
    text, spans = render_document(document)
    structure = chunked["structure"]
    tables = [
        b for b in document.blocks if isinstance(b, Table) and len(render_block(b)) <= MAX_CHARS
    ]
    if tables:
        chosen = next(d for d in structure if any(b is tables[0] for b in d.blocks))
        target = tables[0]
        what = "the first table that fits in one chunk"
    else:
        chosen = structure[min(1, len(structure) - 1)]
        target = chosen.blocks[0]
        what = "the start of the second structure chunk"
    start, end = next((s, e) for s, e, block in spans if block is target)
    fixed = [d for d in chunked["fixed"] if d.start is not None and d.start <= start < d.end]
    covering = [d for d in chunked["fixed"] if d.start < end and d.end > start]
    lines = [f"### {document.doc_id}", "", f"Shown: {what}.", ""]
    s_index = structure.index(chosen)
    f_index = chunked["fixed"].index(fixed[-1])
    lines += [
        f"**Structure chunk {s_index}** ({len(chosen.text)} chars, "
        f"section: {section_label(chosen.section_path)}):",
        "",
        "~~~text",
        chosen.text,
        "~~~",
        "",
        f"**Fixed chunk {f_index}** ({len(fixed[-1].text)} chars; "
        f"this passage spans {len(covering)} fixed chunk(s)):",
        "",
        "~~~text",
        fixed[-1].text,
        "~~~",
        "",
    ]
    return lines


def comparison(documents: list[tuple[dict[str, str], CleanedDocument]], chunked: dict) -> str:
    lines = [
        "# Chunking comparison (S2-05)",
        "",
        "Generated by `make ingest` from `data/cleaned/`; do not edit by hand.",
        "",
        f"- **structure**: whole blocks packed up to {MAX_CHARS} characters, never across a "
        f"top-level section, closed at a heading from {MIN_CHARS} characters; a chunk under "
        f"{MIN_CHARS} joins a neighbour in the same top-level section when both fit. Only a "
        "block over the limit is split (sentences, list items, table rows with the header "
        "repeated).",
        f"- **fixed**: the document text cut every {MAX_CHARS} characters with {OVERLAP} "
        "characters of overlap, each cut moved back to whitespace.",
        "",
        "Columns: *p50 / p95 / Max* are chunk sizes in characters (nearest rank). *In one "
        "top-level section*: chunks whose text all comes from one top-level section (for an SDS, "
        "one of its 16 sections). *Tables kept whole*: tables whose full text is inside one "
        f"chunk; tables over {MAX_CHARS} characters cannot be. *Start lowercase*: chunks that "
        "begin with a lowercase letter, a sign of a cut mid-sentence. *Characters* include the "
        "overlap and repeated table headers.",
        "",
        f"## All {len(documents)} documents",
        "",
        f"| Strategy {STATS_HEADER}",
        "|---" * 10 + "|",
    ]
    for strategy in STRATEGIES:
        stats = chunk_stats((doc, chunked[doc.doc_id][strategy]) for _, doc in documents)
        lines.append(f"| {strategy} | " + " | ".join(stats_cells(stats)) + " |")
    lines += [
        "",
        "## Selected documents",
        "",
        f"| Document | Format | Language | Strategy {STATS_HEADER}",
        "|---" * 13 + "|",
    ]
    selected = [(row, doc) for row, doc in documents if doc.doc_id in EXAMPLES]
    selected.sort(key=lambda pair: EXAMPLES.index(pair[1].doc_id))
    for row, doc in selected:
        for strategy in STRATEGIES:
            stats = chunk_stats([(doc, chunked[doc.doc_id][strategy])])
            lines.append(
                f"| {doc.doc_id} | {doc.format} | {row['language']} | {strategy} | "
                + " | ".join(stats_cells(stats))
                + " |"
            )
    lines += ["", "## Examples", ""]
    for _, doc in selected:
        lines += example(doc, chunked[doc.doc_id])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    documents = load_documents()
    chunked: dict[str, Chunked] = {
        doc.doc_id: {strategy: drafts_for(doc, strategy) for strategy in STRATEGIES}
        for _, doc in documents
    }
    CHUNKS.mkdir(parents=True, exist_ok=True)
    for strategy, path in OUTPUT.items():
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row, doc in documents:
                info = DocumentInfo.from_manifest(row)
                for chunk in to_chunks(chunked[doc.doc_id][strategy], doc, info, strategy):
                    f.write(chunk_line(chunk))
    with MANIFEST_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_row(row, doc, chunked[doc.doc_id]) for row, doc in documents)
    COMPARISON.write_text(comparison(documents, chunked), encoding="utf-8")

    counts = {s: sum(len(chunked[d.doc_id][s]) for _, d in documents) for s in STRATEGIES}
    print(
        f"{len(documents)} documents: {counts['structure']} structure chunks, "
        f"{counts['fixed']} fixed chunks"
    )
    print(f"Chunks, manifest and comparison: {CHUNKS.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
