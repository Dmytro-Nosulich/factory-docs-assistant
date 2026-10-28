"""Golden-set misses of the comparison runs, S1-18: what was extracted instead of what.

Reads the same runs as the comparison tables in REPORT.md (`factory_docs.report`: the newest
complete run of each [comparison] configuration) and scores the first response of every golden
document with the rules of docs/METRICS.md; an invalid or error result misses all 9 fields, as
in the headline. Equal misses (same document, field, extracted and expected value) are grouped,
with the configurations they occur in. Reads only saved results: no model calls.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from factory_docs.config import Config
from factory_docs.golden import GOLDEN, load_golden
from factory_docs.metrics import (
    FIELDS,
    component_match,
    hazard_match,
    load_run,
    markdown_table,
    score_document,
)
from factory_docs.report import cell_label, latest_complete_run, matrix_cells
from factory_docs.run import RUNS, doc_languages

ITEM_MATCH = {"components": component_match, "hazard_statements": hazard_match}
UNITS = {"percent": "%", "celsius": "°C"}


@dataclass
class Miss:
    """One wrong value of one field of one document, and the configurations that returned it."""

    doc_id: str
    field: str
    extracted: str
    expected: str
    cells: list[str] = field(default_factory=list)


def item_diff(
    predicted: list, golden: list, same: Callable[[dict, dict], bool]
) -> tuple[list, list]:
    """Items only in the output and items only in the label, paired one-to-one as in
    `metrics.match_list`."""
    extra = list(predicted)
    missing = []
    for item in golden:
        for i, candidate in enumerate(extra):
            if same(candidate, item):
                del extra[i]
                break
        else:
            missing.append(item)
    return extra, missing


def show_range(value: dict | None) -> str:
    """'[min, max] unit', e.g. '[99.0, null] %'; 'null' for no range."""
    if value is None:
        return "null"
    bounds = ", ".join("null" if value[b] is None else str(value[b]) for b in ("min", "max"))
    return f"[{bounds}] {UNITS[value['unit']]}"


def show_item(list_field: str, item: dict) -> str:
    if list_field == "components":
        return f"{item['name']} {item['cas_number']} {show_range(item['concentration'])}"
    return f"{item['code']} {item['text']}"


def show_value(name: str, value: object) -> str:
    if name == "flash_point":
        return show_range(value)
    return "null" if value is None else f'"{value}"'


def show_items(list_field: str, items: list) -> str:
    return "; ".join(show_item(list_field, item) for item in items) if items else "-"


def document_misses(output: dict | None, status: str, golden: dict) -> list[tuple[str, str, str]]:
    """(field, extracted, expected) for every field of one document that does not match.

    For a list field only the differing items are shown; a failed result shows its status."""
    score = score_document(output, golden)
    misses = []
    for name in FIELDS:
        if score.fields[name]:
            continue
        if output is None:
            extracted, expected = f"<{status}>", ""
        elif name in ITEM_MATCH:
            extra, missing = item_diff(output[name], golden[name], ITEM_MATCH[name])
            extracted, expected = show_items(name, extra), show_items(name, missing)
        else:
            extracted, expected = show_value(name, output[name]), show_value(name, golden[name])
        misses.append((name, extracted, expected))
    return misses


def collect_errors(
    config: Config,
    runs_dir: Path = RUNS,
    golden_dir: Path = GOLDEN,
    doc_ids: list[str] | None = None,
) -> tuple[list[str], list[Miss]]:
    """The run IDs of the comparison and their grouped misses, by document and field."""
    run_ids: list[str] = []
    grouped: dict[tuple[str, str, str, str], Miss] = {}
    for cell in matrix_cells(config):
        run_id = latest_complete_run(cell, config, runs_dir, doc_ids)
        if run_id is None:
            continue
        run_ids.append(run_id)
        for record in load_run(runs_dir / run_id)[1]:
            golden = load_golden(record["doc_id"], golden_dir)
            if golden is None:
                continue
            output = record["output"] if record["status"] == "valid" else None
            for name, extracted, expected in document_misses(output, record["status"], golden):
                key = (record["doc_id"], name, extracted, expected)
                miss = grouped.setdefault(key, Miss(*key))
                miss.cells.append(cell_label(cell))
    order = {name: i for i, name in enumerate(FIELDS)}
    misses = sorted(
        grouped.values(), key=lambda m: (m.doc_id, order[m.field], -len(m.cells), m.extracted)
    )
    return run_ids, misses


def cell_text(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_errors(
    run_ids: list[str], misses: list[Miss], languages: dict[str, str] | None = None
) -> str:
    """The misses as a Markdown table, with the runs read and the misses per field."""
    languages = doc_languages() if languages is None else languages
    rows = [
        [
            m.doc_id,
            languages.get(m.doc_id, "?"),
            m.field,
            f"{len(m.cells)}/{len(run_ids)}: " + ", ".join(m.cells),
            cell_text(m.extracted),
            cell_text(m.expected),
        ]
        for m in misses
    ]
    per_field = {name: sum(len(m.cells) for m in misses if m.field == name) for name in FIELDS}
    notes = [
        "Rules: docs/METRICS.md. First response only; a failed result (`<invalid>`, `<error>`) "
        "misses all 9 fields.",
        "Runs: n/total configurations that returned this value for this document. "
        "List fields show only the items that differ: extracted items not in the label, "
        "and label items not extracted ('-' for none).",
        "Misses per field (documents x runs): "
        + ", ".join(f"{name} {count}" for name, count in per_field.items())
        + f"; total {sum(per_field.values())}.",
    ]
    return "\n\n".join(
        [
            "Runs:\n" + "\n".join(f"- `{run_id}`" for run_id in run_ids),
            markdown_table(["Doc", "Lang", "Field", "Runs", "Extracted", "Expected"], rows),
            "\n".join(f"- {note}" for note in notes),
        ]
    )
