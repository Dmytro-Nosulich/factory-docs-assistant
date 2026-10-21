"""Step 2 measurements of a run, S1-11: field accuracy on the golden set, % valid, latency, cost.

Every rule here is written down, with the reasons, in docs/METRICS.md; keep the two in step.
A run is read only from its files in results/runs/<run_id>/, so the numbers can be re-computed
at any time without calling a model.
"""

import json
import math
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from factory_docs.config import Config, OllamaModel
from factory_docs.golden import GOLDEN, load_golden
from factory_docs.run import SUMMARY
from factory_docs.schema import SafetyDataSheet

FIELDS = list(SafetyDataSheet.model_fields)
TOLERANCE = {"celsius": 0.5, "percent": 0.01}  # docs/METRICS.md: numbers
# Item-level scores: name -> (list field, item matcher); "hazard_codes" is a diagnostic only.
ITEM_SCORES = ("components", "hazard_statements", "hazard_codes")


def normalize_text(text: str) -> str:
    """NFKC, casefold, punctuation and symbols to spaces, whitespace collapsed (docs/METRICS.md)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def ranges_match(predicted: dict | None, golden: dict | None) -> bool:
    """Same unit; min and max each both null or within the unit's tolerance."""
    if predicted is None or golden is None:
        return predicted is golden
    if predicted["unit"] != golden["unit"]:
        return False
    tolerance = TOLERANCE[golden["unit"]]
    for bound in ("min", "max"):
        p, g = predicted[bound], golden[bound]
        if (p is None) != (g is None) or (p is not None and abs(p - g) > tolerance):
            return False
    return True


def component_match(predicted: dict, golden: dict) -> bool:
    return (
        normalize_text(predicted["name"]) == normalize_text(golden["name"])
        and predicted["cas_number"] == golden["cas_number"]
        and ranges_match(predicted["concentration"], golden["concentration"])
    )


def hazard_match(predicted: dict, golden: dict) -> bool:
    """Code exactly (case matters: H360Fd is not H360FD) and normalized text."""
    return predicted["code"] == golden["code"] and normalize_text(
        predicted["text"]
    ) == normalize_text(golden["text"])


def hazard_code_match(predicted: dict, golden: dict) -> bool:
    return predicted["code"] == golden["code"]


def match_list(predicted: list, golden: list, same: Callable[[dict, dict], bool]) -> int:
    """Number of one-to-one matched pairs, order ignored (greedy: first unused match)."""
    unused = list(predicted)
    matched = 0
    for item in golden:
        for i, candidate in enumerate(unused):
            if same(candidate, item):
                del unused[i]
                matched += 1
                break
    return matched


@dataclass
class ItemCounts:
    """Item-level counts of a list field; summed over documents for micro P/R/F1."""

    matched: int = 0
    predicted: int = 0
    golden: int = 0

    def add(self, other: "ItemCounts") -> None:
        self.matched += other.matched
        self.predicted += other.predicted
        self.golden += other.golden

    @property
    def f1(self) -> float:
        """2·matched / (predicted + golden), the harmonic mean of precision and recall."""
        total = self.predicted + self.golden
        return 2 * self.matched / total if total else 1.0


@dataclass
class DocScore:
    fields: dict[str, bool]
    items: dict[str, ItemCounts] = field(default_factory=dict)


def score_document(output: dict | None, golden: dict) -> DocScore:
    """Field-by-field match of one output against its golden label; None (failed) misses all."""
    items = {}
    for name, list_field, same in (
        ("components", "components", component_match),
        ("hazard_statements", "hazard_statements", hazard_match),
        ("hazard_codes", "hazard_statements", hazard_code_match),
    ):
        predicted = output[list_field] if output else []
        matched = match_list(predicted, golden[list_field], same)
        items[name] = ItemCounts(matched, len(predicted), len(golden[list_field]))
    if output is None:
        return DocScore({f: False for f in FIELDS}, items)

    fields = {f: output[f] == golden[f] for f in FIELDS}
    fields["product_name"] = normalize_text(output["product_name"]) == normalize_text(
        golden["product_name"]
    )
    fields["flash_point"] = ranges_match(output["flash_point"], golden["flash_point"])
    for list_field in ("components", "hazard_statements"):
        counts = items[list_field]
        fields[list_field] = counts.matched == counts.predicted == counts.golden
    return DocScore(fields, items)


def percentile(values: list[float], p: float) -> float:
    """Nearest rank: the smallest observed value with at least p % of values at or below it."""
    ordered = sorted(values)
    rank = max(math.ceil(p / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_cached(call: dict) -> bool:
    """At least half of the prompt came from Ollama's cache; a shared system-prompt prefix
    alone (about 114 tokens) does not count (docs/METRICS.md)."""
    cached = (call.get("details") or {}).get("prompt_cached_tokens") or 0
    prompt = call.get("prompt_tokens") or 0
    return prompt > 0 and cached >= 0.5 * prompt


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """run.json and the result records of the run's documents that have one, in run order."""
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    records = []
    for doc_id in summary["docs"]:
        path = run_dir / f"{doc_id}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return summary, records


def run_metrics(run_dir: Path, config: Config, golden_dir: Path = GOLDEN) -> dict:
    """All step 2 metrics of one run, as one table row."""
    summary, records = load_run(run_dir)
    model_key = summary["model"]["key"]

    per_field = dict.fromkeys(FIELDS, 0)
    items = {name: ItemCounts() for name in ITEM_SCORES}
    golden_docs = valid_golden_docs = matched_valid = 0
    for record in records:
        golden = load_golden(record["doc_id"], golden_dir)
        if golden is None:
            continue
        golden_docs += 1
        valid = record["status"] == "valid"
        score = score_document(record["output"] if valid else None, golden)
        for f, ok in score.fields.items():
            per_field[f] += ok
        for name, counts in score.items.items():
            items[name].add(counts)
        if valid:
            valid_golden_docs += 1
            matched_valid += sum(score.fields.values())

    calls = [r["call"] for r in records if r["status"] != "error" and r["call"]]
    latencies = [c["latency_s"] for c in calls]
    model = config.models.get(model_key)
    priced = None
    if isinstance(model, OllamaModel):
        priced = 0.0
    elif model is not None:
        priced = mean(
            [
                (
                    c["prompt_tokens"] * model.price_in_usd_per_mtok
                    + c["completion_tokens"] * model.price_out_usd_per_mtok
                )
                / 1e6
                for c in calls
                if c["prompt_tokens"] is not None and c["completion_tokens"] is not None
            ]
        )
    reported = mean([c["cost_usd"] for c in calls if c["cost_usd"] is not None])

    return {
        "run_id": summary["run_id"],
        "model": model_key,
        "client": summary["model"]["client"],
        "sampling": ((summary["params"] or {}).get("sampling") or {}).get("name"),
        "prompt": summary["prompt"]["version"],
        "docs": len(records),
        "valid": sum(r["status"] == "valid" for r in records),
        "golden_docs": golden_docs,
        "fields_matched": sum(per_field.values()),
        "valid_golden_docs": valid_golden_docs,
        "fields_matched_valid": matched_valid,
        "per_field": per_field,
        "items": items,
        "latency": {
            "n": len(latencies),
            "p50": percentile(latencies, 50) if latencies else None,
            "p95": percentile(latencies, 95) if latencies else None,
            "cached": sum(is_cached(c) for c in calls),
        },
        "cost_per_1000": {
            "priced": None if priced is None else 1000 * priced,
            "reported": None if reported is None else 1000 * reported,
            "price_checked": getattr(model, "price_checked", None),
        },
    }


def ratio(matched: int, total: int) -> str:
    return f"{matched}/{total} ({100 * matched / total:.1f}%)" if total else "-"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def sampling_label(row: dict) -> str:
    if row["client"] != "ollama":
        return "-"
    return row["sampling"] or "not logged"  # runs before S1-10 did not log it


def format_tables(rows: list[dict]) -> str:
    """The main metrics table, the per-field table and footnotes, as Markdown."""
    main = []
    for row in rows:
        latency, cost = row["latency"], row["cost_per_1000"]
        local = row["client"] == "ollama"
        main.append(
            [
                row["run_id"],
                row["model"],
                sampling_label(row),
                row["prompt"],
                ratio(row["valid"], row["docs"]),
                "n/a (S1-15)",
                str(row["golden_docs"]),
                ratio(row["fields_matched"], len(FIELDS) * row["golden_docs"]),
                ratio(row["fields_matched_valid"], len(FIELDS) * row["valid_golden_docs"]),
                f"{latency['p50']:.1f} / {latency['p95']:.1f} (n={latency['n']})"
                if latency["n"]
                else "-",
                str(latency["cached"]) if local else "-",
                "0 (local)" if local else money(cost["priced"]),
                "-" if local else money(cost["reported"]),
            ]
        )
    fields = []
    for row in rows:
        golden_docs = row["golden_docs"]
        fields.append(
            [row["run_id"]]
            + [f"{row['per_field'][f]}/{golden_docs}" if golden_docs else "-" for f in FIELDS]
            + [f"{row['items'][name].f1:.2f}" if golden_docs else "-" for name in ITEM_SCORES]
        )
    checked = sorted(
        {
            f"{r['model']} {r['cost_per_1000']['price_checked']}"
            for r in rows
            if r["cost_per_1000"]["price_checked"]
        }
    )
    notes = [
        "Rules: docs/METRICS.md.",
        "Valid (1st): schema-valid first response over all documents in the run. "
        "After repair: no repair step yet (S1-15).",
        "Field acc.: matched fields / (9 x golden documents); a failed document misses all 9. "
        "Valid only: the same over golden documents with a valid output.",
        "Latency: seconds, the successful attempt of each call, nearest-rank percentiles. "
        "OpenRouter latency includes routing and depends on the upstream provider. "
        "Cached: local calls where Ollama took at least half of the prompt from its cache "
        "(faster than a cold call).",
        "$ / 1000 docs: mean tokens x price in config.toml (checked: "
        + (", ".join(checked) or "-")
        + "); reported: OpenRouter's own cost. Local: no API cost; hardware and time not counted.",
        "Per-field table: documents where the field matches / golden documents; "
        "list F1 is item-level over all golden documents; hazard_codes F1 ignores the text.",
    ]
    return "\n\n".join(
        [
            markdown_table(
                [
                    "Run",
                    "Model",
                    "Sampling",
                    "Prompt",
                    "Valid (1st)",
                    "Valid after repair",
                    "Golden docs",
                    "Field acc.",
                    "Field acc. (valid only)",
                    "Latency p50 / p95 s",
                    "Cached",
                    "$ / 1000 docs",
                    "$ / 1000 (reported)",
                ],
                main,
            ),
            markdown_table(["Run", *FIELDS, *(f"{name} F1" for name in ITEM_SCORES)], fields),
            "\n".join(f"- {note}" for note in notes),
        ]
    )


def money(value: float | None) -> str:
    return "-" if value is None else f"${value:.2f}"
