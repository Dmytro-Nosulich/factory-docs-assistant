"""Checks for hand-labeled golden-set files (data/golden/<id>.json), S1-05."""

import json
from pathlib import Path

from pydantic import ValidationError

from factory_docs.config import ROOT
from factory_docs.schema import SafetyDataSheet

GOLDEN = ROOT / "data" / "golden"
PLACEHOLDER = "TODO"


def load_golden(doc_id: str, golden_dir: Path = GOLDEN) -> dict | None:
    """The golden label as schema-normalized JSON (as model outputs are saved); None if absent."""
    path = golden_dir / f"{doc_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SafetyDataSheet.model_validate(data).model_dump(mode="json")


def find_placeholders(value: object, path: str = "") -> list[str]:
    """Paths of values still set to the skeleton placeholder, e.g. 'components[0].name'."""
    if value == PLACEHOLDER:
        return [path]
    if isinstance(value, dict):
        items = [(f"{path}.{k}" if path else k, v) for k, v in value.items()]
    elif isinstance(value, list):
        items = [(f"{path}[{i}]", v) for i, v in enumerate(value)]
    else:
        return []
    return [p for item_path, v in items for p in find_placeholders(v, item_path)]


def format_loc(loc: tuple[str | int, ...]) -> str:
    return "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc).lstrip(".")


def check_golden(path: Path) -> list[str]:
    """Problems in one golden file; an empty list means it is complete and schema-valid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}"]

    problems = [f"{p}: still {PLACEHOLDER}" for p in find_placeholders(data)]
    try:
        SafetyDataSheet.model_validate(data)
    except ValidationError as e:
        # A placeholder is already reported above; skip the schema error it causes.
        problems += [
            f"{format_loc(err['loc']) or '(top level)'}: {err['msg']}"
            for err in e.errors()
            if err.get("input") != PLACEHOLDER
        ]
    return problems
