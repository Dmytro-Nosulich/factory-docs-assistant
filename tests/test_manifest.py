import csv
import re
from datetime import date
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA / "manifest.csv"
RAW = DATA / "raw"

REQUIRED = [
    "id",
    "title",
    "source_url",
    "license_or_terms",
    "language",
    "doc_type",
    "downloaded_at",
]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_required_columns_present_and_filled(rows):
    missing = [c for c in REQUIRED if c not in rows[0]]
    assert not missing, f"missing columns: {missing}"
    empty = [(r["id"], c) for r in rows for c in REQUIRED if not r[c].strip()]
    assert not empty, f"empty required fields: {empty}"


def test_ids_unique_and_well_formed(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    bad = [i for i in ids if not re.fullmatch(r"D\d{3}", i)]
    assert not bad, f"malformed ids: {bad}"


def test_every_row_has_pdf_and_every_pdf_has_row(rows):
    listed = {r["id"] for r in rows}
    on_disk = {p.stem for p in RAW.glob("*.pdf")}
    assert not listed - on_disk, f"listed but no PDF: {sorted(listed - on_disk)}"
    assert not on_disk - listed, f"PDF not in manifest: {sorted(on_disk - listed)}"


def test_pdfs_are_real_pdfs(rows):
    not_pdf = []
    for r in rows:
        with (RAW / f"{r['id']}.pdf").open("rb") as f:
            if f.read(5) != b"%PDF-":
                not_pdf.append(r["id"])
    assert not not_pdf, f"files without a PDF header: {not_pdf}"


def test_field_formats(rows):
    for r in rows:
        date.fromisoformat(r["downloaded_at"])
        assert r["source_url"].startswith("https://"), r["id"]
        assert re.fullmatch(r"[a-z]{2,3}(/[a-z]{2,3})*", r["language"]), r["id"]


def test_stage1_is_at_least_30_sds_documents(rows):
    # Stage 1 is SDS only (decision 10); the course asks for at least 30 texts (ISSUES.md I-002).
    stage1 = [r for r in rows if "1" in r["stage"].split(",")]
    not_sds = [r["id"] for r in stage1 if r["doc_type"] != "sds"]
    assert not not_sds, f"stage 1 must be SDS only: {not_sds}"
    assert len(stage1) >= 30, f"only {len(stage1)} stage 1 documents"
