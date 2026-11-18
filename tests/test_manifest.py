import csv
import re
import zipfile
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
    "format",
    "in_rag_index",
]
FORMATS = {"pdf", "html", "docx", "md"}
EXCLUDED_FROM_RAG = {"D023", "D024", "D027", "D028"}


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


def raw_path(row: dict[str, str]) -> Path:
    return RAW / f"{row['id']}.{row['format']}"


def test_every_format_is_known_and_present(rows):
    assert {r["format"] for r in rows} == FORMATS


def test_every_row_has_one_raw_file_and_every_raw_file_has_a_row(rows):
    listed = {raw_path(r).name for r in rows}
    on_disk = {p.name for p in RAW.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert not listed - on_disk, f"listed but no file: {sorted(listed - on_disk)}"
    assert not on_disk - listed, f"file not in manifest: {sorted(on_disk - listed)}"


def has_expected_content(row: dict[str, str]) -> bool:
    path = raw_path(row)
    if row["format"] == "pdf":
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    if row["format"] == "docx":
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    text = path.read_text(encoding="utf-8")
    if row["format"] == "html":
        return re.search(r"<(p|div|h[1-6])[\s>]", text) is not None
    return re.search(r"^#{1,6} \S", text, re.MULTILINE) is not None


def test_raw_files_have_the_content_of_their_format(rows):
    wrong = [r["id"] for r in rows if not has_expected_content(r)]
    assert not wrong, f"content does not match the format: {wrong}"


def test_new_versions_point_to_an_older_document_of_the_same_kind(rows):
    by_id = {r["id"]: r for r in rows}
    pairs = [r for r in rows if r["supersedes"]]
    assert len(pairs) >= 3, f"only {len(pairs)} old/new version pairs"
    for new in pairs:
        old = by_id.get(new["supersedes"])
        assert old is not None, f"{new['id']} supersedes an unknown document"
        assert old["id"] != new["id"], new["id"]
        assert old["format"] == new["format"], new["id"]
        assert old["doc_type"] == new["doc_type"], new["id"]
        # Both versions stay searchable: the outdated-version trap questions need them.
        assert {old["in_rag_index"], new["in_rag_index"]} == {"yes"}, new["id"]


def test_rag_scope(rows):
    # Decision 14: the four 540-1028 page documents stay out of the index.
    assert {r["in_rag_index"] for r in rows} == {"yes", "no"}
    assert {r["id"] for r in rows if r["in_rag_index"] == "no"} == EXCLUDED_FROM_RAG
    in_scope = [r for r in rows if r["in_rag_index"] == "yes"]
    assert len(in_scope) >= 30, f"only {len(in_scope)} documents in scope"
    assert {r["format"] for r in in_scope} == FORMATS


def test_field_formats(rows):
    for r in rows:
        date.fromisoformat(r["downloaded_at"])
        assert r["source_url"].startswith("https://"), r["id"]
        assert re.fullmatch(r"[a-z]{2,3}(/[a-z]{2,3})*", r["language"]), r["id"]


def test_stage1_is_at_least_30_sds_documents(rows):
    # Stage 1 is SDS only; the course asks for at least 30 texts (ISSUES.md I-002).
    stage1 = [r for r in rows if "1" in r["stage"].split(",")]
    not_sds = [r["id"] for r in stage1 if r["doc_type"] != "sds"]
    assert not not_sds, f"stage 1 must be SDS only: {not_sds}"
    assert len(stage1) >= 30, f"only {len(stage1)} stage 1 documents"
