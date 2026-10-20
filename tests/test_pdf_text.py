import csv
import re
from pathlib import Path

from factory_docs.pdf_text import extract_pages, to_marked_text

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA / "manifest.csv"
TEXTS = DATA / "texts"
MARKER = re.compile(r"^\[\[page (\d+)\]\]$", re.MULTILINE)


def test_to_marked_text_numbers_pages_and_keeps_empty_ones():
    text = to_marked_text(["first", "", "third"])
    assert text == "[[page 1]]\nfirst\n\n[[page 2]]\n\n[[page 3]]\nthird\n"
    assert MARKER.findall(text) == ["1", "2", "3"]


def test_extract_pages_from_real_pdf():
    pages = extract_pages(DATA / "raw" / "D034.pdf")
    assert len(pages) == 6
    assert all(pages), "every page of D034 has a text layer"


def test_every_stage1_document_has_text_with_all_pages():
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        stage1 = [r for r in csv.DictReader(f) if "1" in r["stage"].split(",")]
    assert len(stage1) >= 15
    problems = []
    for r in stage1:
        path = TEXTS / f"{r['id']}.txt"
        if not path.exists():
            problems.append(f"{r['id']}: no text file")
            continue
        pages = [int(n) for n in MARKER.findall(path.read_text(encoding="utf-8"))]
        if pages != list(range(1, int(r["pages"]) + 1)):
            problems.append(f"{r['id']}: markers {len(pages)}, manifest pages {r['pages']}")
    assert not problems, problems
