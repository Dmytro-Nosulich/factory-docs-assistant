import csv
import re
from pathlib import Path

from factory_docs.pdf_text import cap_pages, extract_pages, to_marked_text

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


def test_cap_pages_keeps_everything_under_the_cap():
    text = to_marked_text(["a" * 40, "b" * 40])
    capped = cap_pages(text, max_tokens=1000)
    assert (capped.text, capped.pages_sent, capped.pages_total) == (text, 2, 2)
    assert not capped.truncated


def test_cap_pages_cuts_at_a_page_boundary():
    text = to_marked_text(["a" * 400, "b" * 400, "c" * 400])  # ~105 tokens per page
    capped = cap_pages(text, max_tokens=220)
    assert (capped.pages_sent, capped.pages_total, capped.truncated) == (2, 3, True)
    assert capped.text == to_marked_text(["a" * 400, "b" * 400])


def test_cap_pages_always_keeps_page_one():
    capped = cap_pages(to_marked_text(["a" * 4000, "b"]), max_tokens=10)
    assert (capped.pages_sent, capped.pages_total) == (1, 2)


def test_long_real_document_is_truncated():
    text = (TEXTS / "D003.txt").read_text(encoding="utf-8")
    capped = cap_pages(text, max_tokens=20000)
    assert capped.truncated
    assert MARKER.findall(capped.text) == [str(n) for n in range(1, capped.pages_sent + 1)]
