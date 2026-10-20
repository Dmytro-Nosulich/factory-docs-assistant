"""PDF to plain text with page markers.

Text is kept as the extractor returns it: layout noise is part of the "dirty texts" test.
Every page gets a `[[page N]]` marker (1-based), even pages without a text layer, so the
page count stays exact and stage 2 can cite pages.
"""

from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: Path) -> list[str]:
    """Return one string per page; a page without a text layer gives ""."""
    reader = PdfReader(pdf_path)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def to_marked_text(pages: list[str]) -> str:
    """Join pages as `[[page 1]]\\n<text>\\n\\n[[page 2]]\\n<text>...`."""
    blocks = [f"[[page {n}]]\n{text}".rstrip() for n, text in enumerate(pages, start=1)]
    return "\n\n".join(blocks) + "\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 characters per token). Undercounts Cyrillic text."""
    return len(text) // 4
