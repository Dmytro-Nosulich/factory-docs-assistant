"""PDF to plain text with page markers.

Text is kept as the extractor returns it: layout noise is part of the "dirty texts" test.
Every page gets a `[[page N]]` marker (1-based), even pages without a text layer, so the
page count stays exact and stage 2 can cite pages.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

PAGE_MARKER = re.compile(r"^\[\[page \d+\]\]$", re.MULTILINE)


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


@dataclass(frozen=True)
class CappedText:
    text: str
    pages_sent: int
    pages_total: int

    @property
    def truncated(self) -> bool:
        return self.pages_sent < self.pages_total


def cap_pages(text: str, max_tokens: int) -> CappedText:
    """Keep whole pages of marked text while the estimate stays within max_tokens (S1-03).

    Page 1 is always kept, even if it alone is over the cap.
    """
    starts = [0] + [m.start() for m in PAGE_MARKER.finditer(text) if m.start() > 0]
    pages = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)], strict=True)]
    kept = 1
    while kept < len(pages) and estimate_tokens("".join(pages[: kept + 1])) <= max_tokens:
        kept += 1
    return CappedText("".join(pages[:kept]).rstrip() + "\n", kept, len(pages))
