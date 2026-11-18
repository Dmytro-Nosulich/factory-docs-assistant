"""PDF to plain text with page markers.

Text is kept as the extractor returns it: layout noise is part of the "dirty texts" test.
Every page gets a `[[page N]]` marker (1-based), even pages without a text layer, so the
page count stays exact and stage 2 can cite pages.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pypdf import PageObject, PdfReader

PAGE_MARKER = re.compile(r"^\[\[page \d+\]\]$", re.MULTILINE)


@dataclass(frozen=True)
class StyledLine:
    """One layout-mode line (column spacing kept); size is None when no style was matched."""

    text: str
    size: float | None = None
    bold: bool = False


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def page_styles(page: PageObject) -> dict[str, tuple[float, bool]]:
    """Map each visual line's text without whitespace to its main font size and boldness."""
    fragments: dict[int, list[tuple[str, float, bool]]] = defaultdict(list)

    def visitor(text, cm, tm, font_dict, font_size) -> None:
        if not text.strip():
            return
        size = round(abs(font_size * (tm[3] or 1)) * abs(cm[3] or 1), 1)
        font_name = str((font_dict or {}).get("/BaseFont", "")).lower()
        y = round(tm[5] * (cm[3] or 1) + cm[5])
        fragments[y].append((text, size, "bold" in font_name or "black" in font_name))

    page.extract_text(visitor_text=visitor)
    styles = {}
    for parts in fragments.values():
        sizes: Counter[float] = Counter()
        bold_chars = 0
        for text, size, bold in parts:
            sizes[size] += len(text)
            bold_chars += len(text) if bold else 0
        total = sum(sizes.values())
        styles[squash("".join(p[0] for p in parts))] = (
            sizes.most_common(1)[0][0],
            bold_chars * 2 > total,
        )
    return styles


def extract_styled_lines(page: PageObject) -> list[StyledLine]:
    """Return the page's layout-mode lines with font style where it can be matched (S2-03)."""
    styles = page_styles(page)
    lines = []
    for text in page.extract_text(extraction_mode="layout").splitlines():
        key = squash(text)
        style = styles.get(key)
        if style is None and key:
            style = next((v for k, v in styles.items() if key in k), None)
        lines.append(StyledLine(text.rstrip(), *(style or (None, False))))
    return lines


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
