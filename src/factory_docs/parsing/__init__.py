"""Parse any corpus document into one `ParsedDocument` structure (S2-03)."""

from pathlib import Path

from factory_docs.parsing.model import ParsedDocument
from factory_docs.parsing.parse_docx import parse_docx
from factory_docs.parsing.parse_html import parse_html
from factory_docs.parsing.parse_markdown import parse_markdown
from factory_docs.parsing.parse_pdf import parse_pdf

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".md": parse_markdown,
}


def parse_document(doc_id: str, path: Path) -> ParsedDocument:
    """Pick the parser from the file type."""
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"{doc_id}: unsupported file type {path.suffix!r}")
    return parser(doc_id, path)
