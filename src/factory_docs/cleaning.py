"""Clean parsed documents and find exact duplicates (S2-04).

Every rule here was written against what the corpus really contains, and most of them guard a trap:
a soft hyphen in D052 is a real hyphen inside a CAS number, and a Symbol-font character in D042 is
a real `≤`. Whatever is removed is recorded, so the report can show it.
"""

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from factory_docs.parsing.model import (
    Block,
    Code,
    Heading,
    ListBlock,
    Paragraph,
    ParsedDocument,
    Table,
    assign_section_paths,
)

MIN_REPEAT_PAGES = 3  # shorter documents give too little evidence that a line repeats
REPEAT_SHARE = 0.5  # a line must be on at least this share of the pages
EDGE_LINES = 3  # "at the page edge" means within this many lines of the top or the bottom
EDGE_SHARE = 0.8  # real content repeats too, but not at the page edge
MIN_KEY_CHARS = 4
MIN_HEADER_WORDS = 2
SAME_PAGE_REPEATS = 3  # page furniture appears once per page; D016 repeats its standards list

# Languages whose line-end hyphens were measured to be wrapped syllables. Elsewhere an unknown word
# keeps its hyphen, because `at-industrial` joined into one word is worse than a stray hyphen.
JOIN_BY_DEFAULT = {"uk", "de"}
SUSPENDED_ANY = {"and", "or"}  # English fragments appear in documents of other languages (D003)
SUSPENDED_BY_LANGUAGE = {
    "de": {"und", "oder", "bzw"},
    "sv": {"och", "eller"},
    "pl": {"lub", "oraz", "albo"},
    "nl": {"en", "of"},
    "fr": {"et", "ou"},
    "es": {"y", "o"},
    "it": {"e", "o", "ed"},
    "uk": {"або"},  # not `та` or `і`: they are also common word endings (`робо-та`)
}

LIGATURES = str.maketrans(
    {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
)
SYMBOL_FONT = str.maketrans(
    {"\uf0a3": "≤", "\uf0b3": "≥", "\uf0fc": "✓", "\uf0df": "←", "\uf0e0": "→"}
)
BROKEN_GLYPHS = {"Ɵ": "ti", "Ʃ": "tt", "Ō": "ft", "\x00": "ti"}

BROKEN_GLYPH = re.compile(
    r"(?<=[a-zƟƩŌ])[ƟƩŌ](?: (?=[a-zƟƩŌ]))?"  # after a lowercase letter, often with a stray space
    r"|(?<![^\W\d_])Ɵ ?(?=[a-zƟ])"  # `Ɵme`; only this glyph, because `Ō` starts real names
    r"|(?<=[a-z])\x00(?=[a-z])"
)
SPACED_SOFT_HYPHEN = re.compile(r" \u00ad(?=[^\W\d_])")  # `1,2 -diol`; `: -40 °C` keeps its space
DELETED_CHARACTER = re.compile(r"[\ue000-\uf8ff]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINE_END_HYPHEN = re.compile(r"([^\W_]+(?:['’][^\W_]+)*|\))-\n(?=([^\W_]+))")  # пов'я-, 3(2H)-
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
DOT_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")
PAGE_NUMBER = re.compile(
    r"^\W*(?:[^\W\d_]+\W*)?\d+\s*(?:/|of|von|av|z|de|di|van|з|from)\s*\d+\W*$", re.IGNORECASE
)
CONTINUATION = re.compile(
    r"fortsetzung|continued|continuation|contd\.|suite de la page|vervolg|continuaci|segue da|"
    r"ciąg dalszy|продовження",
    re.IGNORECASE,
)
INLINE_CODE = re.compile(r"(`[^`]*`)")
MARKDOWN_SYNTAX = (
    re.compile(r"\[([^\]]+)\]\[[^\]]*\]"),  # [text][reference]
    re.compile(r"\[([^\]]+)\]\([^)]*\)"),  # [text](url)
    re.compile(r"\*\*([^*]+)\*\*"),  # **bold**
)


class CleanedDocument(ParsedDocument):
    content_hash: str


@dataclass(frozen=True)
class Removal:
    rule: str
    text: str
    page: int | None = None


@dataclass
class Prepared:
    """A document after the rules that need no knowledge of the other documents."""

    source: ParsedDocument
    language: str
    blocks: list[Block]
    removals: list[Removal] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class CleanResult:
    document: CleanedDocument
    removals: list[Removal]
    counts: Counter[str]
    blocks_before: int
    chars_before: int


def block_texts(block: Block) -> list[str]:
    """Return every piece of text a block holds."""
    if isinstance(block, Heading | Paragraph | Code):
        return [block.text]
    if isinstance(block, ListBlock):
        return [item.text for item in block.items]
    cells = [cell for row in block.rows for cell in row if cell]
    return cells + ([block.caption] if block.caption else [])


def character_count(blocks: Iterable[Block]) -> int:
    return sum(len(text) for block in blocks for text in block_texts(block))


def map_text(block: Block, transform: Callable[[str], str]) -> None:
    """Apply a text rule to a block. Code stays exactly as the source wrote it."""
    if isinstance(block, Heading | Paragraph):
        block.text = transform(block.text)
    elif isinstance(block, ListBlock):
        for item in block.items:
            item.text = transform(item.text)
    elif isinstance(block, Table):
        block.rows = [[transform(cell) for cell in row] for row in block.rows]
        if block.caption:
            block.caption = transform(block.caption)


def line_key(text: str) -> str:
    """Compare lines without digits, so `12 Endress+Hauser` matches `Endress+Hauser 13`."""
    return " ".join(re.sub(r"\d+", " ", text).split()).casefold()


def find_repeated_lines(document: ParsedDocument) -> set[tuple[int, int]]:
    """Return (block index, line index) of repeated page-edge lines, without their first copy."""
    page_count = document.page_count or 0
    if document.format != "pdf" or page_count < MIN_REPEAT_PAGES:
        return set()
    pages: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for block_index, block in enumerate(document.blocks):
        if block.zone == "main" and block.page and isinstance(block, Heading | Paragraph):
            for line_index, line in enumerate(block.text.split("\n")):
                pages[block.page].append((block_index, line_index, line_key(line)))

    seen_on: dict[str, set[int]] = defaultdict(set)
    edge_on: dict[str, set[int]] = defaultdict(set)
    at_edge: dict[str, list[tuple[int, int]]] = defaultdict(list)
    first_copy: dict[str, tuple[int, int]] = {}
    for page, lines in pages.items():
        for position, (block_index, line_index, key) in enumerate(lines):
            if len(key) < MIN_KEY_CHARS:
                continue
            seen_on[key].add(page)
            first_copy.setdefault(key, (block_index, line_index))
            if position < EDGE_LINES or position >= len(lines) - EDGE_LINES:
                edge_on[key].add(page)
                at_edge[key].append((block_index, line_index))

    needed = max(MIN_REPEAT_PAGES, page_count * REPEAT_SHARE)
    return {
        spot
        for key, on_pages in seen_on.items()
        if len(on_pages) >= needed and len(edge_on[key]) / len(on_pages) >= EDGE_SHARE
        for spot in at_edge[key]
        if spot != first_copy[key]
    }


def is_worth_keeping(header_line: str) -> bool:
    """A kept header line must say something: no page numbers or continuation notices."""
    return (
        PAGE_NUMBER.match(header_line.strip()) is None
        and CONTINUATION.search(header_line) is None
        and len(re.findall(r"[^\W\d_]{2,}", header_line)) >= MIN_HEADER_WORDS
    )


def footers_that_are_content(document: ParsedDocument) -> set[str]:
    """Return footer lines repeated within one page, which a real page footer never is."""
    per_page = Counter(
        (block.page, line_key(" ".join(block_texts(block))))
        for block in document.blocks
        if block.zone == "footer" and re.search(r"[^\W\d_]", " ".join(block_texts(block)))
    )
    return {key for (_, key), times in per_page.items() if times >= SAME_PAGE_REPEATS}


def remove_furniture(document: ParsedDocument, prepared: Prepared) -> None:
    """Apply the zone, repeated-line and boilerplate rules."""
    repeated = find_repeated_lines(document)
    content_footers = footers_that_are_content(document)
    seen_headers: set[str] = set()

    def remove(rule: str, text: str, page: int | None) -> None:
        prepared.removals.append(Removal(rule, text, page))
        prepared.counts[rule] += 1

    for index, block in enumerate(document.blocks):
        if block.zone != "main":
            text = " ".join(block_texts(block))
            if block.zone == "footer" and line_key(text) in content_footers:
                block.zone = "main"
                prepared.counts["footer_restored"] += 1
                prepared.blocks.append(block)
                continue
            first_copy = False
            if document.format == "pdf" and block.zone == "header":
                first_copy = line_key(text) not in seen_headers
                seen_headers.add(line_key(text))
            if first_copy and is_worth_keeping(text):
                block.zone = "main"
                prepared.counts["header_kept"] += 1
                prepared.blocks.append(block)
            else:
                remove(f"zone_{block.zone}", text, block.page)
            continue
        if isinstance(block, Heading | Paragraph):
            following = document.blocks[index + 1] if index + 1 < len(document.blocks) else None
            title = block.text.strip().casefold()
            if title in ("contents", "table of contents") and following and following.zone == "toc":
                remove("contents_title", block.text, block.page)
                continue
            lines = []
            for line_index, line in enumerate(block.text.split("\n")):
                if (index, line_index) in repeated:
                    remove("repeated_line", line, block.page)
                elif DOT_LEADER.search(line):
                    remove("toc_line", line, block.page)
                elif line.strip().casefold() == "back to top":
                    remove("back_to_top", line, block.page)
                else:
                    lines.append(line)
            if not lines:
                continue
            block.text = "\n".join(lines)
        prepared.blocks.append(block)


def normalize_text(
    text: str,
    counts: Counter[str],
    removals: list[Removal] | None = None,
    page: int | None = None,
) -> str:
    """Repair the encoding. NFKC is not used: it would turn `m³` into `m3` and `µ` into `μ`."""
    normalized = unicodedata.normalize("NFC", text)
    counts["nfc"] += normalized != text

    counts["ligature"] += sum(normalized.count(ligature) for ligature in "ﬀﬁﬂﬃﬄﬅﬆ")
    normalized = normalized.translate(LIGATURES)

    counts["soft_hyphen"] += normalized.count("\u00ad")
    normalized = SPACED_SOFT_HYPHEN.sub("-", normalized).replace("\u00ad", "-")

    normalized, broken = BROKEN_GLYPH.subn(
        lambda match: BROKEN_GLYPHS[match.group().strip()], normalized
    )
    counts["broken_glyph"] += broken

    fixed = normalized.translate(SYMBOL_FONT)
    counts["symbol_glyph"] += sum(1 for a, b in zip(normalized, fixed, strict=True) if a != b)
    normalized = fixed

    deleted = DELETED_CHARACTER.findall(normalized)
    if not deleted:
        return normalized
    for character in deleted:
        rule = "control_char" if ord(character) < 0xE000 else "icon_glyph"
        counts[rule] += 1
        if removals is not None:
            removals.append(Removal(rule, f"U+{ord(character):04X}", page))
    lines = (" ".join(line.split()) for line in DELETED_CHARACTER.sub("", normalized).split("\n"))
    return "\n".join(line for line in lines if line)


def strip_markdown(text: str, counts: Counter[str]) -> str:
    """Turn links and bold into plain text, leaving inline code such as `a[0][1]` alone."""
    parts = INLINE_CODE.split(text)
    for index in range(0, len(parts), 2):  # odd positions hold the inline code
        for pattern in MARKDOWN_SYNTAX:
            parts[index], found = pattern.subn(r"\1", parts[index])
            counts["markdown_syntax"] += found
    return "".join(parts)


def prepare(document: ParsedDocument, language: str) -> Prepared:
    """Run every rule that works on one document alone."""
    source = document.model_copy(deep=True)
    prepared = Prepared(source=document, language=language, blocks=[])
    remove_furniture(source, prepared)

    def clean_text(page: int | None) -> Callable[[str], str]:
        def transform(text: str) -> str:
            normalized = normalize_text(text, prepared.counts, prepared.removals, page)
            is_markdown = document.format == "md"
            return strip_markdown(normalized, prepared.counts) if is_markdown else normalized

        return transform

    for block in prepared.blocks:
        map_text(block, clean_text(block.page))
    return prepared


def add_vocabulary(vocabulary: Counter[str], blocks: Iterable[Block]) -> None:
    """Count the words of a text. Words wrapped at a line end are not evidence, so they are cut."""
    for block in blocks:
        if isinstance(block, Code):
            continue
        for text in block_texts(block):
            vocabulary.update(word.casefold() for word in WORD.findall(text.replace("-\n", "\0")))


def hyphen_decision(left: str, right: str, language: str, vocabulary: Counter[str]) -> str:
    """Decide what a line-end hyphen becomes: `join`, `keep` or `suspended`."""
    if not left[-1].isalpha() or right[0].isdigit() or right[0].isupper():
        return "keep"
    if len(left) == 1 or left.isupper():
        return "keep"  # n-octanol, STOT-RE
    if right.casefold() in SUSPENDED_ANY | SUSPENDED_BY_LANGUAGE.get(language, set()):
        return "suspended"  # `Haut- und Augenkontakt`: the word after `und` completes it
    hyphenated = vocabulary[f"{left}-{right}".casefold()] > 0
    joined = vocabulary[f"{left}{right}".casefold()] > 0
    if hyphenated != joined:
        return "keep" if hyphenated else "join"
    return "join" if language in JOIN_BY_DEFAULT else "keep"


def repair_hyphens(text: str, language: str, vocabulary: Counter[str], counts: Counter[str]) -> str:
    """Rejoin words that a PDF line break split. Other line breaks stay as they are."""

    def repair(match: re.Match[str]) -> str:
        left, right = match.groups()
        decision = hyphen_decision(left, right, language, vocabulary)
        counts[f"hyphen_{decision}"] += 1
        return left + {"join": "", "keep": "-", "suspended": "- "}[decision]

    return LINE_END_HYPHEN.sub(repair, text)


def content_hash(blocks: Iterable[Block]) -> str:
    """Hash the cleaned text only, so layout and page numbers cannot hide a duplicate."""
    text = " ".join(piece for block in blocks for piece in block_texts(block))
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def has_text(block: Block) -> bool:
    if isinstance(block, ListBlock):
        block.items = [item for item in block.items if item.text.strip()]
    return any(text.strip() for text in block_texts(block))


def finish(prepared: Prepared, vocabulary: Counter[str]) -> CleanResult:
    """Repair hyphens, drop what became empty, and rebuild the section paths."""
    for block in prepared.blocks:
        if isinstance(block, Paragraph):
            block.text = repair_hyphens(block.text, prepared.language, vocabulary, prepared.counts)
    blocks = [block for block in prepared.blocks if has_text(block)]
    assign_section_paths(blocks)
    source = prepared.source
    document = CleanedDocument(
        doc_id=source.doc_id,
        format=source.format,
        blocks=blocks,
        page_count=source.page_count,
        pages_without_text=source.pages_without_text,
        warnings=source.warnings,
        content_hash=content_hash(blocks),
    )
    return CleanResult(
        document=document,
        removals=prepared.removals,
        counts=prepared.counts,
        blocks_before=len(source.blocks),
        chars_before=character_count(source.blocks),
    )


def clean_corpus(documents: Iterable[tuple[ParsedDocument, str]]) -> list[CleanResult]:
    """Clean (document, language) pairs. Hyphen repair uses the words of the same language."""
    prepared = [prepare(document, language) for document, language in documents]
    vocabularies: dict[str, Counter[str]] = defaultdict(Counter)
    for item in prepared:
        add_vocabulary(vocabularies[item.language], item.blocks)
    return [finish(item, vocabularies[item.language]) for item in prepared]


def find_duplicates(hashes: dict[str, str], supersedes: dict[str, str]) -> dict[str, str]:
    """Return {duplicate ID: kept ID}. The lowest ID stays; linked versions are never merged."""
    linked = {frozenset(pair) for pair in supersedes.items()}
    kept: dict[str, list[str]] = defaultdict(list)
    duplicates: dict[str, str] = {}
    for doc_id in sorted(hashes):
        group = kept[hashes[doc_id]]
        if group and not any(frozenset((doc_id, other)) in linked for other in group):
            duplicates[doc_id] = group[0]
        else:
            group.append(doc_id)
    return duplicates


def page_ranges(pages: Iterable[int]) -> str:
    """Write pages compactly: 1, 2, 3 and 7 become `1-3 7`."""
    ranges: list[list[int]] = []
    for page in sorted(set(pages)):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1][1] = page
        else:
            ranges.append([page, page])
    return " ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def removal_rows(doc_id: str, removals: Iterable[Removal]) -> list[dict[str, str | int]]:
    """Group removals for the report, so a header repeated on 45 pages is one row."""
    groups: dict[tuple[str, str], list[Removal]] = defaultdict(list)
    for removal in removals:
        exact = removal.rule in ("icon_glyph", "control_char")
        groups[(removal.rule, removal.text if exact else line_key(removal.text))].append(removal)
    return [
        {
            "id": doc_id,
            "rule": rule,
            "pages": page_ranges(item.page for item in items if item.page is not None),
            "times": len(items),
            "text": " ".join(items[0].text.split()),
        }
        for (rule, _), items in sorted(groups.items(), key=lambda group: group[0])
    ]
