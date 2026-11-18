import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from factory_docs.cleaning import (
    CleanResult,
    Removal,
    add_vocabulary,
    clean_corpus,
    content_hash,
    find_duplicates,
    find_repeated_lines,
    hyphen_decision,
    is_worth_keeping,
    normalize_text,
    page_ranges,
    removal_rows,
    repair_hyphens,
    strip_markdown,
)
from factory_docs.parsing.model import (
    Block,
    Code,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    ParsedDocument,
    Table,
    to_review_json,
)

DATA = Path(__file__).resolve().parent.parent / "data"
PARSED = DATA / "parsed"
CLEANED = DATA / "cleaned"


def document(blocks: list[Block], fmt: str = "pdf", pages: int | None = 6) -> ParsedDocument:
    return ParsedDocument(doc_id="T001", format=fmt, blocks=blocks, page_count=pages)


def clean_one(blocks: list[Block], fmt: str = "pdf", language: str = "en") -> CleanResult:
    pages = 6 if fmt == "pdf" else None
    return clean_corpus([(document(blocks, fmt, pages), language)])[0]


def paragraphs(result: CleanResult) -> list[str]:
    return [b.text for b in result.document.blocks if isinstance(b, Paragraph)]


def normalized(text: str) -> str:
    return normalize_text(text, Counter())


def body_page(page: int, *lines: str) -> Paragraph:
    return Paragraph(text="\n".join(lines), page=page)


FILLER = ["alpha line one", "beta line two", "gamma line three", "delta line four"]
MORE = ["epsilon line five", "zeta line six", "eta line seven", "theta line eight"]


# Zones
def test_every_non_main_zone_is_removed_and_recorded():
    blocks = [
        Paragraph(text="Skip to content", zone="nav"),
        Paragraph(text="Search the site", zone="header"),
        Paragraph(text="Real content"),
        Paragraph(text="Related links", zone="aside"),
        Paragraph(text="Copyright notice", zone="footer"),
    ]
    result = clean_one(blocks, "html")
    assert paragraphs(result) == ["Real content"]
    assert {r.rule for r in result.removals} == {
        "zone_nav",
        "zone_header",
        "zone_aside",
        "zone_footer",
    }


def test_first_copy_of_a_pdf_header_line_stays_as_main_text():
    blocks = [
        Paragraph(text="Revision date: 20.02.2020", page=1, zone="header"),
        Paragraph(text="Body one", page=1),
        Paragraph(text="Revision date: 20.02.2020", page=2, zone="header"),
        Paragraph(text="Body two", page=2),
    ]
    result = clean_one(blocks)
    assert paragraphs(result) == ["Revision date: 20.02.2020", "Body one", "Body two"]
    assert result.document.blocks[0].zone == "main"
    assert result.counts["header_kept"] == 1
    assert result.counts["zone_header"] == 1


def test_pdf_footers_are_removed_completely():
    blocks = [
        Paragraph(text="Company address line", page=1, zone="footer"),
        Paragraph(text="Body", page=1),
    ]
    assert paragraphs(clean_one(blocks)) == ["Body"]


def test_a_footer_repeated_within_one_page_is_content():
    blocks = [Paragraph(text="EN 809:1998; EN 60034-1:2010", page=2, zone="footer") for _ in "abc"]
    blocks.append(Paragraph(text="Company address", page=2, zone="footer"))
    result = clean_one(blocks)
    assert paragraphs(result) == ["EN 809:1998; EN 60034-1:2010"] * 3
    assert result.counts["footer_restored"] == 3


def test_a_repeated_bare_number_in_a_footer_is_still_removed():
    blocks = [Paragraph(text="2", page=2, zone="footer") for _ in "abc"]
    assert paragraphs(clean_one(blocks)) == []


@pytest.mark.parametrize(
    "line",
    ["Page 3 of 12", "Seite: 1/8", "1 / 15", "Strona 2 z 45", "(Fortsetzung von Seite 1)"]
    + ["(Contd. of page 1)", "(Segue da pagina 1)", "DE", "63.3.7"],
)
def test_header_lines_without_information_are_not_kept(line):
    assert not is_worth_keeping(line)


@pytest.mark.parametrize(
    "line",
    ["Safety data sheet", "Last revised date: 20.02.2020 1/12", "Trade name: Abrasive, fine"],
)
def test_header_lines_with_information_are_kept(line):
    assert is_worth_keeping(line)


# Repeated lines
def repeated_blocks(line_for_page) -> list[Block]:
    """Six pages with a line placed by `line_for_page`, surrounded by unique body lines."""
    blocks: list[Block] = []
    for page in range(1, 7):
        before = [f"{text} page {chr(96 + page)}" for text in FILLER]
        after = [f"{text} page {chr(96 + page)}" for text in MORE]
        blocks.append(body_page(page, *line_for_page(page, before, after)))
    return blocks


def test_a_line_repeated_at_the_page_edge_keeps_only_its_first_copy():
    blocks = repeated_blocks(lambda page, before, after: [f"{page} Acme Corp", *before, *after])
    result = clean_one(blocks)
    text = "\n".join(paragraphs(result))
    assert text.count("Acme Corp") == 1
    assert result.counts["repeated_line"] == 5


def test_a_line_repeated_in_the_middle_of_pages_is_real_content():
    blocks = repeated_blocks(
        lambda page, before, after: [*before, "No information available.", *after]
    )
    assert find_repeated_lines(document(blocks)) == set()


def test_a_line_only_sometimes_at_the_page_edge_is_real_content():
    def place(page, before, after):
        line = "No information available."
        return [line, *before, *after] if page <= 2 else [*before, line, *after]

    assert find_repeated_lines(document(repeated_blocks(place))) == set()


def test_a_line_on_too_few_pages_is_not_furniture():
    blocks = repeated_blocks(
        lambda page, before, after: [*(["Acme Corp"] if page <= 2 else []), *before, *after]
    )
    assert find_repeated_lines(document(blocks)) == set()


def test_short_documents_give_no_evidence_of_repetition():
    blocks = [body_page(page, "Acme Corp", "Body") for page in (1, 2)]
    assert find_repeated_lines(document(blocks, pages=2)) == set()


def test_a_standalone_number_survives():
    blocks = [Paragraph(text="Breakthrough time\n480", page=1)]
    assert paragraphs(clean_one(blocks)) == ["Breakthrough time\n480"]


# Boilerplate
def test_back_to_top_is_removed():
    result = clean_one([Paragraph(text="Text"), Paragraph(text="Back to top")], "html")
    assert paragraphs(result) == ["Text"]
    assert result.counts["back_to_top"] == 1


def test_contents_title_goes_only_when_a_table_of_contents_follows():
    blocks = [
        Paragraph(text="Contents"),
        Paragraph(text="1.1 Site ... 3", zone="toc"),
        Paragraph(text="Contents"),
        Paragraph(text="Contents of the spill kit are listed below."),
    ]
    result = clean_one(blocks, "docx")
    assert paragraphs(result) == ["Contents", "Contents of the spill kit are listed below."]
    assert result.counts["contents_title"] == 1


def test_dot_leader_lines_are_removed_from_a_paragraph():
    blocks = [Paragraph(text="1 About this document ........ 4\nReal sentence.", page=2)]
    result = clean_one(blocks)
    assert paragraphs(result) == ["Real sentence."]
    assert result.counts["toc_line"] == 1


# Encoding
@pytest.mark.parametrize("text", ["m³", "25 µg/m²", "№ 5", "H₂O", "T™", "…"])
def test_units_and_symbols_are_not_flattened(text):
    assert normalized(text) == text


def test_decomposed_characters_are_composed():
    assert normalized("Café") == "Café"


def test_ligatures_become_letters():
    assert normalized("ﬁlter ﬂow oﬀ") == "filter flow off"


def test_soft_hyphen_is_a_real_hyphen_in_cas_numbers():
    assert normalized("107\u00ad21\u00ad1") == "107-21-1"


def test_soft_hyphen_after_a_space_joins_a_word_but_not_a_negative_number():
    assert normalized("Propane 1,2 \u00addiol") == "Propane 1,2-diol"
    assert normalized("Freezing point: \u00ad40 °C") == "Freezing point: -40 °C"


@pytest.mark.parametrize(
    ("broken", "fixed"),
    [
        ("protecƟve", "protective"),
        ("boƩle", "bottle"),
        ("aŌer", "after"),
        ("informaƟ on", "information"),
        ("aŌ er", "after"),
        ("quanƟƟes", "quantities"),
        ("Ɵme", "time"),
        ("pH shiŌ", "pH shift"),
        ("Interna\x00onal", "International"),
    ],
)
def test_broken_font_glyphs_are_repaired(broken, fixed):
    assert normalized(broken) == fixed


def test_the_same_letter_in_a_real_name_stays():
    assert normalized("Ōsaka") == "Ōsaka"


@pytest.mark.parametrize(
    ("symbol", "real"),
    [("CO2 \uf0a3 0,2", "CO2 ≤ 0,2"), ("\uf0b3480 min", "≥480 min"), ("\uf0fc", "✓")],
)
def test_symbol_font_characters_keep_their_meaning(symbol, real):
    assert normalized(symbol) == real


def test_icons_and_control_characters_are_deleted_and_listed():
    counts: Counter[str] = Counter()
    removals: list[Removal] = []
    text = normalize_text("Press \ue036 now\n\ue04b\n\x01 0,1%", counts, removals, page=4)
    assert text == "Press now\n0,1%"
    assert counts["icon_glyph"] == 2
    assert counts["control_char"] == 1
    assert Removal("icon_glyph", "U+E036", 4) in removals


def test_every_text_field_of_a_block_is_normalized():
    blocks = [
        Heading(level=1, text="ﬁrst"),
        ListBlock(items=[ListItem(text="ﬁve")]),
        Table(rows=[["CO \uf0a3 1", "ﬁ"]], caption="ﬁg"),
    ]
    heading, items, table = clean_one(blocks, "docx").document.blocks
    assert heading.text == "first"
    assert items.items[0].text == "five"
    assert table.rows == [["CO ≤ 1", "fi"]]
    assert table.caption == "fig"


def test_code_blocks_stay_exactly_as_written():
    code = 'url = "a\u00adb"  # **keep** [x][y]'
    block = clean_one([Code(text=code)], "md").document.blocks[0]
    assert block.text == code


# Line-end hyphens
def decide(left: str, right: str, language: str = "en", known: tuple[str, ...] = ()) -> str:
    return hyphen_decision(left, right, language, Counter(known))


@pytest.mark.parametrize(
    ("left", "right"),
    [("n", "octanol"), ("STOT", "RE"), ("REACH", "konform"), ("90", "day"), ("Part", "B")]
    + [("CDI", "RJ45"), (")", "one")],
)
def test_hyphen_stays_for_names_codes_and_numbers(left, right):
    assert decide(left, right, "de") == "keep"


@pytest.mark.parametrize(
    ("left", "right", "language"),
    [("Haut", "und", "de"), ("yrkes", "och", "sv"), ("Formulation", "or", "pl")],
)
def test_suspended_hyphen_keeps_the_space(left, right, language):
    assert decide(left, right, language) == "suspended"


def test_a_ukrainian_word_ending_is_not_taken_for_a_conjunction():
    assert decide("робо", "та", "uk") == "join"


def test_the_form_seen_elsewhere_in_the_language_wins():
    assert decide("self", "contained", "de", ("self-contained",)) == "keep"
    assert decide("extin", "guishing", "en", ("extinguishing",)) == "join"


def test_unknown_words_join_only_in_languages_measured_to_wrap_syllables():
    assert decide("nachge", "wiesen", "de") == "join"
    assert decide("токсич", "них", "uk") == "join"
    assert decide("at", "industrial", "pl") == "keep"
    assert decide("cross", "ventilation", "en") == "keep"


def test_repair_replaces_the_line_break_and_leaves_other_breaks_alone():
    counts: Counter[str] = Counter()
    text = "nachge-\nwiesen und\nHaut-\nund Augen\nmask -\nRequirements"
    repaired = repair_hyphens(text, "de", Counter(), counts)
    assert repaired == "nachgewiesen und\nHaut- und Augen\nmask -\nRequirements"
    assert counts["hyphen_join"] == 1
    assert counts["hyphen_suspended"] == 1


def test_repair_handles_an_apostrophe_and_a_closing_bracket():
    assert repair_hyphens("пов'я-\nзана", "uk", Counter(), Counter()) == "пов'язана"
    text = "3(2H)-\none"
    assert repair_hyphens(text, "fr", Counter(), Counter()) == "3(2H)-one"


def test_a_word_wrapped_twice_is_repaired_twice():
    assert repair_hyphens("a1-\nb2-\nc3", "en", Counter(), Counter()) == "a1-b2-c3"


def test_a_wrapped_word_is_not_evidence_for_itself():
    vocabulary: Counter[str] = Counter()
    add_vocabulary(vocabulary, [Paragraph(text="self-\ncontained")])
    assert "self-contained" not in vocabulary


def test_hyphen_repair_uses_words_from_other_documents_of_the_language():
    first = document([Paragraph(text="Wear self-\ncontained apparatus.")])
    second = document([Paragraph(text="A self-contained unit.")])
    results = clean_corpus([(first, "pl"), (second, "pl")])
    assert paragraphs(results[0]) == ["Wear self-contained apparatus."]


# Markdown
@pytest.mark.parametrize(
    ("text", "plain"),
    [
        ("data from [Modbus][modbus] registers", "data from Modbus registers"),
        ("see [this](https://example.com/a) page", "see this page"),
        ("**Namespace Index** is used", "Namespace Index is used"),
    ],
)
def test_markdown_syntax_becomes_plain_text(text, plain):
    assert strip_markdown(text, Counter()) == plain


def test_inline_code_keeps_its_brackets():
    text = "Set `registers[0][1]` or `**raw**` here"
    assert strip_markdown(text, Counter()) == text


def test_markdown_rule_runs_only_on_markdown_files():
    text = "**bold** stays"
    assert paragraphs(clean_one([Paragraph(text=text)], "html")) == [text]
    assert paragraphs(clean_one([Paragraph(text=text)], "md")) == ["bold stays"]


# Finish
def test_blocks_left_empty_are_dropped_and_section_paths_rebuilt():
    blocks = [
        Heading(level=1, text="1 Title ........ 4", page=1),
        Heading(level=1, text="Real section", page=1),
        Paragraph(text="\ue04b", page=1),
        Paragraph(text="Body", page=1),
    ]
    result = clean_one(blocks)
    assert [type(b) for b in result.document.blocks] == [Heading, Paragraph]
    assert result.document.blocks[1].section_path == ["Real section"]


def test_the_source_document_is_not_changed():
    source = document([Paragraph(text="ﬁne", page=1), Paragraph(text="Top", zone="header")])
    before = source.model_dump()
    clean_corpus([(source, "en")])
    assert source.model_dump() == before


# Deduplication
def test_hash_ignores_layout_but_not_words():
    one = [Paragraph(text="Same  text\nhere", page=1)]
    two = [Paragraph(text="Same text here", page=9)]
    other = [Paragraph(text="Same text there")]
    assert content_hash(one) == content_hash(two)
    assert content_hash(one) != content_hash(other)


def test_the_lowest_id_of_identical_documents_stays():
    hashes = {"D003": "x", "D001": "x", "D002": "y"}
    assert find_duplicates(hashes, {}) == {"D003": "D001"}


def test_linked_versions_are_never_merged():
    hashes = {"D066": "x", "D067": "x", "D070": "x"}
    assert find_duplicates(hashes, {"D067": "D066"}) == {"D070": "D066"}


# Report helpers
def test_page_ranges_are_compact():
    assert page_ranges([7, 1, 2, 3, 3]) == "1-3 7"
    assert page_ranges([]) == ""


def test_repeats_become_one_report_row():
    removals = [Removal("zone_footer", f"Page {n} of 3", n) for n in (1, 2, 3)]
    assert removal_rows("D001", removals) == [
        {"id": "D001", "rule": "zone_footer", "pages": "1-3", "times": 3, "text": "Page 1 of 3"}
    ]


def test_different_icons_stay_on_separate_report_rows():
    removals = [Removal("icon_glyph", "U+E015", 1), Removal("icon_glyph", "U+E016", 1)]
    assert len(removal_rows("D019", removals)) == 2


# The committed corpus
def manifest() -> list[dict[str, str]]:
    with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row["in_rag_index"] == "yes"]


@lru_cache
def rebuilt() -> dict[str, CleanResult]:
    pairs = [
        (
            ParsedDocument.model_validate_json(
                (PARSED / f"{row['id']}.json").read_text(encoding="utf-8")
            ),
            row["language"],
        )
        for row in manifest()
    ]
    return {result.document.doc_id: result for result in clean_corpus(pairs)}


def cleaned_text(doc_id: str) -> str:
    data = json.loads((CLEANED / f"{doc_id}.json").read_text(encoding="utf-8"))
    return json.dumps(data["blocks"], ensure_ascii=False)


def test_committed_files_match_the_rules():
    for doc_id, result in rebuilt().items():
        committed = (CLEANED / f"{doc_id}.json").read_text(encoding="utf-8")
        assert committed == to_review_json(result.document), doc_id


def test_report_covers_every_in_scope_document():
    with (CLEANED / "clean_report.csv").open(encoding="utf-8", newline="") as f:
        report = list(csv.DictReader(f))
    assert [row["id"] for row in report] == [row["id"] for row in manifest()]
    assert all(len(row["content_hash"]) == 64 for row in report)


def test_removed_report_lists_what_was_removed():
    with (CLEANED / "removed.csv").open(encoding="utf-8", newline="") as f:
        removed = list(csv.DictReader(f))
    borealis = [row for row in removed if row["id"] == "D003" and row["rule"] == "repeated_line"]
    assert borealis[0]["pages"] == "2-45"
    assert "Trabrennstrasse" in borealis[0]["text"]


def test_no_document_in_the_corpus_is_a_duplicate():
    hashes = {doc_id: r.document.content_hash for doc_id, r in rebuilt().items()}
    assert find_duplicates(hashes, {}) == {}


def test_version_pairs_stay_separate_documents():
    for doc_id in ("D066", "D067", "D071", "D072", "D074", "D075"):
        assert (CLEANED / f"{doc_id}.json").exists()


def test_d054_broken_glyphs_are_gone():
    text = cleaned_text("D054")
    assert not set("ƟƩŌ") & set(text)
    assert "protective" in text


def test_d052_cas_number_is_intact():
    text = cleaned_text("D052")
    assert "107-21-1" in text
    assert "\u00ad" not in text


def test_d042_keeps_its_purity_limits():
    assert "≤" in cleaned_text("D042")


def test_d033_keeps_one_copy_of_its_revision_line():
    text = cleaned_text("D033")
    assert text.count("überarbeitet am: 11.05.2026") == 1
    assert "(Fortsetzung von Seite 1)" not in text


def test_d016_keeps_the_standards_list_of_every_language():
    assert cleaned_text("D016").count("EN 809:1998+A1:2009") >= 25


def test_repeated_real_content_survives_in_d048():
    assert cleaned_text("D048").count("No information available.") >= 8
