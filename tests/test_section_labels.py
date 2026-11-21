import pytest

from factory_docs.chunking import Chunk, Draft
from factory_docs.cleaning import CleanedDocument
from factory_docs.config import ROOT
from factory_docs.parsing.model import Heading, Paragraph
from factory_docs.retrieval_eval import replace_section
from factory_docs.retrieval_gold import Evidence, Question
from factory_docs.section_labels import (
    END,
    MARKDOWN,
    START,
    build_outputs,
    corpus_labels,
    gold_labels,
    label_names,
    passed,
)

S4 = ["SECTION 4: First aid"]
S5 = ["SECTION 5: Firefighting"]
S5_2 = [*S5, "5.2 Special hazards"]
S5_3 = [*S5, "5.3 Advice for firefighters"]
QUOTE = "Heat may cause the containers to explode."


@pytest.mark.parametrize(
    ("label", "section", "expected"),
    [
        (S5_2, S5_2, True),  # the same section
        (S5, S5_2, True),  # a parent names it too
        (S5_3, S5_2, False),  # a sibling subsection
        (S4, S5_2, False),  # another section
        ([*S4, "4.1 Description"], S5_2, False),
        ([], S5_2, False),  # (no section heading)
        ([], [], False),  # nothing to name is never correct
        (S5_2, S5, False),  # more specific than the block: claims a subsection it is not in
    ],
)
def test_a_label_names_a_section_only_as_a_non_empty_prefix(label, section, expected):
    assert label_names(label, section) is expected


# An invented document, not a corpus document.
DOC = CleanedDocument(
    doc_id="D901",
    format="pdf",
    content_hash="x",
    blocks=[
        Heading(page=1, level=1, text=S4[0], section_path=S4),
        Paragraph(page=1, text="Move the person to fresh air.", section_path=S4),
        Heading(page=2, level=1, text=S5[0], section_path=S5),
        Heading(page=2, level=2, text=S5_2[1], section_path=S5_2),
        Paragraph(
            page=2, text="Fire: Heat may cause the\ncontainers to explode.", section_path=S5_2
        ),
        Paragraph(page=2, text="Wear breathing apparatus.", section_path=S5_3),
    ],
)
OTHER = CleanedDocument(
    doc_id="D902",
    format="md",
    content_hash="y",
    blocks=[Paragraph(text="Keep cylinders upright.", section_path=["Storage"])],
)
DOCUMENTS = {"D901": DOC, "D902": OTHER}


def chunk(chunk_id: str, text: str, section_path: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure" if "-s-" in chunk_id else "fixed",
        doc_id=chunk_id[:4],
        index=0,
        title="t",
        section=" > ".join(section_path) or "(no section heading)",
        section_path=section_path,
        page_start=None,
        page_end=None,
        anchor=None,
        version_or_date="v1",
        supersedes=None,
        language="en",
        doc_type="sds",
        format="pdf",
        source_url="https://example.com",
        content_hash="x",
        chunk_hash="h",
        chars=len(text),
        block_kinds=["paragraph"],
        table_part=None,
        text=text,
    )


COLLECTIONS = {
    "structure": [
        chunk("D901-s-0000", "Move the person to fresh air.", S4),
        chunk("D901-s-0001", "Fire: Heat may cause the containers to explode.", S5),
        chunk("D902-s-0000", "Keep cylinders upright.", ["Storage"]),
    ],
    "fixed": [
        # The overlap puts the quote in two chunks; the first starts in section 4
        chunk("D901-f-0000", "fresh air. Fire: Heat may cause the containers to explode.", S4),
        chunk("D901-f-0001", "Fire: Heat may cause the containers to explode.", S5_2),
        chunk("D902-f-0000", "Keep cylinders", ["Storage"]),  # the quote is cut off
    ],
}


def question(qid: str, *evidence: Evidence) -> Question:
    return Question(id=qid, question="q?", language="en", evidence=list(evidence))


FIRE = Evidence(doc_id="D901", section="5.2", page=2, quote=QUOTE)
UPRIGHT = Evidence(doc_id="D902", section="Storage", quote="Keep cylinders upright.")


def test_every_chunk_that_holds_the_quote_must_carry_a_correct_label():
    [labels] = gold_labels([question("R01", FIRE)], DOCUMENTS, COLLECTIONS)
    [evidence] = labels.evidence
    assert evidence.answer_sections == (tuple(S5_2),)
    assert [(c.chunk_id, c.correct) for c in evidence.chunks["structure"]] == [
        ("D901-s-0001", True)
    ]
    assert [(c.chunk_id, c.correct) for c in evidence.chunks["fixed"]] == [
        ("D901-f-0000", False),
        ("D901-f-0001", True),
    ]
    assert labels.correct("structure")
    assert not labels.correct("fixed")  # one wrong chunk of two is enough to fail


def test_no_chunk_with_the_quote_fails_and_any_alternative_evidence_can_pass():
    labels = gold_labels([question("R02", UPRIGHT)], DOCUMENTS, COLLECTIONS)
    assert labels[0].correct("structure")
    assert not labels[0].correct("fixed")  # no fixed chunk holds the whole quote

    both = gold_labels([question("R03", UPRIGHT, FIRE)], DOCUMENTS, COLLECTIONS)
    assert not both[0].correct("fixed")
    with_fire_first = gold_labels(
        [question("R04", FIRE, UPRIGHT), question("R05", UPRIGHT)], DOCUMENTS, COLLECTIONS
    )
    assert passed(with_fire_first, "structure") == 2
    assert passed(with_fire_first, "fixed") == 0


def test_corpus_pairs_are_counted_by_hand():
    first_aid, paragraph = DOC.blocks[0], DOC.blocks[1]
    fire_heading, fire_sub, fire_text, advice = DOC.blocks[2:]
    front = Paragraph(text="SAFETY DATA SHEET")  # no section: skipped
    drafts = [
        Draft("a", (front,), ()),  # 0 pairs, 1 skipped; correct (nothing to name)
        Draft("b", (first_aid, paragraph), tuple(S4)),  # 2 of 2
        Draft("c", (paragraph, fire_heading, fire_sub, fire_text), tuple(S4)),  # 1 of 4
        Draft("d", (fire_text, advice), tuple(S5_2)),  # 1 of 2: advice is in 5.3
        Draft("e", (fire_heading, fire_sub, fire_text, advice), tuple(S5)),  # 4 of 4
    ]
    result = corpus_labels(drafts)
    assert (result.pairs, result.pairs_correct, result.skipped) == (12, 8, 1)
    assert (result.chunks, result.chunks_correct) == (5, 3)


def test_the_readme_block_is_replaced_between_its_markers_only():
    text = f"# Title\nprose 1500\n{START}\n\nold\n\n{END}\nmore prose\n"
    new = replace_section(text, "| new |", START, END)
    assert new == f"# Title\nprose 1500\n{START}\n\n| new |\n\n{END}\nmore prose\n"
    with pytest.raises(ValueError):
        replace_section("# Report\nno markers\n", "x", START, END)


@pytest.fixture(scope="module")
def outputs():
    return build_outputs()


def test_the_committed_outputs_follow_from_the_committed_data(outputs):
    assert MARKDOWN.read_text(encoding="utf-8") == outputs.markdown
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(START) == 1 and readme.count(END) == 1
    assert replace_section(readme, outputs.readme_block, START, END) == readme


def test_structure_labels_are_correct_for_every_block_by_construction(outputs):
    structure = outputs.corpus["structure"]
    assert structure.pairs_correct == structure.pairs > 0
    assert structure.chunks_correct == structure.chunks
