from pathlib import Path

import pytest
from pydantic import ValidationError

from factory_docs.answer_schema import Answer, AnswerStatus, answer_schema_text, render_context
from factory_docs.index import load_chunks

CHUNKS = Path(__file__).resolve().parents[1] / "data" / "chunks" / "chunks.jsonl"

# Invented answers for the schema rules, citing a document that does not exist (D999); not
# answer labels (those are the user's, S2-16).
CLAIM = {
    "text": "Der Beispielreiniger ist flüssig.",
    "citations": [{"chunk_id": "D999-s-0001", "quote": "Aggregatzustand: flüssig"}],
}
ANSWERED = {"status": "answered", "claims": [CLAIM], "missing": None}
PARTIAL = {"status": "partial", "claims": [CLAIM], "missing": "Die Farbe fehlt."}
REFUSED = {
    "status": "not_in_documents",
    "claims": [],
    "missing": "The context gives no storage temperature.",
}


@pytest.fixture(scope="module")
def chunks():
    return {chunk.chunk_id: chunk for chunk in load_chunks(CHUNKS)}


@pytest.mark.parametrize("data", [ANSWERED, PARTIAL, REFUSED])
def test_valid_answers(data):
    answer = Answer.model_validate(data)
    assert answer.status == AnswerStatus(data["status"])


def test_cited_chunk_ids():
    two = {**CLAIM, "citations": [*CLAIM["citations"], {"chunk_id": "D999-f-0002", "quote": "x"}]}
    answer = Answer.model_validate({**ANSWERED, "claims": [CLAIM, two]})
    assert answer.cited_chunk_ids() == {"D999-s-0001", "D999-f-0002"}


def citation(**changes) -> dict:
    return {**CLAIM, "citations": [{**CLAIM["citations"][0], **changes}]}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({**ANSWERED, "claims": []}, "answered needs at least one claim"),
        ({**ANSWERED, "missing": "x"}, "answered must have missing = null"),
        ({**PARTIAL, "claims": []}, "partial needs at least one claim"),
        ({**PARTIAL, "missing": None}, "partial must say what is missing"),
        ({**PARTIAL, "missing": "  "}, "partial must say what is missing"),
        ({**REFUSED, "claims": [CLAIM]}, "not_in_documents must have no claims"),
        ({**REFUSED, "missing": None}, "not_in_documents must say what is missing"),
        ({**ANSWERED, "claims": [{**CLAIM, "citations": []}]}, "at least one citation"),
        ({**ANSWERED, "claims": [{**CLAIM, "text": " "}]}, "claim text must not be empty"),
        ({**ANSWERED, "claims": [citation(quote=" ")]}, "quote must not be empty"),
        ({**ANSWERED, "claims": [citation(chunk_id="D999-s-12")]}, "should match pattern"),
        ({**ANSWERED, "claims": [citation(chunk_id="D99-s-0001")]}, "should match pattern"),
        ({**ANSWERED, "claims": [citation(chunk_id="D999-x-0001")]}, "should match pattern"),
        ({**ANSWERED, "status": "maybe"}, "status"),
        ({**ANSWERED, "confidence": 0.9}, "Extra inputs are not permitted"),
        ({**ANSWERED, "claims": [{**CLAIM, "page": 4}]}, "Extra inputs are not permitted"),
        ({"status": "answered", "claims": [CLAIM]}, "missing"),
    ],
)
def test_invalid_answers(data, message):
    with pytest.raises(ValidationError, match=message):
        Answer.model_validate(data)


def test_json_schema_is_strict():
    # OpenAI strict structured output: every property required, no extra keys, in every object.
    schema = Answer.model_json_schema()
    objects = [schema, *schema["$defs"].values()]
    assert {o["title"] for o in objects if o.get("type") == "object"} == {
        "Answer",
        "Claim",
        "Citation",
    }
    for obj in (o for o in objects if o.get("type") == "object"):
        assert obj["additionalProperties"] is False, obj["title"]
        assert set(obj["required"]) == set(obj["properties"]), obj["title"]


def test_schema_text_names_every_field():
    text = answer_schema_text()
    for name in ("status", "claims", "missing", "citations", "chunk_id", "quote"):
        assert f'"{name}"' in text
    assert "not_in_documents" in text


def test_every_corpus_chunk_id_matches_the_citation_pattern(chunks):
    for chunk_id in chunks:
        Answer.model_validate({**ANSWERED, "claims": [citation(chunk_id=chunk_id)]})


def test_render_context_pdf_chunk_has_pages(chunks):
    chunk = chunks["D033-s-0012"]
    text = render_context([chunk])
    head, body = text.split("\n", 1)
    assert head == (
        '<chunk id="D033-s-0012" document="D033" title="Karcher PressurePro Systempflege RM 110: '
        'Sicherheitsdatenblatt" section="ABSCHNITT 9: Physikalische und chemische '
        'Eigenschaften" date="v15, 2026-05-11" page="4-5">'
    )
    assert body == f"{chunk.text}\n</chunk>", "the chunk text is sent unchanged"


def test_render_context_html_chunk_has_no_page_and_shows_supersedes(chunks):
    old, new = chunks["D066-s-0001"], next(c for c in chunks.values() if c.doc_id == "D067")
    text = render_context([old, new])
    first, second = text.split("\n\n<chunk ")
    assert 'id="D066-s-0001"' in first and "page=" not in first and "supersedes=" not in first
    assert 'supersedes="D066"' in second and "page=" not in second
    assert text.count("<chunk ") == 2 and text.count("</chunk>") == 2


def test_render_context_single_page_and_escaped_attributes(chunks):
    chunk = chunks["D001-s-0000"].model_copy(update={"section": 'A "quoted"\nsection & more'})
    head = render_context([chunk]).split("\n", 1)[0]
    assert 'page="1"' in head
    assert 'section="A &quot;quoted&quot; section &amp; more"' in head


def test_render_context_empty():
    assert render_context([]) == ""
