import re

import pytest

from factory_docs.answer_schema import answer_schema_text, render_context
from factory_docs.index import load_chunks
from factory_docs.prompts import PROMPTS, load_prompt
from factory_docs.schema import schema_text


def test_extract_v1_loads_from_file():
    prompt = load_prompt("extract", "v1")
    assert (prompt.name, prompt.version) == ("extract", "v1")
    assert re.fullmatch(r"[0-9a-f]{12}", prompt.sha256)
    assert prompt.system and prompt.user
    assert "<!--" not in prompt.system, "the header comment is not sent to the model"
    assert prompt.placeholders == {"document", "schema"}


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_later_versions_keep_the_v1_placeholders(version):
    v1, later = load_prompt("extract", "v1"), load_prompt("extract", version)
    assert later.version == version
    assert later.placeholders == v1.placeholders
    assert later.sha256 != v1.sha256


def test_repair_v1_has_no_system_message():
    prompt = load_prompt("repair", "v1")
    assert prompt.system is None
    assert prompt.placeholders == {"errors"}
    (user,) = prompt.render(errors="- signal_word: bad value")
    assert user["role"] == "user"
    assert "- signal_word: bad value" in user["content"]


def test_answer_v1_loads_from_file():
    prompt = load_prompt("answer", "v1")
    assert (prompt.name, prompt.version) == ("answer", "v1")
    assert prompt.system and prompt.user
    assert "<!--" not in prompt.system, "the header comment is not sent to the model"
    assert prompt.placeholders == {"question", "context", "schema"}


def test_answer_v1_renders_a_real_chunk():
    chunk = next(
        c
        for c in load_chunks(PROMPTS.parent / "data/chunks/chunks.jsonl")
        if c.chunk_id == "D033-s-0012"
    )
    prompt = load_prompt("answer", "v1")
    system, user = prompt.render(
        question="Wie hoch ist der Flammpunkt von RM 110?",
        context=render_context([chunk]),
        schema=answer_schema_text(),
    )
    assert system == {"role": "system", "content": prompt.system}
    assert '<chunk id="D033-s-0012"' in user["content"]
    assert chunk.text in user["content"]
    assert "Wie hoch ist der Flammpunkt von RM 110?" in user["content"]
    assert '"not_in_documents"' in user["content"]
    assert "{{" not in user["content"]


def test_render_fills_document_and_schema():
    prompt = load_prompt("extract", "v1")
    system, user = prompt.render(document="SECTION 1: Example Cleaner", schema=schema_text())
    assert system == {"role": "system", "content": prompt.system}
    assert user["role"] == "user"
    assert "SECTION 1: Example Cleaner" in user["content"]
    assert '"product_name"' in user["content"]
    assert "{{" not in user["content"]


def test_render_leaves_braces_in_values_alone():
    prompt = load_prompt("extract", "v1")
    _, user = prompt.render(document="odd text {{schema}} {x}", schema="S")
    assert "odd text {{schema}} {x}" in user["content"]


def test_render_rejects_missing_and_unknown_values():
    prompt = load_prompt("extract", "v1")
    with pytest.raises(ValueError, match="missing values: \\['schema'\\]"):
        prompt.render(document="x")
    with pytest.raises(ValueError, match="unknown values: \\['documnet'\\]"):
        prompt.render(document="x", schema="y", documnet="z")


def test_unknown_version_lists_available(tmp_path):
    (tmp_path / "extract_v1.md").write_text("## system\ns\n## user\nu\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="available versions: \\['v1'\\]"):
        load_prompt("extract", "v9", prompts_dir=tmp_path)


def test_file_without_user_section_is_rejected(tmp_path):
    (tmp_path / "extract_v1.md").write_text("## system\nonly a system part\n", encoding="utf-8")
    with pytest.raises(ValueError, match="found \\['system'\\]"):
        load_prompt("extract", "v1", prompts_dir=tmp_path)
