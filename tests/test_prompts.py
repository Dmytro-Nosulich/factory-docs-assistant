import re

import pytest

from factory_docs.prompts import load_prompt
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
