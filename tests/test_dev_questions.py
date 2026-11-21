import pytest

from factory_docs.dev_questions import DEV, Kind, load_dev

GOLD_TOML = """
[[question]]
id = "R01"
question = "What is the UN number of argon?"
language = "en"

[[question.evidence]]
doc_id = "D039"
section = "SECTION 14"
page = 10
quote = "UN 1006"
"""

ANSWERABLE = """
[[question]]
id = "V01"
question = "Which UN number does compressed argon have?"
language = "en"
kind = "answerable"

[[question.evidence]]
doc_id = "D039"
section = "SECTION 14"
page = 10
quote = "UN 1006"
"""

NEAR_MISS = """
[[question]]
id = "V02"
question = "What is the UN number of hydrogen?"
language = "en"
kind = "near_miss"
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path, text, gold=GOLD_TOML):
    return load_dev(write(tmp_path, "dev.toml", text), write(tmp_path, "gold.toml", gold))


def test_valid_file_loads(tmp_path):
    questions = load(tmp_path, ANSWERABLE + NEAR_MISS)
    assert [q.id for q in questions] == ["V01", "V02"]
    assert [q.answerable for q in questions] == [True, False]
    assert questions[1].kind is Kind.near_miss


def test_answerable_without_evidence_fails(tmp_path):
    text = ANSWERABLE.split("[[question.evidence]]")[0]
    with pytest.raises(ValueError, match="V01: Value error, an answerable question needs"):
        load(tmp_path, text)


def test_unanswerable_with_evidence_fails(tmp_path):
    text = ANSWERABLE.replace('kind = "answerable"', 'kind = "off_topic"')
    with pytest.raises(ValueError, match="off_topic question must have no evidence"):
        load(tmp_path, text)


def test_duplicate_ids_fail(tmp_path):
    with pytest.raises(ValueError, match="duplicate question id: V02"):
        load(tmp_path, NEAR_MISS + NEAR_MISS)


def test_unknown_kind_fails(tmp_path):
    with pytest.raises(ValueError, match="V02 kind"):
        load(tmp_path, NEAR_MISS.replace("near_miss", "maybe"))


def test_gold_question_reused_fails(tmp_path):
    reused = NEAR_MISS.replace(
        "What is the UN number of hydrogen?", "  what is the UN number of   ARGON? "
    )
    with pytest.raises(ValueError, match="also in gold.toml.*: V02"):
        load(tmp_path, reused)


def test_gold_check_can_be_skipped(tmp_path):
    reused = NEAR_MISS.replace("hydrogen", "argon")
    assert load_dev(write(tmp_path, "dev.toml", reused), gold=None)[0].id == "V02"


def test_committed_dev_file_validates():
    questions = load_dev(DEV)
    kinds = {q.kind for q in questions}
    assert kinds == set(Kind)
    assert len(questions) >= 20
