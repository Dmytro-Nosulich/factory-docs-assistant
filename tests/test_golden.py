import json
from pathlib import Path

from factory_docs.golden import check_golden, find_placeholders

# An invented product, not a corpus document.
VALID = {
    "product_name": "Example Cleaner",
    "language": "en",
    "signal_word": "warning",
    "physical_state": "liquid",
    "components": [{"name": "example alcohol", "cas_number": "64-17-5", "concentration": None}],
    "hazard_statements": [{"code": "H319", "text": "Causes serious eye irritation."}],
    "flash_point": None,
    "un_number": None,
    "revision_date": "2024-08-20",
}


def write(tmp_path, data) -> Path:
    path = tmp_path / "D999.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_valid_file_has_no_problems(tmp_path):
    assert check_golden(write(tmp_path, VALID)) == []


def test_find_placeholders_reports_paths():
    data = {"a": "TODO", "b": [{"c": 1}, {"c": "TODO"}], "d": {"e": "TODO"}}
    assert find_placeholders(data) == ["a", "b[1].c", "d.e"]


def test_placeholder_reported_once_not_as_schema_error(tmp_path):
    data = {**VALID, "signal_word": "TODO", "product_name": "TODO"}
    assert check_golden(write(tmp_path, data)) == [
        "product_name: still TODO",
        "signal_word: still TODO",
    ]


def test_schema_error_names_the_field(tmp_path):
    data = {**VALID, "components": [{**VALID["components"][0], "cas_number": "64175"}]}
    [problem] = check_golden(write(tmp_path, data))
    assert problem.startswith("components[0].cas_number:")


def test_broken_json_reports_line(tmp_path):
    [problem] = check_golden(write(tmp_path, '{\n  "product_name": "x",\n}'))
    assert problem.startswith("not valid JSON at line 3")
