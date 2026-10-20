from datetime import date

import pytest
from pydantic import ValidationError

from factory_docs.schema import PhysicalState, SafetyDataSheet, SignalWord, Unit

# An invented product, not a corpus document: the golden set is labeled by hand (S1-05).
VALID = {
    "product_name": "Example Cleaner",
    "language": "en",
    "signal_word": "warning",
    "physical_state": "liquid",
    "components": [
        {
            "name": "example alcohol",
            "cas_number": "64-17-5",
            "concentration": {"min": 1, "max": 3, "unit": "percent"},
        },
        {"name": "fragrance", "cas_number": None, "concentration": None},
    ],
    "hazard_statements": [{"code": "H319", "text": "Causes serious eye irritation."}],
    "flash_point": {"min": 62.5, "max": None, "unit": "celsius"},
    "un_number": "UN1993",
    "revision_date": "2024-08-20",
}


def with_changes(**changes) -> dict:
    return {**VALID, **changes}


def test_valid_example():
    sds = SafetyDataSheet.model_validate(VALID)
    assert sds.signal_word is SignalWord.warning
    assert sds.physical_state is PhysicalState.liquid
    assert sds.components[0].concentration.max == 3
    assert sds.flash_point.unit is Unit.celsius
    assert sds.revision_date == date(2024, 8, 20)


def test_missing_values_as_null():
    sds = SafetyDataSheet.model_validate(
        with_changes(
            signal_word="none",
            physical_state=None,
            components=[],
            hazard_statements=[],
            flash_point=None,
            un_number=None,
            revision_date=None,
        )
    )
    assert sds.components == [] and sds.flash_point is None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(with_changes(signal_word="caution"), id="unknown-signal-word"),
        pytest.param(
            with_changes(flash_point={"min": 10, "max": 5, "unit": "celsius"}), id="min-gt-max"
        ),
        pytest.param(
            with_changes(flash_point={"min": None, "max": None, "unit": "celsius"}),
            id="empty-range",
        ),
        pytest.param(
            with_changes(components=[{"name": "x", "cas_number": "64175", "concentration": None}]),
            id="bad-cas",
        ),
        pytest.param(with_changes(un_number="1993"), id="un-not-normalized"),
        pytest.param(with_changes(storage_conditions="cool"), id="unknown-field"),
        pytest.param({k: v for k, v in VALID.items() if k != "revision_date"}, id="field-missing"),
    ],
)
def test_invalid_examples(data):
    with pytest.raises(ValidationError):
        SafetyDataSheet.model_validate(data)


def test_json_schema_is_strict():
    # OpenAI strict structured output: every property required, no extra keys, in every object.
    schema = SafetyDataSheet.model_json_schema()
    objects = [schema, *schema["$defs"].values()]
    for obj in (o for o in objects if o.get("type") == "object"):
        assert obj["additionalProperties"] is False, obj["title"]
        assert set(obj["required"]) == set(obj["properties"]), obj["title"]
