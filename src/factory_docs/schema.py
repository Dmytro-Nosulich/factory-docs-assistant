"""Extraction schema for safety data sheets (SDS), stage 1 (decision 7).

Every field is required in the JSON; a value the document does not give is `null`
(or an empty list), never left out. "Not stated" and "not applicable" are both `null`.
Text values stay in the document's language, as written. The field descriptions are the
labeling rules: they guide hand-labeling of the golden set and go into the JSON Schema
the models see.
"""

import json
from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

HAZARD_CODE = r"(EUH\d{3}|H\d{3}[A-Za-z]{0,3})"


class Strict(BaseModel):
    """Base: unknown keys are rejected (`additionalProperties: false`)."""

    model_config = ConfigDict(extra="forbid")


class SignalWord(StrEnum):
    danger = "danger"
    warning = "warning"
    none = "none"


class PhysicalState(StrEnum):
    solid = "solid"
    liquid = "liquid"
    gas = "gas"


class Unit(StrEnum):
    percent = "percent"
    celsius = "celsius"


class ValueRange(Strict):
    """A number or a range as SDS write them: "-17 °C", "> 190 °C", "1-<3 %", "≥ 99,5 %"."""

    min: float | None = Field(
        description="Lower bound; equal to max for a single value; null if there is no lower "
        "bound (e.g. '<5 %'). '<' and '≤' are not distinguished. Decimal comma -> point."
    )
    max: float | None = Field(
        description="Upper bound; equal to min for a single value; null if there is no upper "
        "bound (e.g. '> 190 °C', '≥ 99,5 %')."
    )
    unit: Unit = Field(description="percent for concentrations, celsius for temperatures.")

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("at least one of min and max must be set")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is greater than max ({self.max})")
        return self


class Component(Strict):
    name: str = Field(description="Substance name as written in the section 3 table.")
    cas_number: str | None = Field(
        pattern=r"^\d{2,7}-\d{2}-\d$",
        description="CAS number, e.g. '7727-37-9'; if alternative CAS numbers are listed, the "
        "main one only; null if not given.",
    )
    concentration: ValueRange | None = Field(
        description="Concentration in percent; null if not given."
    )


class HazardStatement(Strict):
    code: str = Field(
        pattern=rf"^{HAZARD_CODE}(\+{HAZARD_CODE})*$",
        description="Hazard code, e.g. 'H280', 'EUH066', or a combined one like 'H301+H311'.",
    )
    text: str = Field(
        description="Statement text as written; empty string if only the code is given."
    )


class SafetyDataSheet(Strict):
    product_name: str = Field(description="Trade name of the product from section 1, as written.")
    language: str = Field(
        pattern=r"^[a-z]{2}$",
        description="ISO 639-1 code of the document's language, e.g. 'en', 'de', 'uk'.",
    )
    signal_word: SignalWord = Field(
        description="Signal word from section 2; 'none' if the document states there is none "
        "or gives none."
    )
    physical_state: PhysicalState | None = Field(
        description="Physical state from section 9 (compressed gas -> gas); null if not given."
    )
    components: list[Component] = Field(
        description="Substances in the section 3 table only (not ingredient-group lines such as "
        "'non-ionic surfactants <5%'); empty if none are listed."
    )
    hazard_statements: list[HazardStatement] = Field(
        description="The product's own hazard statements from section 2 only (not the "
        "ingredients' ones from sections 3 or 16); empty if none."
    )
    flash_point: ValueRange | None = Field(
        description="Flash point in celsius (convert from Fahrenheit if needed); null if not "
        "given or not applicable."
    )
    un_number: str | None = Field(
        pattern=r"^UN\d{4}$",
        description="UN number from section 14, normalized to 'UN1066'; null if not given or "
        "not applicable.",
    )
    revision_date: date | None = Field(
        description="Date of the current version (e.g. 'Last revised', 'Revisionsdatum', "
        "'überarbeitet am'), not the print date; null if not given."
    )


def schema_text() -> str:
    """The JSON Schema of SafetyDataSheet as text, for the `{{schema}}` prompt placeholder."""
    return json.dumps(SafetyDataSheet.model_json_schema(), indent=2, ensure_ascii=False)
