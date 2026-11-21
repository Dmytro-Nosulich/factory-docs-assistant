"""Development questions for "I don't know" (S2-14): answerable and unanswerable questions.

The refusal threshold and the answer prompt versions are tuned on these questions, so they are
kept apart from the retrieval gold set (S2-09) and the test set (S2-16). An answerable question
carries evidence in the gold-set format (`retrieval_gold.Evidence`), checked the same way. An
unanswerable one has none: either the corpus is on its topic but does not state the fact
(`near_miss`), or the corpus has nothing on its topic (`off_topic`).
"""

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from factory_docs.config import ROOT
from factory_docs.retrieval_gold import GOLD, Evidence, data_location, load_gold, normalize

DEV = ROOT / "evals" / "dev_questions.toml"


class Kind(StrEnum):
    answerable = "answerable"
    near_miss = "near_miss"  # on the corpus's topics, but no document states the fact
    off_topic = "off_topic"  # nothing in the corpus is about it


class DevQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^V\d{2}$")
    question: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2}$")
    kind: Kind
    notes: str = ""
    evidence: list[Evidence] = []  # answerable only; alternatives, any one of them answers it

    @model_validator(mode="after")
    def evidence_matches_kind(self) -> "DevQuestion":
        if self.kind is Kind.answerable and not self.evidence:
            raise ValueError("an answerable question needs evidence")
        if self.kind is not Kind.answerable and self.evidence:
            raise ValueError(f"a {self.kind.value} question must have no evidence")
        return self

    @property
    def answerable(self) -> bool:
        return self.kind is Kind.answerable


class DevSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: list[DevQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "DevSet":
        ids = [q.id for q in self.question]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question id: {', '.join(duplicates)}")
        return self


def question_key(text: str) -> str:
    """Whitespace and case do not make a question new."""
    return normalize(text).casefold()


def overlaps(questions: list[DevQuestion], other_texts: list[str]) -> list[str]:
    """IDs of dev questions whose text is also a question of another set."""
    taken = {question_key(text) for text in other_texts}
    return [q.id for q in questions if question_key(q.question) in taken]


def load_dev(path: Path = DEV, gold: Path | None = GOLD) -> list[DevQuestion]:
    """The questions of a dev file; raises ValueError naming every problem.

    gold: a retrieval gold file whose questions must not reappear here (None: no check).
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name}: not valid TOML: {error}") from error
    try:
        questions = DevSet.model_validate(data).question
    except ValidationError as error:
        problems = []
        for err in error.errors():
            loc = data_location(data, err["loc"])
            problems.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(problems)) from error
    if gold is not None:
        shared = overlaps(questions, [q.question for q in load_gold(gold)])
        if shared:
            raise ValueError(
                f"{path.name}: also in {gold.name}, so not a separate question: "
                + ", ".join(shared)
            )
    return questions
