"""LLM judges for answers to the test set (S2-18). Rules: `docs/METRICS.md`.

Two judges, each a versioned prompt sent to `[judge] model` through any `ChatClient`:

- Quality: the question, the test set's English reference answer (`expected_answer`) and the
  shown answer (status, claim texts, `missing`; no citations) -> reasoning, then `correct`,
  `partially_correct` or `incorrect`. A question with no shown answer (the score gate, a
  rejected answer, a call error) is shown to the judge as a refusal, as S2-17 scores it.
- Faithfulness: every chunk that was sent to the answer model, rendered as the answer model saw
  it, and the numbered claims -> per claim reasoning, then `supported`, `partially_supported`
  or `not_supported`. An answer with no claims (a refusal) is not sent (`no_claims`).

A reply must be JSON, match the schema and, for faithfulness, give exactly one verdict for
each claim number. An invalid reply gets no repair (user's choice: the plan's default); it is
recorded with its errors and counted as not judged. The judges work on saved rows of the shape
`{final_status, answer}` (the rows of `scripts/eval_answers.py`), so they never answer again.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, ValidationError

from evals.scorers import claims, refused
from factory_docs.answer import check_budget
from factory_docs.answer_schema import render_context
from factory_docs.chunking import Chunk
from factory_docs.config import Config, run_params
from factory_docs.extract import prompt_info
from factory_docs.golden import format_loc
from factory_docs.models import ChatClient, ModelCallError
from factory_docs.prompts import load_prompt
from factory_docs.refusal_eval import fraction
from factory_docs.schema import Strict
from factory_docs.test_set import TestQuestion

QUALITY = "quality"
FAITHFULNESS = "faithfulness"
NOT_JUDGED = "not_judged"  # the judge's reply was invalid, or the call failed
NO_CLAIMS = "no_claims"  # faithfulness only: a refusal has nothing to check


class QualityVerdict(StrEnum):
    correct = "correct"
    partially_correct = "partially_correct"
    incorrect = "incorrect"


class SupportVerdict(StrEnum):
    supported = "supported"
    partially_supported = "partially_supported"
    not_supported = "not_supported"


QUALITY_LEVELS = (*QualityVerdict, NOT_JUDGED)
SUPPORT_LEVELS = tuple(SupportVerdict)


class QualityJudgement(Strict):
    reasoning: str = Field(
        description="One to three sentences, in English, comparing the answer's facts with the "
        "reference answer."
    )
    verdict: QualityVerdict = Field(
        description="correct: every reference fact, nothing contradicting; partially_correct: "
        "some reference facts, nothing contradicting; incorrect: contradicts the reference, "
        "another product's or an older version's value, or the wrong refusal decision."
    )


class ClaimJudgement(Strict):
    claim: int = Field(description="The claim's number, as given in <claims>.")
    reasoning: str = Field(
        description="In English: which chunk states the claim, or what the context does not state."
    )
    verdict: SupportVerdict = Field(
        description="supported: the context states all of it; partially_supported: part of "
        "it; not_supported: the context does not state it or contradicts it."
    )


class FaithfulnessJudgement(Strict):
    claims: list[ClaimJudgement] = Field(description="One entry per claim, with its number.")


SCHEMAS = {QUALITY: QualityJudgement, FAITHFULNESS: FaithfulnessJudgement}
SCHEMA_NAMES = {QUALITY: "judge_quality", FAITHFULNESS: "judge_faithfulness"}


def schema_text(judge: str) -> str:
    return json.dumps(SCHEMAS[judge].model_json_schema(), indent=2, ensure_ascii=False)


# --- judge input ---------------------------------------------------------------------------


def shown_claims(row: Mapping) -> list[Mapping]:
    """The claims the user sees: none for a refusal (S2-17 scores it the same way)."""
    return [] if refused(row) else claims(row["answer"])


def render_answer(row: Mapping) -> str:
    """The shown answer for the quality judge: status, numbered claim texts, missing.

    Citations are left out: code checks them (S2-13, S2-17). A row with no shown answer is a
    refusal for the user, so it is shown as one.
    """
    answer = row["answer"]
    if answer is None:
        return "Status: no answer shown (the system refused to answer)\nClaims: (none)"
    lines = [f"Status: {answer['status']}"]
    texts = [claim["text"] for claim in shown_claims(row)]
    lines += ["Claims:", *numbered(texts)] if texts else ["Claims: (none)"]
    if answer.get("missing"):
        lines.append(f"Missing: {answer['missing']}")
    return "\n".join(lines)


def numbered(texts: Sequence[str]) -> list[str]:
    return [f"{i}. {' '.join(text.split())}" for i, text in enumerate(texts, start=1)]


def quality_messages(question: TestQuestion, row: Mapping, config: Config):
    settings = config.judge
    prompt = load_prompt(settings.quality_prompt, settings.quality_prompt_version)
    messages = prompt.render(
        question=question.question,
        reference=question.expected_answer,
        answer=render_answer(row),
        schema=schema_text(QUALITY),
    )
    return prompt, messages


def sent_chunks(chunk_ids: Sequence[str], chunks: Mapping[str, Chunk]) -> list[Chunk]:
    """The chunks that were sent, in the order they were sent. A missing ID is an error: the
    judge must see what the answer model saw."""
    missing = [c for c in chunk_ids if c not in chunks]
    if missing:
        raise ValueError(f"sent chunks not in the chunk file: {', '.join(missing)}")
    return [chunks[c] for c in chunk_ids]


def faithfulness_messages(
    row: Mapping, sent: Sequence[str], chunks: Mapping[str, Chunk], config: Config
):
    settings = config.judge
    prompt = load_prompt(settings.faithfulness_prompt, settings.faithfulness_prompt_version)
    messages = prompt.render(
        context=render_context(sent_chunks(sent, chunks)),
        claims="\n".join(numbered([claim["text"] for claim in shown_claims(row)])),
        schema=schema_text(FAITHFULNESS),
    )
    return prompt, messages


# --- validation and calls ------------------------------------------------------------------


def validate_judgement(
    judge: str, content: str | None, n_claims: int = 0
) -> tuple[dict | None, list[str]]:
    """Parsed judgement and problems; the judgement is None unless every check passes.

    Faithfulness: the claim numbers must be exactly 1..n_claims, each once. The entries are
    returned sorted by claim number.
    """
    if not content:
        return None, ["empty reply"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}"]
    try:
        parsed = SCHEMAS[judge].model_validate(data)
    except ValidationError as e:
        return None, [
            f"{format_loc(err['loc']) or '(top level)'}: {err['msg']}" for err in e.errors()
        ]
    output = parsed.model_dump(mode="json")
    if judge == FAITHFULNESS:
        numbers = [c["claim"] for c in output["claims"]]
        expected = list(range(1, n_claims + 1))
        if sorted(numbers) != expected:
            return None, [f"claim numbers {numbers}, expected each of {expected} once"]
        output["claims"].sort(key=lambda c: c["claim"])
    return output, []


def call_judge(
    client: ChatClient, judge: str, messages: list[dict[str, str]], n_claims: int = 0
) -> dict:
    """One judge call (retries included), validated: status, call, raw, output, errors."""
    try:
        response = client.complete(
            messages, SCHEMAS[judge].model_json_schema(), schema_name=SCHEMA_NAMES[judge]
        )
    except ModelCallError as e:
        return {
            "status": "error",
            "call": {"attempts": e.attempts, "http_status": e.status},
            "raw": None,
            "output": None,
            "errors": [str(e)],
        }
    call = asdict(response)
    raw = call.pop("content")
    output, errors = validate_judgement(judge, raw, n_claims)
    return {
        "status": "valid" if output is not None else "invalid",
        "call": call,
        "raw": raw,
        "output": output,
        "errors": errors,
    }


def judge_record(judge: str, question_id: str, prompt, config: Config) -> dict:
    key = config.judge.model
    return {
        "judge": judge,
        "question_id": question_id,
        "model": {"key": key, "slug": config.models[key].slug},
        "prompt": prompt_info(prompt),
        "params": run_params(config, key),
        "status": None,
        "call": None,
        "raw": None,
        "output": None,
        "errors": [],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def judge_quality(
    question: TestQuestion, row: Mapping, client: ChatClient, config: Config, spent: float
) -> dict:
    """The quality judgement of one saved answer. spent: the stage 2 logs so far; at or above
    `[answer] stop_at_usd`, BudgetExceeded is raised before the call."""
    prompt, messages = quality_messages(question, row, config)
    record = judge_record(QUALITY, question.id, prompt, config)
    check_budget(spent, config)
    record.update(call_judge(client, QUALITY, messages))
    return record


def judge_faithfulness(
    question_id: str,
    row: Mapping,
    sent: Sequence[str],
    chunks: Mapping[str, Chunk],
    client: ChatClient,
    config: Config,
    spent: float,
) -> dict:
    """The faithfulness judgement of one saved answer's claims against the chunks it was sent
    (`sent`: their IDs, in order). No claims: no call, status `no_claims`."""
    n_claims = len(shown_claims(row))
    if not n_claims:
        settings = config.judge
        prompt = load_prompt(settings.faithfulness_prompt, settings.faithfulness_prompt_version)
        record = judge_record(FAITHFULNESS, question_id, prompt, config)
        record["status"] = NO_CLAIMS
        return record
    prompt, messages = faithfulness_messages(row, sent, chunks, config)
    record = judge_record(FAITHFULNESS, question_id, prompt, config)
    check_budget(spent, config)
    record.update(call_judge(client, FAITHFULNESS, messages, n_claims))
    return record


# --- verdicts and rates --------------------------------------------------------------------


def quality_verdict(judgement: Mapping) -> str:
    """correct, partially_correct, incorrect, or not_judged."""
    return judgement["output"]["verdict"] if judgement["status"] == "valid" else NOT_JUDGED


def claim_verdicts(judgement: Mapping) -> list[str] | None:
    """The verdict of each claim in claim order; [] for no claims; None when not judged."""
    if judgement["status"] == NO_CLAIMS:
        return []
    if judgement["status"] != "valid":
        return None
    return [c["verdict"] for c in judgement["output"]["claims"]]


def quality_counts(judgements: Sequence[Mapping]) -> dict:
    """Each quality level as {n, total, rate}, over all questions."""
    verdicts = [quality_verdict(j) for j in judgements]
    return {level: fraction(verdicts.count(level), len(verdicts)) for level in QUALITY_LEVELS}


def faithfulness_counts(judgements: Sequence[Mapping]) -> dict:
    """Claims per support level, over every claim of a judged answer; answers whose claims are
    all supported, over judged answers with claims; answers not judged (an invalid reply)."""
    judged = [v for v in map(claim_verdicts, judgements) if v]
    not_judged = sum(claim_verdicts(j) is None for j in judgements)
    with_claims = sum(j["status"] != NO_CLAIMS for j in judgements)
    every_claim = [verdict for verdicts in judged for verdict in verdicts]
    out = {
        f"claims_{level}": fraction(every_claim.count(level), len(every_claim))
        for level in SUPPORT_LEVELS
    }
    out["answers_all_supported"] = fraction(
        sum(all(v == SupportVerdict.supported for v in verdicts) for verdicts in judged),
        len(judged),
    )
    out["answers_not_judged"] = fraction(not_judged, with_claims)
    return out


def agreement_table(passed: Sequence[bool], judgements: Sequence[Mapping]) -> dict:
    """How often the S2-17 pass verdict and the quality judge agree: {passed: {level: n}}."""
    table = {"passed": dict.fromkeys(QUALITY_LEVELS, 0), "failed": dict.fromkeys(QUALITY_LEVELS, 0)}
    for ok, judgement in zip(passed, judgements, strict=True):
        table["passed" if ok else "failed"][quality_verdict(judgement)] += 1
    return table
