"""The S2-18 LLM judges: what they are shown, how their replies are checked, the guard, rates.

Questions, chunks and answers are invented (documents D901 and D902 are not corpus documents);
every expected rate is counted by hand in the comments. No model is called: a fake client
returns the replies.
"""

import json

import pytest

from evals.judges import (
    FAITHFULNESS,
    NO_CLAIMS,
    NOT_JUDGED,
    QUALITY,
    QualityJudgement,
    agreement_table,
    call_judge,
    claim_verdicts,
    faithfulness_counts,
    faithfulness_messages,
    judge_faithfulness,
    judge_quality,
    quality_counts,
    quality_messages,
    quality_verdict,
    render_answer,
    validate_judgement,
)
from factory_docs.answer import JUDGEMENTS, BudgetExceeded, save_record, spent_on_answers
from factory_docs.answer_schema import render_context
from factory_docs.chunking import Chunk
from factory_docs.config import load_config, run_params
from factory_docs.models import ModelCallError, ModelResponse
from factory_docs.run import spent_usd
from factory_docs.test_set import TestQuestion


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure",
        doc_id=chunk_id.split("-")[0],
        index=int(chunk_id[-4:]),
        title="Example cleaner SDS",
        section="9. Physical and chemical properties",
        section_path=["9. Physical and chemical properties"],
        page_start=4,
        page_end=4,
        anchor=None,
        version_or_date="2024-08-20",
        supersedes=None,
        language="en",
        doc_type="sds",
        format="pdf",
        source_url="https://example.com/d901.pdf",
        content_hash="x",
        chunk_hash="h",
        chars=len(text),
        block_kinds=["paragraph"],
        table_part=None,
        text=text,
    )


CHUNKS = {
    c.chunk_id: c
    for c in [
        chunk("D901-s-0001", "Flash point: 23 °C (closed cup)"),
        chunk("D901-s-0002", "Physical state: liquid. Colour: colourless."),
        chunk("D902-s-0001", "Flash point: 61 °C (closed cup)"),  # never sent below
    ]
}
SENT = ["D901-s-0002", "D901-s-0001"]  # the order the answer model saw them in

ANSWERABLE = TestQuestion(
    id="T90",
    question="What is the flash point of the example cleaner?",
    language="en",
    category="answerable",
    expected_answer="The flash point is 23 °C (closed cup).",
    key_facts=[["23 °C"]],
    evidence=[
        {"doc_id": "D901", "section": "9", "page": 4, "quote": "Flash point: 23 °C (closed cup)"}
    ],
)
UNANSWERABLE = TestQuestion(
    id="T91",
    question="What is the boiling point of the example cleaner?",
    language="en",
    category="unanswerable",
    expected_answer="Not in the documents.",
)


def claim(text: str, chunk_id: str = "D901-s-0001", quote: str = "Flash point: 23 °C") -> dict:
    return {"text": text, "citations": [{"chunk_id": chunk_id, "quote": quote}]}


ANSWERED = {
    "final_status": "valid",
    "answer": {
        "status": "answered",
        "claims": [
            claim("The flash point is 23 °C."),
            claim("It is measured in a  closed cup.", quote="closed cup"),
        ],
        "missing": None,
    },
}
PARTIAL = {
    "final_status": "repaired",
    "answer": {
        "status": "partial",
        "claims": [claim("The flash point is 23 °C.")],
        "missing": "The test method is not given.",
    },
}
NOT_IN_DOCS = {
    "final_status": "valid",
    "answer": {"status": "not_in_documents", "claims": [], "missing": "No boiling point."},
}
GATED = {"final_status": "below_threshold", "answer": None}
REJECTED = {"final_status": "failed", "answer": None}


class FakeClient:
    """Replies in order: a string is the model's content, an exception is raised."""

    def __init__(self, *replies: str | Exception, cost: float = 0.0002):
        self.replies = list(replies)
        self.cost = cost
        self.calls = []

    def complete(self, messages, schema, schema_name=None):
        self.calls.append({"messages": messages, "schema": schema, "schema_name": schema_name})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(
            content=reply,
            finish_reason="stop",
            response_model="anthropic/claude-haiku-5.5",
            provider="Vendor",
            generation_id="gen-1",
            prompt_tokens=1200,
            completion_tokens=90,
            reasoning_tokens=0,
            cost_usd=self.cost,
            latency_s=1.0,
            total_s=1.0,
            attempts=1,
        )


@pytest.fixture(scope="module")
def config():
    return load_config()


def quality_reply(verdict: str = "correct") -> str:
    return json.dumps({"reasoning": "States 23 °C like the reference.", "verdict": verdict})


def faithfulness_reply(*verdicts: str, numbers: list[int] | None = None) -> str:
    numbers = numbers or list(range(1, len(verdicts) + 1))
    return json.dumps(
        {
            "claims": [
                {"claim": n, "reasoning": "Chunk D901-s-0001 states it.", "verdict": v}
                for n, v in zip(numbers, verdicts, strict=True)
            ]
        }
    )


def user_text(messages) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


# --- what the judges see -------------------------------------------------------------------


def test_quality_judge_sees_status_numbered_claims_and_missing_but_no_citations():
    text = render_answer(ANSWERED)
    assert text == (
        "Status: answered\nClaims:\n"
        "1. The flash point is 23 °C.\n2. It is measured in a closed cup."
    )
    assert "D901-s-0001" not in text  # citations are code's job (S2-13, S2-17)
    assert render_answer(PARTIAL).endswith("Missing: The test method is not given.")


def test_a_refusal_is_shown_to_the_quality_judge_as_one():
    assert render_answer(NOT_IN_DOCS) == (
        "Status: not_in_documents\nClaims: (none)\nMissing: No boiling point."
    )
    for row in (GATED, REJECTED):
        assert "no answer shown (the system refused" in render_answer(row)


def test_quality_prompt_holds_question_reference_and_answer(config):
    prompt, messages = quality_messages(ANSWERABLE, ANSWERED, config)
    assert (prompt.name, prompt.version) == ("judge_quality", "v1")
    text = user_text(messages)
    assert ANSWERABLE.question in text
    assert ANSWERABLE.expected_answer in text
    assert render_answer(ANSWERED) in text
    assert '"verdict"' in text  # the schema


def test_faithfulness_prompt_holds_every_sent_chunk_in_order_and_only_those(config):
    prompt, messages = faithfulness_messages(ANSWERED, SENT, CHUNKS, config)
    assert (prompt.name, prompt.version) == ("judge_faithfulness", "v1")
    text = user_text(messages)
    context = render_context([CHUNKS[c] for c in SENT])
    assert context in text
    assert text.index('id="D901-s-0002"') < text.index('id="D901-s-0001"')
    assert "D902-s-0001" not in text  # in the chunk file, but not sent
    assert "1. The flash point is 23 °C.\n2. It is measured in a closed cup." in text


def test_a_sent_chunk_missing_from_the_chunk_file_stops_the_judge(config):
    with pytest.raises(ValueError, match="D901-s-0009"):
        faithfulness_messages(ANSWERED, ["D901-s-0009"], CHUNKS, config)


# --- replies -------------------------------------------------------------------------------


def test_valid_replies_pass():
    output, errors = validate_judgement(QUALITY, quality_reply("partially_correct"))
    assert errors == [] and output["verdict"] == "partially_correct"
    reply = faithfulness_reply("not_supported", "supported", numbers=[2, 1])
    output, errors = validate_judgement(FAITHFULNESS, reply, n_claims=2)
    assert errors == []
    assert [(c["claim"], c["verdict"]) for c in output["claims"]] == [
        (1, "supported"),
        (2, "not_supported"),
    ]  # sorted by claim number


@pytest.mark.parametrize(
    "reply, n_claims",
    [
        (faithfulness_reply("supported"), 2),  # a claim left out
        (faithfulness_reply("supported", "supported", "supported"), 2),  # one too many
        (faithfulness_reply("supported", "supported", numbers=[1, 1]), 2),  # duplicate
        (faithfulness_reply("supported", "supported", numbers=[1, 3]), 2),  # wrong number
        (faithfulness_reply("true", "supported"), 2),  # not a verdict
    ],
)
def test_faithfulness_reply_needs_one_valid_verdict_per_claim(reply, n_claims):
    output, errors = validate_judgement(FAITHFULNESS, reply, n_claims)
    assert output is None and errors


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "{not json",
        json.dumps({"reasoning": "x", "verdict": "mostly_correct"}),
        json.dumps({"verdict": "correct"}),
        json.dumps({"reasoning": "x", "verdict": "correct", "score": 1}),
    ],
)
def test_invalid_quality_replies_are_not_judged(reply, config):
    record = judge_quality(ANSWERABLE, ANSWERED, FakeClient(reply), config, 0.0)
    assert record["status"] == "invalid" and record["errors"]
    assert quality_verdict(record) == NOT_JUDGED


def test_judge_schemas_are_strict():
    schema = QualityJudgement.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"reasoning", "verdict"}


def test_a_failed_call_is_recorded_and_not_judged():
    result = call_judge(FakeClient(ModelCallError("HTTP 500", 500, 4)), QUALITY, [])
    assert result["status"] == "error"
    assert result["call"] == {"attempts": 4, "http_status": 500}
    assert quality_verdict(result) == NOT_JUDGED


# --- calls and records ---------------------------------------------------------------------


def test_quality_judge_record_names_model_prompt_schema_and_cost(config):
    client = FakeClient(quality_reply())
    record = judge_quality(ANSWERABLE, ANSWERED, client, config, 0.0)
    [call] = client.calls
    assert call["schema_name"] == "judge_quality"
    assert call["schema"] == QualityJudgement.model_json_schema()
    assert record["judge"] == QUALITY and record["question_id"] == "T90"
    assert record["model"] == {"key": "haiku-5.5", "slug": "anthropic/claude-haiku-5.5"}
    assert record["prompt"]["name"] == "judge_quality" and len(record["prompt"]["sha256"]) == 12
    assert record["params"] == run_params(config, "haiku-5.5")
    assert (record["status"], record["call"]["cost_usd"]) == ("valid", 0.0002)
    assert record["call"]["prompt_tokens"] == 1200


def test_faithfulness_judge_checks_every_claim_of_a_shown_answer(config):
    client = FakeClient(faithfulness_reply("supported", "partially_supported"))
    record = judge_faithfulness("T90", ANSWERED, SENT, CHUNKS, client, config, 0.0)
    assert client.calls[0]["schema_name"] == "judge_faithfulness"
    assert claim_verdicts(record) == ["supported", "partially_supported"]


@pytest.mark.parametrize("row", [NOT_IN_DOCS, GATED, REJECTED])
def test_a_refusal_has_no_claims_so_no_faithfulness_call(row, config):
    client = FakeClient()
    record = judge_faithfulness("T91", row, SENT, CHUNKS, client, config, 0.0)
    assert client.calls == []
    assert (record["status"], record["call"], claim_verdicts(record)) == (NO_CLAIMS, None, [])


def test_guard_stops_before_either_judge_call(config):
    limit = config.answer.stop_at_usd
    client = FakeClient(quality_reply(), faithfulness_reply("supported", "supported"))
    with pytest.raises(BudgetExceeded):
        judge_quality(ANSWERABLE, ANSWERED, client, config, limit)
    with pytest.raises(BudgetExceeded):
        judge_faithfulness("T90", ANSWERED, SENT, CHUNKS, client, config, limit)
    assert client.calls == []


def test_judge_logs_count_for_the_stage_2_guard_and_not_for_stage_1(tmp_path):
    answers, judgements = tmp_path / "answers", tmp_path / "judgements"
    save_record({"call": {"cost_usd": 0.25}, "repair": None}, answers)
    save_record({"call": {"cost_usd": 0.5}}, judgements)
    assert spent_on_answers(answers) == 0.25
    assert spent_on_answers(answers, judgements) == 0.75
    assert spent_usd(tmp_path) == 0.0  # the stage 1 guard skips both folders
    assert JUDGEMENTS.name == "judgements"


# --- rates ---------------------------------------------------------------------------------


def judged(verdict: str) -> dict:
    return {"status": "valid", "output": {"reasoning": "r", "verdict": verdict}}


def claims_judged(*verdicts: str) -> dict:
    return {
        "status": "valid",
        "output": {
            "claims": [
                {"claim": i, "reasoning": "r", "verdict": v} for i, v in enumerate(verdicts, 1)
            ]
        },
    }


INVALID = {"status": "invalid", "output": None}
NONE = {"status": NO_CLAIMS, "output": None}


def test_quality_counts_by_hand():
    rows = [judged("correct"), judged("correct"), judged("incorrect"), INVALID]
    rates = quality_counts(rows)
    # 2 of 4 correct, 0 partial, 1 incorrect, 1 not judged
    assert {k: v["n"] for k, v in rates.items()} == {
        "correct": 2,
        "partially_correct": 0,
        "incorrect": 1,
        "not_judged": 1,
    }
    assert all(v["total"] == 4 for v in rates.values())
    assert rates["correct"]["rate"] == 0.5


def test_faithfulness_counts_by_hand():
    rows = [
        claims_judged("supported", "supported"),  # all supported
        claims_judged("supported", "partially_supported", "not_supported"),
        INVALID,  # has claims, not judged
        NONE,  # a refusal: in no denominator
    ]
    rates = faithfulness_counts(rows)
    # claims of judged answers: 2 + 3 = 5; supported 3, partially 1, not 1
    assert (rates["claims_supported"]["n"], rates["claims_supported"]["total"]) == (3, 5)
    assert rates["claims_partially_supported"]["n"] == 1
    assert rates["claims_not_supported"]["n"] == 1
    # judged answers with claims: 2, of which 1 has every claim supported
    assert (rates["answers_all_supported"]["n"], rates["answers_all_supported"]["total"]) == (1, 2)
    # answers with claims: 3, of which 1 not judged
    assert (rates["answers_not_judged"]["n"], rates["answers_not_judged"]["total"]) == (1, 3)


def test_agreement_table_by_hand():
    passed = [True, True, False, False, False]
    rows = [judged("correct"), judged("incorrect"), judged("incorrect"), judged("correct"), INVALID]
    table = agreement_table(passed, rows)
    assert table["passed"] == {
        "correct": 1,
        "partially_correct": 0,
        "incorrect": 1,
        "not_judged": 0,
    }
    assert table["failed"] == {
        "correct": 1,
        "partially_correct": 0,
        "incorrect": 1,
        "not_judged": 1,
    }
