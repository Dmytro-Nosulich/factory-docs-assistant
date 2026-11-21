"""The answer pipeline (S2-13): token budget, citation checks, repair, budget guard, record.

Chunks and answers are invented (document D999 does not exist); they test the rules, they are
not answer labels (those are the user's, S2-16). No model or index is used.
"""

import json

import pytest

from factory_docs.answer import (
    SCHEMA_NAME,
    BudgetExceeded,
    answer_question,
    check_citations,
    context_tokens,
    fit_context,
    save_record,
    shown_answer,
    spent_on_answers,
    validate_answer,
)
from factory_docs.answer_schema import Answer
from factory_docs.chunking import Chunk
from factory_docs.config import load_config
from factory_docs.index import Hit
from factory_docs.models import ModelCallError, ModelResponse

FLASH = "Flammpunkt: 23 °C (geschlossener Tiegel)\nZündtemperatur: 260 °C"
STATE = "Aggregatzustand: flüssig\nFarbe: farblos"
STORAGE = "Kühl und trocken lagern. " * 20  # a long chunk, for the budget


def chunk(chunk_id: str, text: str, page: int | None = 3) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        strategy="structure",
        doc_id="D999",
        index=int(chunk_id[-4:]),
        title="Beispielreiniger SDS",
        section="9. Physikalische und chemische Eigenschaften",
        section_path=["9. Physikalische und chemische Eigenschaften"],
        page_start=page,
        page_end=page,
        anchor=None,
        version_or_date="2024-08-20",
        supersedes=None,
        language="de",
        doc_type="sds",
        format="pdf",
        source_url="https://example.com/d999.pdf",
        content_hash="x",
        chunk_hash="h",
        chars=len(text),
        block_kinds=["paragraph"],
        table_part=None,
        text=text,
    )


CHUNKS = [chunk("D999-s-0001", FLASH), chunk("D999-s-0002", STATE), chunk("D999-s-0003", STORAGE)]
HITS = [
    Hit(1.0 - i / 10, c, vector_rank=i + 1, first_stage_rank=i + 1) for i, c in enumerate(CHUNKS)
]


def claim(chunk_id: str = "D999-s-0001", quote: str = "Flammpunkt: 23 °C") -> dict:
    return {
        "text": "The flash point is 23 °C.",
        "citations": [{"chunk_id": chunk_id, "quote": quote}],
    }


def answered(*claims: dict) -> str:
    return json.dumps({"status": "answered", "claims": list(claims) or [claim()], "missing": None})


REFUSED = json.dumps(
    {"status": "not_in_documents", "claims": [], "missing": "The context gives no boiling point."}
)


class FakeClient:
    """Replies in order: a string is the model's content, an exception is raised."""

    def __init__(self, *replies: str | Exception, cost: float = 0.0001):
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
            prompt_tokens=900,
            completion_tokens=80,
            reasoning_tokens=0,
            cost_usd=self.cost,
            latency_s=1.0,
            total_s=1.0,
            attempts=1,
        )


@pytest.fixture(scope="module")
def config():
    return load_config()


def with_budget(config, max_context_tokens: int):
    answer = config.answer.model_copy(update={"max_context_tokens": max_context_tokens})
    return config.model_copy(update={"answer": answer})


def ask(config, *replies, hits=HITS, spent=0.0, cost=0.0001):
    client = FakeClient(*replies, cost=cost)
    return answer_question(
        "Flammpunkt?", hits, client, config, spent, {"collection": "test"}
    ), client


# --- token budget ---------------------------------------------------------------------------


def test_budget_keeps_rank_order_and_stops_at_the_first_hit_that_does_not_fit():
    two = context_tokens(CHUNKS[:2])
    sent, dropped = fit_context(HITS, two)  # exactly the first two: the boundary fits
    assert [h.chunk.chunk_id for h in sent] == ["D999-s-0001", "D999-s-0002"]
    assert [h.chunk.chunk_id for h in dropped] == ["D999-s-0003"]
    assert context_tokens([h.chunk for h in sent]) <= two
    sent, dropped = fit_context(HITS, two - 1)
    assert [h.chunk.chunk_id for h in sent] == ["D999-s-0001"]
    assert len(dropped) == 2


def test_budget_never_skips_a_hit_to_add_a_smaller_one_after_it():
    hits = [HITS[0], HITS[2], HITS[1]]  # small, large, small
    fits_both_small = context_tokens([CHUNKS[0], CHUNKS[1]])
    assert fits_both_small < context_tokens([CHUNKS[0], CHUNKS[2]])
    sent, dropped = fit_context(hits, fits_both_small)
    assert [h.chunk.chunk_id for h in sent] == ["D999-s-0001"]
    assert [h.chunk.chunk_id for h in dropped] == ["D999-s-0003", "D999-s-0002"]


def test_budget_that_holds_everything_drops_nothing():
    sent, dropped = fit_context(HITS, 100_000)
    assert (len(sent), dropped) == (3, [])
    assert fit_context([], 100) == ([], [])


def test_record_logs_the_budget_and_the_model_sees_only_the_sent_chunks(config):
    budget = context_tokens(CHUNKS[:2])
    record, client = ask(with_budget(config, budget), answered())
    retrieval = record["retrieval"]
    assert [s["chunk_id"] for s in retrieval["sent"]] == ["D999-s-0001", "D999-s-0002"]
    assert retrieval["dropped"] == ["D999-s-0003"]
    assert retrieval["est_context_tokens"] == budget
    assert retrieval["max_context_tokens"] == budget
    prompt = client.calls[0]["messages"][-1]["content"]
    assert 'id="D999-s-0002"' in prompt and 'id="D999-s-0003"' not in prompt


def test_no_hits_means_no_model_call(config):
    record, client = ask(config, hits=[])
    assert (record["status"], record["final_status"], client.calls) == (
        "no_context",
        "no_context",
        [],
    )
    assert shown_answer(record) is None


# --- citation checks ------------------------------------------------------------------------


def test_valid_citation_passes():
    assert check_citations(Answer.model_validate_json(answered()), CHUNKS) == []


def test_chunk_that_was_not_sent_is_rejected():
    answer = Answer.model_validate_json(answered(claim("D999-s-0042", "Flammpunkt")))
    assert check_citations(answer, CHUNKS) == [
        "claims[0].citations[0]: chunk D999-s-0042 is not in the context"
    ]


def test_chunk_retrieved_but_dropped_by_the_budget_is_rejected(config):
    reply = answered(claim("D999-s-0003", "Kühl und trocken lagern."))
    record, _ = ask(with_budget(config, context_tokens(CHUNKS[:2])), reply, reply)
    assert record["final_status"] == "failed"
    assert record["errors"] == ["claims[0].citations[0]: chunk D999-s-0003 is not in the context"]


def test_quote_that_is_not_in_its_chunk_is_rejected():
    # The quote is real text, but from another chunk.
    answer = Answer.model_validate_json(answered(claim("D999-s-0002", "Flammpunkt: 23 °C")))
    [error] = check_citations(answer, CHUNKS)
    assert error.startswith("claims[0].citations[0]: the quote is not in chunk D999-s-0002")


def test_quote_with_other_whitespace_passes():
    answer = Answer.model_validate_json(
        answered(claim(quote="23 °C (geschlossener\n  Tiegel) Zündtemperatur"))
    )
    assert check_citations(answer, CHUNKS) == []


def test_every_bad_citation_gets_its_own_error():
    two = claim()
    two["citations"].append({"chunk_id": "D999-s-0002", "quote": "farbig"})
    answer = Answer.model_validate_json(answered(claim("D999-s-0042", "x"), two))
    errors = check_citations(answer, CHUNKS)
    assert [e.split(":")[0] for e in errors] == ["claims[0].citations[0]", "claims[1].citations[1]"]


def test_refusal_has_nothing_to_check():
    output, errors = validate_answer(REFUSED, CHUNKS)
    assert (output["status"], errors) == ("not_in_documents", [])


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (None, "empty reply"),
        ("{not json", "not valid JSON"),
        (
            json.dumps({"status": "answered", "claims": [], "missing": None}),
            "needs at least one claim",
        ),
    ],
)
def test_bad_json_and_status_rules_are_errors(content, error):
    output, errors = validate_answer(content, CHUNKS)
    assert output is None
    assert error in errors[0]


# --- flow: first reply, repair, reject ------------------------------------------------------


def test_valid_answer_is_shown_and_logged(config):
    record, client = ask(config, answered())
    assert (record["status"], record["final_status"], record["repair"]) == ("valid", "valid", None)
    assert shown_answer(record)["claims"][0]["citations"][0]["chunk_id"] == "D999-s-0001"
    [call] = client.calls
    assert call["schema_name"] == SCHEMA_NAME
    assert call["schema"] == Answer.model_json_schema()


def test_invalid_answer_is_repaired_with_its_errors_listed(config):
    bad = answered(claim("D999-s-0042", "Flammpunkt"))
    record, client = ask(config, bad, answered())
    assert (record["status"], record["final_status"]) == ("invalid", "repaired")
    assert shown_answer(record) == record["repair"]["output"]
    repair_messages = client.calls[1]["messages"]
    assert repair_messages[:-2] == client.calls[0]["messages"]  # the same conversation
    assert repair_messages[-2] == {"role": "assistant", "content": bad}
    assert "chunk D999-s-0042 is not in the context" in repair_messages[-1]["content"]
    assert record["repair"]["prompt"]["name"] == "answer_repair"


def test_answer_invalid_twice_is_rejected(config):
    bad = answered(claim("D999-s-0002", "Flammpunkt"))
    record, client = ask(config, bad, bad)
    assert (record["final_status"], len(client.calls)) == ("failed", 2)
    assert shown_answer(record) is None


def test_failed_call_is_an_error_without_repair(config):
    record, client = ask(config, ModelCallError("HTTP 500", 500, 4))
    assert (record["status"], record["final_status"], record["repair"]) == ("error", "error", None)
    assert shown_answer(record) is None


# --- stage 2 budget guard -------------------------------------------------------------------


def test_guard_stops_before_the_first_call(config):
    client = FakeClient(answered())
    with pytest.raises(BudgetExceeded, match="stop_at_usd"):
        answer_question("Flammpunkt?", HITS, client, config, config.answer.stop_at_usd, None)
    assert client.calls == []


def test_guard_stops_before_the_repair_call(config):
    spent = config.answer.stop_at_usd - 0.00005  # the first call's $0.0001 reaches the limit
    record, client = ask(config, answered(claim("D999-s-0042", "x")), spent=spent)
    assert len(client.calls) == 1
    assert (record["repair"]["status"], record["final_status"]) == ("not_run", "failed")


def test_guard_counts_only_answer_logs(tmp_path):
    answers = tmp_path / "answers"
    assert spent_on_answers(answers) == 0.0
    save_record({"call": {"cost_usd": 0.25}, "repair": {"call": {"cost_usd": 0.5}}}, answers)
    save_record({"call": None, "repair": None}, answers)
    assert spent_on_answers(answers) == 0.75


# --- record ---------------------------------------------------------------------------------


def test_record_names_model_prompt_params_and_cost(config):
    record, _ = ask(config, answered())
    assert record["question"] == "Flammpunkt?"
    assert record["model"] == {
        "key": "haiku-5.5",
        "client": "openrouter",
        "slug": "anthropic/claude-haiku-5.5",
        "response_model": "anthropic/claude-haiku-5.5",
        "provider": "Vendor",
        "digest": None,
    }
    assert (record["prompt"]["name"], record["prompt"]["version"]) == ("answer", "v1")
    assert len(record["prompt"]["sha256"]) == 12
    assert record["params"] == {"reasoning_effort": "none", "max_tokens": 8000, "sampling": None}
    assert record["call"]["cost_usd"] == 0.0001
    assert record["retrieval"]["collection"] == "test"
    assert record["retrieval"]["hits"][0] == {
        "chunk_id": "D999-s-0001",
        "score": 1.0,
        "vector_rank": 1,
        "bm25_rank": None,
        "first_stage_rank": 1,
    }
    assert record["retrieval"]["sent"][0]["pages"] == "3"


def test_save_record_never_overwrites(tmp_path):
    first = save_record({"n": 1}, tmp_path)
    second = save_record({"n": 2}, tmp_path)
    assert first != second
    assert json.loads(first.read_text())["n"] == 1
