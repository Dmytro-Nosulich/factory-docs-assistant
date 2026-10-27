import json

from factory_docs.config import load_config
from factory_docs.extract import extract_document
from factory_docs.models import ModelCallError, ModelResponse

# An invented product, not a corpus document.
VALID = {
    "product_name": "Example Cleaner",
    "language": "en",
    "signal_word": "warning",
    "physical_state": "liquid",
    "components": [],
    "hazard_statements": [{"code": "H319", "text": "Causes serious eye irritation."}],
    "flash_point": None,
    "un_number": None,
    "revision_date": "2024-08-20",
}


class FakeClient:
    """Replies in order: a string is the model's content, an exception is raised.

    A single reply is repeated for every call.
    """

    def __init__(self, replies: str | Exception | list[str | Exception], digest=None):
        self.replies = replies
        self.digest = digest
        self.conversations = []

    @property
    def messages(self):
        return self.conversations[0]

    def complete(self, messages, schema):
        self.conversations.append(messages)
        reply = self.replies.pop(0) if isinstance(self.replies, list) else self.replies
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(
            content=reply,
            finish_reason="stop",
            response_model="vendor/model",
            provider="Vendor",
            generation_id="gen-1",
            prompt_tokens=10,
            completion_tokens=4,
            reasoning_tokens=0,
            cost_usd=0.00001,
            latency_s=0.5,
            total_s=0.5,
            attempts=1,
            model_digest=self.digest,
        )


def run(tmp_path, reply, model_key="haiku-5.5", sampling=None, digest=None):
    (tmp_path / "D999.txt").write_text("[[page 1]]\nExample Cleaner SDS\n", encoding="utf-8")
    client = FakeClient(reply, digest)
    record = extract_document("D999", client, model_key, load_config(), tmp_path, sampling)
    return record, client


def test_valid_reply(tmp_path):
    record, client = run(tmp_path, json.dumps(VALID))
    assert record["status"] == "valid"
    assert (record["final_status"], record["repair"], len(client.conversations)) == (
        "valid",
        None,
        1,
    )
    assert record["output"]["product_name"] == "Example Cleaner"
    assert record["errors"] == []
    assert record["prompt"]["version"] == "v1"
    assert record["model"]["provider"] == "Vendor"
    assert record["model"]["digest"] is None  # cloud model: response_model and provider instead
    assert record["params"] == {"reasoning_effort": "none", "max_tokens": 8000, "sampling": None}
    assert (record["input"]["pages_sent"], record["input"]["truncated"]) == (1, False)
    assert record["input"]["max_input_tokens"] == 20000
    assert "Example Cleaner SDS" in client.messages[1]["content"]


def test_local_model_record_logs_sampling_and_digest(tmp_path):
    record, _ = run(tmp_path, json.dumps(VALID), "qwen3.5-9b", "greedy", digest="abc123")
    assert record["params"]["sampling"] == {"name": "greedy", "temperature": 0.0, "top_p": 1.0}
    assert record["model"]["digest"] == "abc123"
    assert "model_digest" not in record["call"]  # moved to model, like provider
    record, _ = run(tmp_path, json.dumps(VALID), "qwen3.5-9b")  # the model's default
    assert record["params"]["sampling"]["name"] == "default"


def test_schema_invalid_reply(tmp_path):
    record, _ = run(tmp_path, json.dumps({**VALID, "signal_word": "caution"}))
    assert record["status"] == "invalid"
    assert record["output"] is None
    assert record["errors"][0].startswith("signal_word:")
    assert record["raw"] is not None


def test_non_json_reply(tmp_path):
    record, _ = run(tmp_path, "Sure! Here is the JSON:")
    assert record["status"] == "invalid"
    assert record["errors"][0].startswith("not valid JSON")


def test_failed_call(tmp_path):
    record, client = run(tmp_path, ModelCallError("timeout (gave up after 4 attempts)", None, 4))
    assert record["status"] == "error"
    assert record["call"] == {"attempts": 4, "http_status": None}
    assert "gave up" in record["errors"][0]
    # no reply to repair: no repair call
    assert (record["final_status"], record["repair"], len(client.conversations)) == (
        "error",
        None,
        1,
    )


INVALID = json.dumps({**VALID, "flash_point": {"min": 10, "max": 5, "unit": "celsius"}})


def test_invalid_reply_is_repaired(tmp_path):
    record, client = run(tmp_path, [INVALID, json.dumps(VALID)])
    assert record["final_status"] == "repaired"
    # the first response stays as it was
    assert (record["status"], record["output"], record["raw"]) == ("invalid", None, INVALID)
    assert record["errors"][0].startswith("flash_point:")
    repair = record["repair"]
    assert (repair["status"], repair["errors"]) == ("valid", [])
    assert repair["output"]["product_name"] == "Example Cleaner"
    assert repair["prompt"]["name"] == "repair" and repair["prompt"]["version"] == "v1"
    assert repair["call"]["cost_usd"] == 0.00001
    # the same conversation, plus the invalid reply and a message with the validation errors
    first, second = client.conversations
    assert second[: len(first)] == first
    assert second[len(first)] == {"role": "assistant", "content": INVALID}
    assert second[-1]["role"] == "user"
    assert f"- {record['errors'][0]}" in second[-1]["content"]


def test_invalid_repair_fails(tmp_path):
    record, client = run(tmp_path, [INVALID, "still not JSON"])
    assert (record["status"], record["final_status"]) == ("invalid", "failed")
    assert record["repair"]["status"] == "invalid"
    assert record["repair"]["errors"][0].startswith("not valid JSON")
    assert len(client.conversations) == 2  # one repair attempt only


def test_repair_call_error_fails(tmp_path):
    record, _ = run(
        tmp_path, [INVALID, ModelCallError("HTTP 503 (gave up after 4 attempts)", 503, 4)]
    )
    assert (record["status"], record["final_status"]) == ("invalid", "failed")
    assert record["repair"]["status"] == "error"
    assert record["repair"]["call"] == {"attempts": 4, "http_status": 503}
