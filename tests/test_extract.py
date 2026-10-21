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
    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.messages = None

    def complete(self, messages, schema):
        self.messages = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return ModelResponse(
            content=self.reply,
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
        )


def run(tmp_path, reply):
    (tmp_path / "D999.txt").write_text("[[page 1]]\nExample Cleaner SDS\n", encoding="utf-8")
    client = FakeClient(reply)
    return extract_document("D999", client, "haiku-5.5", load_config(), texts_dir=tmp_path), client


def test_valid_reply(tmp_path):
    record, client = run(tmp_path, json.dumps(VALID))
    assert record["status"] == "valid"
    assert record["output"]["product_name"] == "Example Cleaner"
    assert record["errors"] == []
    assert record["prompt"]["version"] == "v1"
    assert record["model"]["provider"] == "Vendor"
    assert record["params"] == {"reasoning_effort": "none", "max_tokens": 8000}
    assert (record["input"]["pages_sent"], record["input"]["truncated"]) == (1, False)
    assert "Example Cleaner SDS" in client.messages[1]["content"]


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
    record, _ = run(tmp_path, ModelCallError("timeout (gave up after 4 attempts)", None, 4))
    assert record["status"] == "error"
    assert record["call"] == {"attempts": 4, "http_status": None}
    assert "gave up" in record["errors"][0]
