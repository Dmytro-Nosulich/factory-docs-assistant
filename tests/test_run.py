import json

from factory_docs.config import BudgetSettings, load_config
from factory_docs.models import ModelCallError, ModelResponse
from factory_docs.run import SUMMARY, run_all, spent_usd, stage1_doc_ids

# An invented product, not a corpus document.
VALID = {
    "product_name": "Example Cleaner",
    "language": "en",
    "signal_word": "warning",
    "physical_state": "liquid",
    "components": [],
    "hazard_statements": [],
    "flash_point": None,
    "un_number": None,
    "revision_date": "2024-08-20",
}


class FakeClient:
    """Replies in order: a string is the model's content, an exception is raised."""

    def __init__(self, replies: list[str | Exception], cost: float = 0.01):
        self.replies = list(replies)
        self.cost = cost
        self.calls = 0

    def complete(self, messages, schema):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(
            content=reply,
            finish_reason="stop",
            response_model="vendor/model",
            provider="Vendor",
            generation_id="gen-1",
            prompt_tokens=100,
            completion_tokens=20,
            reasoning_tokens=0,
            cost_usd=self.cost,
            latency_s=0.5,
            total_s=0.5,
            attempts=1,
        )


def write_texts(tmp_path, doc_ids):
    texts = tmp_path / "texts"
    texts.mkdir()
    for doc_id in doc_ids:
        (texts / f"{doc_id}.txt").write_text(f"[[page 1]]\n{doc_id} SDS\n", encoding="utf-8")
    return texts


def config_with_budget(stop_at_usd=4.0):
    return load_config().model_copy(update={"budget": BudgetSettings(stop_at_usd=stop_at_usd)})


def run(tmp_path, doc_ids, client, config=None, spent_before=0.0, texts=None):
    texts = texts or write_texts(tmp_path, doc_ids)
    run_dir = tmp_path / "results" / "runs" / "test-run"
    summary = run_all(
        doc_ids, client, "haiku-5.5", config or config_with_budget(), run_dir, spent_before, texts
    )
    return summary, run_dir


def test_one_file_per_document_and_counts(tmp_path):
    client = FakeClient([json.dumps(VALID), "not json", ModelCallError("HTTP 400", 400, 1)])
    summary, run_dir = run(tmp_path, ["D901", "D902", "D903"], client)

    assert sorted(p.name for p in run_dir.iterdir()) == [
        "D901.json",
        "D902.json",
        "D903.json",
        SUMMARY,
    ]
    assert summary["counts"] == {"valid": 1, "invalid": 1, "error": 1, "not_run": 0}
    assert [f["doc_id"] for f in summary["failures"]] == ["D902", "D903"]
    assert summary["cost_usd"] == 0.02
    assert summary["tokens"] == {"prompt": 200, "completion": 40}
    assert summary["finished_at"] is not None
    assert summary["stopped_by_budget"] is False
    assert json.loads((run_dir / SUMMARY).read_text(encoding="utf-8")) == summary


def test_unexpected_exception_is_recorded_and_run_continues(tmp_path):
    texts = write_texts(tmp_path, ["D902"])  # D901 has no text file
    summary, run_dir = run(tmp_path, ["D901", "D902"], FakeClient([json.dumps(VALID)]), texts=texts)

    record = json.loads((run_dir / "D901.json").read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["errors"][0].startswith("FileNotFoundError")
    assert summary["counts"]["valid"] == 1


def test_budget_guard_stops_before_next_call(tmp_path):
    client = FakeClient([json.dumps(VALID)] * 3, cost=0.5)
    summary, run_dir = run(
        tmp_path, ["D901", "D902", "D903"], client, config_with_budget(1.0), spent_before=0.4
    )

    assert client.calls == 2  # 0.4 -> 0.9 -> 1.4, then stop
    assert summary["not_run"] == ["D903"]
    assert summary["counts"]["not_run"] == 1
    assert summary["stopped_by_budget"] is True
    assert not (run_dir / "D903.json").exists()


def test_resume_fills_only_missing_documents(tmp_path):
    doc_ids = ["D901", "D902", "D903"]
    texts = write_texts(tmp_path, doc_ids)
    run(
        tmp_path,
        doc_ids,
        FakeClient(["not json", json.dumps(VALID)], cost=0.5),
        config_with_budget(1.0),
        texts=texts,
    )

    client = FakeClient([json.dumps(VALID)])
    summary, _ = run(tmp_path, doc_ids, client, spent_before=1.0, texts=texts)

    assert client.calls == 1  # only D903; the invalid D901 is not re-run
    assert summary["counts"] == {"valid": 2, "invalid": 1, "error": 0, "not_run": 0}


def test_spent_usd_sums_records_and_skips_summaries(tmp_path):
    (tmp_path / "smoke").mkdir()
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "smoke" / "a.json").write_text(json.dumps({"call": {"cost_usd": 0.25}}))
    (tmp_path / "runs" / "r1" / "b.json").write_text(json.dumps({"call": {"cost_usd": None}}))
    (tmp_path / "runs" / "r1" / "c.json").write_text(json.dumps({"call": None}))
    (tmp_path / "runs" / "r1" / SUMMARY).write_text(json.dumps({"cost_usd": 9.0}))
    assert spent_usd(tmp_path) == 0.25


def test_stage1_has_35_documents():
    doc_ids = stage1_doc_ids()
    assert len(doc_ids) == 35
    assert doc_ids[0] == "D001"
