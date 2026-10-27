import json

import pytest

from factory_docs.config import BudgetSettings, load_config
from factory_docs.models import ModelCallError, ModelResponse
from factory_docs.run import (
    SUMMARY,
    RunMismatch,
    new_run_id,
    repair_run,
    run_all,
    spent_usd,
    stage1_doc_ids,
)

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

    def __init__(self, replies: list[str | Exception], cost: float = 0.01, digest=None):
        self.replies = list(replies)
        self.cost = cost
        self.digest = digest
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
            model_digest=self.digest,
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
    replies = [json.dumps(VALID), "not json", "still not json", ModelCallError("HTTP 400", 400, 1)]
    client = FakeClient(replies)
    summary, run_dir = run(tmp_path, ["D901", "D902", "D903"], client)

    assert sorted(p.name for p in run_dir.iterdir()) == [
        "D901.json",
        "D902.json",
        "D903.json",
        SUMMARY,
    ]
    assert summary["counts"] == {"valid": 1, "invalid": 1, "error": 1, "not_run": 0}
    assert summary["final_counts"] == {"valid": 1, "repaired": 0, "failed": 1, "error": 1}
    assert [f["doc_id"] for f in summary["failures"]] == ["D902", "D903"]
    assert summary["failures"][0]["final_status"] == "failed"
    assert summary["cost_usd"] == 0.03  # D902's repair call is paid too
    assert summary["tokens"] == {"prompt": 300, "completion": 60}
    assert summary["finished_at"] is not None
    assert summary["stopped_by_budget"] is False
    assert json.loads((run_dir / SUMMARY).read_text(encoding="utf-8")) == summary


def test_unexpected_exception_is_recorded_and_run_continues(tmp_path):
    texts = write_texts(tmp_path, ["D902"])  # D901 has no text file
    summary, run_dir = run(tmp_path, ["D901", "D902"], FakeClient([json.dumps(VALID)]), texts=texts)

    record = json.loads((run_dir / "D901.json").read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["errors"][0].startswith("FileNotFoundError")
    # the failed record still says which configuration it belongs to (S1-16)
    assert record["model"] == {
        "key": "haiku-5.5",
        "client": "openrouter",
        "slug": "anthropic/claude-haiku-5.5",
    }
    assert record["prompt"] == summary["prompt"]
    assert record["params"] == summary["params"]
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
        FakeClient(["not json", "still not json", json.dumps(VALID)], cost=0.25),
        config_with_budget(0.7),  # 0.5 after D901 and its repair, 0.75 after D902: stop
        texts=texts,
    )

    client = FakeClient([json.dumps(VALID)])
    summary, _ = run(tmp_path, doc_ids, client, spent_before=1.0, texts=texts)

    assert client.calls == 1  # only D903; the failed D901 is neither re-run nor repaired again
    assert summary["counts"] == {"valid": 2, "invalid": 1, "error": 0, "not_run": 0}


def test_summary_logs_sampling_and_digest(tmp_path):
    texts = write_texts(tmp_path, ["D901"])
    run_dir = tmp_path / "results" / "runs" / "test-run"
    summary = run_all(
        ["D901"],
        FakeClient([json.dumps(VALID)], digest="abc123"),
        "qwen3.5-9b",
        config_with_budget(),
        run_dir,
        0.0,
        texts,
        sampling="greedy",
    )
    assert summary["params"]["sampling"] == {"name": "greedy", "temperature": 0.0, "top_p": 1.0}
    record = json.loads((run_dir / "D901.json").read_text(encoding="utf-8"))
    assert record["params"] == summary["params"]
    assert (record["model"]["digest"], summary["model"]["digests"]) == ("abc123", ["abc123"])


def test_run_id_names_model_sampling_and_prompt():
    assert new_run_id("qwen3.5-9b", "v1", "greedy").endswith("Z_qwen3.5-9b_greedy_v1")
    assert new_run_id("haiku-5.5", "v1").endswith("Z_haiku-5.5_v1")


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


def saved_before_repair(tmp_path):
    """A run saved before the repair step: D901 valid, D902 invalid, D903 error."""
    doc_ids = ["D901", "D902", "D903"]
    texts = write_texts(tmp_path, doc_ids)
    client = FakeClient([json.dumps(VALID), "not json", "-", ModelCallError("HTTP 400", 400, 1)])
    _, run_dir = run(tmp_path, doc_ids, client, texts=texts)
    for doc_id in doc_ids:
        path = run_dir / f"{doc_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["repair"], record["final_status"]
        path.write_text(json.dumps(record), encoding="utf-8")
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    summary["spent_total_usd"] = 0.02  # D901 and D902's first call, as before the repair step
    (run_dir / SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    return run_dir, texts


def test_repair_run_repairs_only_invalid_records(tmp_path):
    run_dir, texts = saved_before_repair(tmp_path)
    before = {d: json.loads((run_dir / f"{d}.json").read_text()) for d in ("D901", "D902")}
    client = FakeClient([json.dumps(VALID)])

    summary = repair_run(run_dir, client, config_with_budget(), 0.0, texts)

    assert client.calls == 1  # only D902
    records = {d: json.loads((run_dir / f"{d}.json").read_text()) for d in ("D901", "D902", "D903")}
    assert [r["final_status"] for r in records.values()] == ["valid", "repaired", "error"]
    assert records["D901"] == {**before["D901"], "repair": None, "final_status": "valid"}
    # the first response of D902 is kept as it was
    assert {k: records["D902"][k] for k in before["D902"]} == before["D902"]
    assert records["D902"]["repair"]["status"] == "valid"
    assert summary["counts"] == {"valid": 1, "invalid": 1, "error": 1, "not_run": 0}
    assert summary["final_counts"] == {"valid": 1, "repaired": 1, "failed": 0, "error": 1}
    assert summary["cost_usd"] == 0.03  # D901, D902 and D902's repair

    assert summary["spent_total_usd"] == 0.03  # the run's 0.02 plus the repair call
    # nothing left to repair: no calls, the same summary
    assert repair_run(run_dir, FakeClient([]), config_with_budget(), 5.0, texts) == summary


def test_repair_run_refuses_a_changed_prompt_or_params(tmp_path):
    run_dir, texts = saved_before_repair(tmp_path)
    config = config_with_budget()
    config.extraction.prompt_version = "v2"  # the run used v1
    with pytest.raises(RunMismatch, match="prompt"):
        repair_run(run_dir, FakeClient([]), config, 0.0, texts)

    config = config_with_budget()
    config.models["haiku-5.5"].max_tokens = 1
    with pytest.raises(RunMismatch, match="parameters"):
        repair_run(run_dir, FakeClient([]), config, 0.0, texts)


def test_repair_run_accepts_params_saved_before_s1_16(tmp_path):
    run_dir, texts = saved_before_repair(tmp_path)
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    del summary["params"]["sampling"]  # cloud runs before S1-16 had no sampling key
    (run_dir / SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    client = FakeClient([json.dumps(VALID)])
    assert repair_run(run_dir, client, config_with_budget(), 0.0, texts)["final_counts"]
    assert client.calls == 1
