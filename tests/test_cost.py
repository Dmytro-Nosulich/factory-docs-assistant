"""The S2-15 cost per 1000 questions: gate, repair, tokens x price, report.

The questions and rows are invented; every expected number is computed by hand in the comments.
Prices: $0.10 in, $0.50 out per 1M tokens (the Haiku 5.5 prices in config.toml).
"""

import json

import pytest

from factory_docs.config import ROOT, load_config
from factory_docs.cost import (
    END,
    START,
    markdown,
    one_line,
    per_1000,
    question_costs,
    record_tokens,
    report_section,
    summarize,
    tokens_usd,
)
from factory_docs.retrieval_eval import replace_section

IN, OUT = 0.10, 0.50

QUESTIONS = [
    {"id": "V01", "kind": "answerable", "top_score": 2.0},
    {"id": "V02", "kind": "answerable", "top_score": -1.0},  # exactly at the threshold: called
    {"id": "V03", "kind": "off_topic", "top_score": -5.0},  # below: gated
    {"id": "V04", "kind": "near_miss", "top_score": None},  # no hit: gated (no_context)
]
ROWS = [
    # 1000 in, 200 out: (1000 x 0.10 + 200 x 0.50) / 1e6 = 0.0002
    {"id": "V01", "prompt_tokens": 1000, "completion_tokens": 200, "cost_usd": 0.00021},
    # first call + repair: 3000 in, 600 out: (300 + 300) / 1e6 = 0.0006
    {"id": "V02", "prompt_tokens": 3000, "completion_tokens": 600, "cost_usd": 0.00059},
    # answered with the gate off in the eval, but `make ask` would not call: $0
    {"id": "V03", "prompt_tokens": 5000, "completion_tokens": 100, "cost_usd": 0.0006},
    {"id": "V04", "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None},
]


def data(rows=ROWS, questions=QUESTIONS, model="haiku-5.5"):
    return {
        "created_at": "2026-11-21T00:00:00+00:00",
        "dev": {"sha256": "abc"},
        "settings": {"model": model},
        "questions": questions,
        "runs": {"v2": rows},
    }


def summary(threshold=-1.0, **kwargs):
    return summarize(data(**kwargs), "v2", threshold, "haiku-5.5", IN, OUT, "2026-10-21")


def test_record_tokens_add_the_repair_call():
    first = {"prompt_tokens": 1200, "completion_tokens": 150}
    repair = {"prompt_tokens": 1500, "completion_tokens": 180}
    assert record_tokens({"call": first, "repair": None}) == (1200, 150)
    # 1200 + 1500, 150 + 180
    assert record_tokens({"call": first, "repair": {"call": repair}}) == (2700, 330)
    # no call at all (gate, no hit) or a failed call without usage
    assert record_tokens({"call": None, "repair": None}) == (0, 0)
    assert record_tokens({"call": {"attempts": 3, "http_status": 500}}) == (0, 0)


def test_tokens_times_price():
    assert tokens_usd(1000, 200, IN, OUT) == pytest.approx(0.0002)
    assert tokens_usd(0, 0, IN, OUT) == 0


def test_gated_question_costs_nothing_and_threshold_itself_passes():
    costs = {c["id"]: c for c in question_costs(QUESTIONS, ROWS, -1.0, IN, OUT)}
    assert [costs[i]["gated"] for i in ("V01", "V02", "V03", "V04")] == [False, False, True, True]
    assert costs["V03"]["tokens_usd"] == 0 and costs["V03"]["reported_usd"] == 0
    assert costs["V03"]["prompt_tokens"] == 0
    # the repair call's tokens were already summed into the row: 3000 in, 600 out
    assert costs["V02"]["tokens_usd"] == pytest.approx(0.0006)


def test_no_gate_prices_every_called_question():
    costs = {c["id"]: c for c in question_costs(QUESTIONS, ROWS, None, IN, OUT)}
    # V03 is no longer gated: (5000 x 0.10 + 100 x 0.50) / 1e6 = 0.00055; V04 has no hit
    assert costs["V03"]["tokens_usd"] == pytest.approx(0.00055)
    assert costs["V04"]["gated"]


def test_per_1000_over_all_and_over_called_questions():
    s = summary()
    # all 4: (0.0002 + 0.0006 + 0 + 0) / 4 x 1000 = 0.2; reported (0.00021 + 0.00059) / 4 x 1000
    assert s["all"]["n"] == 4
    assert s["all"]["tokens_usd"] == pytest.approx(0.2)
    assert s["all"]["reported_usd"] == pytest.approx(0.2)
    # only the 2 called: 0.0008 / 2 x 1000 = 0.4; mean tokens (1000 + 3000) / 2, (200 + 600) / 2
    assert s["called"]["n"] == 2
    assert s["called"]["tokens_usd"] == pytest.approx(0.4)
    assert s["called"]["prompt_tokens"] == 2000
    assert s["called"]["completion_tokens"] == 400
    assert s["unanswerable"] == 2


def test_everything_gated_leaves_no_called_row():
    s = summary(threshold=10.0)
    assert s["all"]["tokens_usd"] == 0
    assert s["called"] == {"n": 0}
    assert per_1000([]) == {"n": 0}
    assert "- | - | - | - |" in report_section(s)


def test_rows_without_token_counts_are_refused():
    old = [{k: v for k, v in r.items() if k != "prompt_tokens"} for r in ROWS]
    with pytest.raises(ValueError, match="no token counts"):
        summary(rows=old)


def test_missing_row_version_or_model_is_refused():
    with pytest.raises(ValueError, match="no answer row"):
        summary(rows=ROWS[:3])
    with pytest.raises(ValueError, match="prompt v9"):
        summarize(data(), "v9", -1.0, "haiku-5.5", IN, OUT, "2026-10-21")
    with pytest.raises(ValueError, match="answered by haiku-4.5"):
        summary(model="haiku-4.5")
    with pytest.raises(ValueError, match="no questions"):
        summary(questions=[])


def test_report_shows_the_numbers():
    s = summary()
    section = report_section(s)
    assert "| All development questions (2 gated cost $0) | 4 | - | - | $0.20 | $0.20 |" in section
    assert "| Only those that reach the model | 2 | 2000 | 400 | $0.40 | $0.40 |" in section
    assert "2 of 4 unanswerable" in section
    md = markdown(s)
    assert "| V03 | off_topic | -5.00 | yes | 0 | 0 | $0.000000 | $0.000000 |" in md
    assert "| V04 | near_miss | - | yes |" in md
    assert "$0.20 over the 4 development questions, $0.40 per 1000" in one_line(s)


def test_report_section_replaces_only_its_markers():
    text = f"before\n{START}\nold\n{END}\nafter\n"
    new = replace_section(text, report_section(summary()), START, END)
    assert new.startswith("before\n") and new.endswith("after\n") and "old" not in new


def test_committed_cost_report_matches_committed_refusal_run():
    config = load_config()
    model = config.models[config.answer.model]
    saved = json.loads((ROOT / "evals/results/refusal.json").read_text(encoding="utf-8"))
    s = summarize(
        saved,
        config.answer.prompt_version,
        config.answer.min_rerank_score,
        config.answer.model,
        model.price_in_usd_per_mtok,
        model.price_out_usd_per_mtok,
        model.price_checked.isoformat(),
    )
    assert (ROOT / "evals/results/cost.md").read_text(encoding="utf-8") == markdown(s)
    report = (ROOT / "docs" / "reports" / "stage-2.md").read_text(encoding="utf-8")
    assert report_section(s) in report
