"""The S2-15 one-command run: preflight checks, sample questions, the make recipe."""

import subprocess

import pytest

from factory_docs.config import ROOT, load_config
from factory_docs.dev_questions import load_dev
from factory_docs.pipeline import KEY, preflight_problems, sample_questions


def test_preflight_passes_with_a_key_and_budget_left():
    config = load_config()
    assert preflight_problems(config, {KEY: "sk-test"}, 0.0) == []


def test_preflight_fails_without_a_key():
    config = load_config()
    for env in ({}, {KEY: ""}, {KEY: "   "}):
        [problem] = preflight_problems(config, env, 0.0)
        assert "OPENROUTER_API_KEY is not set" in problem


def test_preflight_fails_at_the_budget_guard():
    config = load_config()
    limit = config.answer.stop_at_usd
    assert preflight_problems(config, {KEY: "k"}, limit - 0.01) == []
    [problem] = preflight_problems(config, {KEY: "k"}, limit)
    assert "stop_at_usd" in problem
    assert len(preflight_problems(config, {}, limit + 1)) == 2


def test_committed_sample_questions_are_dev_questions_in_order():
    config = load_config()
    picked = sample_questions(config.run.sample_questions, load_dev())
    assert [q.id for q in picked] == config.run.sample_questions
    # at least one answer in a language other than English, and one refusal to show
    assert {q.language for q in picked} - {"en"}
    assert any(not q.answerable for q in picked)


def test_unknown_sample_question_is_rejected():
    with pytest.raises(ValueError, match="V99"):
        sample_questions(["V01", "V99"], load_dev())


def test_make_run_rag_runs_every_step_in_order():
    out = subprocess.run(
        ["make", "-n", "run-rag"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    steps = [
        "scripts/preflight.py",
        "scripts/parse_corpus.py",
        "scripts/clean_corpus.py",
        "scripts/ingest.py",
        "scripts/build_index.py --if-stale",
        "scripts/check_retrieval_gold.py",
        "scripts/check_dev_questions.py",
        "scripts/section_labels.py",
        "scripts/eval_retrieval.py",
        "scripts/eval_refusal.py",
        "scripts/eval_answers.py",
        "scripts/sample_answers.py",
        "scripts/cost_report.py",
        "scripts/eval_gate.py",
    ]
    positions = [out.find(step) for step in steps]
    assert -1 not in positions, [s for s, p in zip(steps, positions, strict=True) if p == -1]
    assert positions == sorted(positions)
