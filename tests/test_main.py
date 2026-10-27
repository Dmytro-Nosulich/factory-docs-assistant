import pytest

from factory_docs.__main__ import main


def test_cloud_model_with_sampling_fails_before_any_call(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "haiku-5.5", "--sampling", "greedy"])
    assert "I-003" in capsys.readouterr().err


def test_unknown_sampling_lists_the_valid_names(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "qwen3.5-9b", "--sampling", "nope"])
    assert "config.toml has default, greedy" in capsys.readouterr().err


def test_unknown_prompt_version_lists_the_available_ones(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "haiku-5.5", "--prompt-version", "v99"])
    assert "available versions: ['v1', 'v2', 'v3']" in capsys.readouterr().err


def test_repair_needs_a_saved_run(capsys):
    with pytest.raises(SystemExit):
        main(["--repair", "no-such-run"])
    assert "no run.json in results/runs/no-such-run" in capsys.readouterr().err


def test_repair_and_resume_are_exclusive(capsys):
    with pytest.raises(SystemExit):
        main(["--repair", "a", "--resume", "b"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_all_takes_no_single_run_options(capsys):
    with pytest.raises(SystemExit):
        main(["--all", "--model", "qwen3.5-9b"])
    assert "it takes only --docs" in capsys.readouterr().err


def test_all_checks_ollama_before_any_call(monkeypatch, capsys):
    def no_ollama(settings, slugs):
        raise RuntimeError(f"Ollama has no model {slugs[0]!r} (run `ollama pull {slugs[0]}`)")

    def no_runs(*args, **kwargs):
        raise AssertionError("a run started")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("factory_docs.__main__.check_ollama", no_ollama)
    monkeypatch.setattr("factory_docs.__main__.run_one", no_runs)
    with pytest.raises(SystemExit):
        main(["--all", "--docs", "D001"])
    assert "ollama pull qwen3.5:9b" in capsys.readouterr().err


def test_all_needs_the_cloud_key_before_any_call(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # set, so .env cannot fill it in
    monkeypatch.setattr("factory_docs.__main__.check_ollama", lambda settings, slugs: None)
    with pytest.raises(SystemExit):
        main(["--all", "--docs", "D001"])
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err
