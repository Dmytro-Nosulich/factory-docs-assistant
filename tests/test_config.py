import pytest
from pydantic import ValidationError

from factory_docs.config import (
    CONFIG,
    OllamaModel,
    load_config,
    load_env,
    params_match,
    require_env,
    resolve_sampling,
    run_params,
)


def test_repo_config_loads():
    config = load_config()
    assert config.extraction.prompt_version == "v1"
    assert config.models["haiku-5.5"].slug == "anthropic/claude-haiku-5.5"


def test_local_models_are_ollama_with_explicit_context():
    config = load_config()
    local = {k: m for k, m in config.models.items() if m.client == "ollama"}
    assert set(local) >= {"gemma4-e4b", "gemma4-12b", "qwen3.5-9b"}
    for model in local.values():
        assert isinstance(model, OllamaModel)
        assert model.think is False
        assert model.num_ctx > model.num_predict


def test_sampling_configurations_load_and_local_models_name_one():
    config = load_config()
    assert config.sampling["default"].model_dump() == {"temperature": 1.0, "top_p": 0.95}
    assert config.sampling["greedy"].model_dump() == {"temperature": 0.0, "top_p": 1.0}
    for model in config.models.values():
        if isinstance(model, OllamaModel):
            assert model.sampling in config.sampling


def config_with(tmp_path, old: str, new: str):
    text = CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_model_naming_a_missing_sampling_table_is_rejected(tmp_path):
    path = config_with(tmp_path, 'sampling = "default"', 'sampling = "nope"')
    with pytest.raises(ValidationError, match="no \\[sampling.'nope'\\] table"):
        load_config(path)


def test_comparison_naming_a_missing_model_is_rejected(tmp_path):
    path = config_with(tmp_path, '"haiku-4.5", "qwen3.5-9b"]', '"haiku-4.5", "nope"]')
    with pytest.raises(ValidationError, match="\\[comparison\\] model 'nope'"):
        load_config(path)


def test_comparison_naming_a_missing_sampling_table_is_rejected(tmp_path):
    path = config_with(tmp_path, '["default", "greedy"]', '["default", "nope"]')
    with pytest.raises(ValidationError, match="\\[comparison\\] sampling 'nope'"):
        load_config(path)


def test_out_of_range_temperature_is_rejected(tmp_path):
    path = config_with(tmp_path, "temperature = 1.0", "temperature = 3.0")
    with pytest.raises(ValidationError, match="temperature"):
        load_config(path)


def test_resolve_sampling_for_local_model():
    config = load_config()
    assert resolve_sampling(config, "qwen3.5-9b")[0] == "default"
    name, values = resolve_sampling(config, "qwen3.5-9b", "greedy")
    assert (name, values.temperature, values.top_p) == ("greedy", 0.0, 1.0)
    with pytest.raises(ValueError, match="unknown sampling 'nope'; config.toml has default"):
        resolve_sampling(config, "qwen3.5-9b", "nope")


def test_cloud_model_takes_no_sampling():
    config = load_config()
    assert resolve_sampling(config, "haiku-5.5") is None
    with pytest.raises(ValueError, match="I-003"):
        resolve_sampling(config, "haiku-5.5", "greedy")


def test_haiku_4_5_takes_sampling_but_sends_only_temperature():
    config = load_config()
    assert resolve_sampling(config, "haiku-4.5")[0] == "default"
    assert run_params(config, "haiku-4.5", "greedy") == {
        "reasoning_effort": None,
        "max_tokens": 8000,
        "sampling": {"name": "greedy", "temperature": 0.0},  # no top_p: Claude 4+ rejects both
    }
    assert run_params(config, "haiku-4.5")["sampling"] == {"name": "default", "temperature": 1.0}


@pytest.mark.parametrize("kept", ['sampling = "default"\n', 'sampling_params = ["temperature"]\n'])
def test_cloud_sampling_and_its_params_go_together(tmp_path, kept):
    path = config_with(tmp_path, 'sampling = "default"\nsampling_params = ["temperature"]\n', kept)
    with pytest.raises(ValidationError, match="sampling and sampling_params together"):
        load_config(path)


def test_run_params_log_sampling_name_and_values():
    config = load_config()
    assert run_params(config, "haiku-5.5") == {
        "reasoning_effort": "none",
        "max_tokens": 8000,
        "sampling": None,  # not sent: the provider's default (I-003)
    }
    assert run_params(config, "qwen3.5-9b", "greedy") == {
        "num_ctx": 65536,
        "num_predict": 4000,
        "think": False,
        "sampling": {"name": "greedy", "temperature": 0.0, "top_p": 1.0},
    }


def test_haiku_5_5_params_still_match_its_saved_runs():
    """S1-22 added cloud sampling fields; the params a Haiku 5.5 run logs must not change, or
    the comparison would drop the existing runs (report.latest_complete_run)."""
    config = load_config()
    saved = {"reasoning_effort": "none", "max_tokens": 8000, "sampling": None}  # S1-17 runs
    assert run_params(config, "haiku-5.5") == saved


def test_params_saved_before_s1_16_still_match():
    config = load_config()
    old_cloud = {"reasoning_effort": "none", "max_tokens": 8000}  # no sampling key
    assert params_match(old_cloud, run_params(config, "haiku-5.5"))
    assert not params_match({**old_cloud, "max_tokens": 1}, run_params(config, "haiku-5.5"))
    local = run_params(config, "qwen3.5-9b", "greedy")
    assert params_match(local, local)
    assert not params_match(local, run_params(config, "qwen3.5-9b", "default"))


def test_ollama_model_without_num_ctx_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    extra = (
        '\n[models."bad"]\nclient = "ollama"\nslug = "x:1b"\nnum_predict = 100\n'
        'sampling = "default"\n'
    )
    path.write_text(CONFIG.read_text(encoding="utf-8") + extra, encoding="utf-8")
    with pytest.raises(ValidationError, match="num_ctx"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG.read_text(encoding="utf-8") + "\n[typo]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="typo"):
        load_config(path)


def test_load_env_reads_file_and_keeps_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("FD_TEST_NEW", raising=False)
    monkeypatch.setenv("FD_TEST_SET", "from-shell")
    path = tmp_path / ".env"
    path.write_text("# comment\n\nFD_TEST_NEW='abc'\nFD_TEST_SET=from-file\n", encoding="utf-8")
    load_env(path)
    assert require_env("FD_TEST_NEW") == "abc"
    assert require_env("FD_TEST_SET") == "from-shell"
    monkeypatch.delenv("FD_TEST_NEW")


def test_missing_key_explains_what_to_do(monkeypatch):
    monkeypatch.setenv("FD_TEST_EMPTY", "")
    with pytest.raises(RuntimeError, match="copy .env.example to .env"):
        require_env("FD_TEST_EMPTY")


def test_answer_settings_load():
    answer = load_config().answer
    assert (answer.model, answer.collection, answer.k) == ("haiku-5.5", "structure", 10)
    assert (answer.max_context_tokens, answer.stop_at_usd) == (4000, 5.0)


def test_context_budget_smaller_than_one_chunk_is_rejected(tmp_path):
    path = config_with(tmp_path, "max_context_tokens = 4000", "max_context_tokens = 400")
    with pytest.raises(ValidationError, match="max_context_tokens"):
        load_config(path)


def test_answer_naming_a_missing_model_or_collection_is_rejected(tmp_path):
    path = config_with(tmp_path, 'model = "haiku-5.5"            # decision 15', 'model = "nope"')
    with pytest.raises(ValidationError, match="\\[answer\\] model 'nope'"):
        load_config(path)
    path = config_with(
        tmp_path, 'collection = "structure"       # decision 19', 'collection = "nope"'
    )
    with pytest.raises(ValidationError, match="\\[answer\\] collection 'nope'"):
        load_config(path)


def test_score_gate_needs_the_reranker(tmp_path):
    assert load_config().answer.min_rerank_score == -1.03  # picked by `make eval-refusal`
    path = config_with(tmp_path, "min_rerank_score = -1.03", "min_rerank_score = -3.5")
    assert load_config(path).answer.min_rerank_score == -3.5
    text = path.read_text(encoding="utf-8").replace("rerank = true", "rerank = false", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="min_rerank_score is a reranker score"):
        load_config(path)


def test_refusal_eval_versions_must_be_distinct(tmp_path):
    assert load_config().refusal_eval.prompt_versions == ["v1", "v2"]
    path = config_with(tmp_path, 'prompt_versions = ["v1", "v2"]', 'prompt_versions = ["v1", "v1"]')
    with pytest.raises(ValidationError, match="prompt_versions must be distinct"):
        load_config(path)
