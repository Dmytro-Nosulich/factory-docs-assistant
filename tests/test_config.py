import pytest
from pydantic import ValidationError

from factory_docs.config import CONFIG, OllamaModel, load_config, load_env, require_env


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


def test_ollama_model_without_num_ctx_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    extra = '\n[models."bad"]\nclient = "ollama"\nslug = "x:1b"\nnum_predict = 100\n'
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
