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
