import pytest

from factory_docs.__main__ import main


def test_main_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--resume" in capsys.readouterr().out


def test_main_rejects_unknown_model(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "no-such-model"])
    assert "unknown model" in capsys.readouterr().err
