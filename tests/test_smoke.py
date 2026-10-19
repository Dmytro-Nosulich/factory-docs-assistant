from factory_docs.__main__ import main


def test_main_runs(capsys):
    main()
    assert "factory_docs" in capsys.readouterr().out
