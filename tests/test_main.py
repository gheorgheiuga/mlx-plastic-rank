import importlib


def test_main_prints_banner(capsys):
    main = importlib.import_module("main")
    main.main()
    out = capsys.readouterr().out.strip()
    assert "Hello from mlx-plastic-rank!" in out
