import subprocess

import scripts.retrain_models as retrain


def test_retrain_ticker_surfaces_stdout_when_stderr_is_empty(monkeypatch, tmp_path, capsys):
    parquet = tmp_path / "QQQ.parquet"
    parquet.write_text("placeholder")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="line before\nError loading data from file: expected string or bytes-like object, got 'NoneType'\nBacktest failed!\n",
            stderr="",
        )

    monkeypatch.setattr(retrain.subprocess, "run", fake_run)

    ok = retrain.retrain_ticker({
        "ticker": "QQQ",
        "model_type": "gnb",
        "parquet_path": parquet,
        "label": "QQQ",
    })

    out = capsys.readouterr().out
    assert not ok
    assert "stdout tail" in out
    assert "Error loading data from file" in out
    assert "Backtest failed!" in out
