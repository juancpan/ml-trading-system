from argparse import Namespace
from pathlib import Path

from algos.backtest_code import run_backtest_optimized as rbo


def _args(**overrides):
    defaults = dict(
        model_name="gnb",
        ticker="BBB",
        start=None,
        end=None,
        interval="1d",
        data_path="data/market_data/BBB.parquet",
        train_split=0.5,
        rf_rate=0.04,
        ptc=0.00035,
        symbol="Adj Close",
        source=None,
        max_leverage=1.3,
        embargo_pct=0.02,
        no_plots=False,
        no_save_intermediates=False,
        skip_model_save=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_data_path_mode_preserves_ticker_and_interval():
    config = rbo.build_backtest_config(_args(), model_params={})

    assert config.data_path == "data/market_data/BBB.parquet"
    assert config.ticker == "BBB"
    assert config.interval == "1d"


def test_default_log_dir_is_backtest_logs_dir_not_caller_cwd():
    bt = rbo.OptimizedBacktester()

    assert bt.logs_dir == Path(rbo.__file__).resolve().parent / "logs"
