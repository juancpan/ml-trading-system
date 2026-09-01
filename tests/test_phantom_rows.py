# tests/test_phantom_rows.py
"""Tests for phantom row stripping in data preprocessing."""

import pandas as pd
import numpy as np
import os
import pytest


def _make_ohlcv(n=100, phantom_indices=None):
    """Create synthetic OHLCV data with optional phantom rows."""
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))
    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "Close": close,
            "Volume": np.random.randint(100000, 10000000, n).astype(float),
        },
        index=dates,
    )
    if phantom_indices:
        for i in phantom_indices:
            c = df.iloc[i]["Close"]
            df.iloc[i] = {"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0}
    return df


class TestPhantomRowStripping:
    def test_phantom_rows_stripped_when_env_set(self):
        from algos.backtest_code.run_backtest_optimized import OptimizedBacktester

        bt = OptimizedBacktester()
        df = _make_ohlcv(100, phantom_indices=[10, 20, 30])
        os.environ["STRIP_PHANTOM_ROWS"] = "1"
        try:
            result = bt._preprocess_dataframe(df, "TEST")
            assert len(result) == 96, f"Expected 96, got {len(result)}"
            if "volume" in result.columns:
                assert (result["volume"] == 0).sum() == 0
        finally:
            del os.environ["STRIP_PHANTOM_ROWS"]

    def test_phantom_rows_kept_when_env_not_set(self):
        from algos.backtest_code.run_backtest_optimized import OptimizedBacktester

        bt = OptimizedBacktester()
        df = _make_ohlcv(100, phantom_indices=[10, 20, 30])
        os.environ.pop("STRIP_PHANTOM_ROWS", None)
        result = bt._preprocess_dataframe(df, "TEST")
        assert len(result) == 99

    def test_legitimate_zero_volume_not_stripped(self):
        from algos.backtest_code.run_backtest_optimized import OptimizedBacktester

        bt = OptimizedBacktester()
        df = _make_ohlcv(100)
        df.iloc[10, df.columns.get_loc("Volume")] = 0
        os.environ["STRIP_PHANTOM_ROWS"] = "1"
        try:
            result = bt._preprocess_dataframe(df, "TEST")
            assert len(result) == 99
        finally:
            del os.environ["STRIP_PHANTOM_ROWS"]

    def test_no_volume_column_no_crash(self):
        from algos.backtest_code.run_backtest_optimized import OptimizedBacktester

        bt = OptimizedBacktester()
        df = _make_ohlcv(50)
        df = df.drop(columns=["Volume"])
        os.environ["STRIP_PHANTOM_ROWS"] = "1"
        try:
            result = bt._preprocess_dataframe(df, "TEST")
            assert len(result) == 49
        finally:
            del os.environ["STRIP_PHANTOM_ROWS"]
