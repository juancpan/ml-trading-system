"""Tests for FeatureEngine indicator edge cases."""

import pandas as pd
import numpy as np
import sys
import pytest

sys.path.insert(0, ".")

from algos.common.feature_engine import FeatureRegistry, get_registry


def _make_df(n=200, zero_vol_indices=None, flat_close=False, all_up=False):
    """Create synthetic OHLCV DataFrame for indicator testing."""
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    np.random.seed(42)
    if flat_close:
        close = np.full(n, 91.5)
    elif all_up:
        close = 100 * np.cumprod(1 + np.abs(np.random.normal(0.001, 0.005, n)))
    else:
        close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "close": close,
            "volume": np.random.randint(100000, 10000000, n).astype(float),
            "price": close,
        },
        index=dates,
    )
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df["direction"] = np.where(df["returns"] > 0, 1, -1)
    df.dropna(subset=["returns"], inplace=True)
    if zero_vol_indices:
        for i in zero_vol_indices:
            if i < len(df):
                df.iloc[i, df.columns.get_loc("volume")] = 0
    return df


class TestVolumeRatio:
    def test_no_inf_with_zero_volume(self):
        registry = get_registry()
        df = _make_df(200, zero_vol_indices=[50, 60, 70, 80])
        func = registry.get("volume_ratio")
        result = func(df, period=20)
        assert not result.empty
        col = result.columns[0]
        assert np.isinf(result[col]).sum() == 0, (
            f"Found {np.isinf(result[col]).sum()} inf values in volume_ratio"
        )

    def test_zero_volume_produces_finite_negative(self):
        registry = get_registry()
        df = _make_df(200, zero_vol_indices=[100])
        func = registry.get("volume_ratio")
        result = func(df, period=20)
        col = result.columns[0]
        # _make_df drops first row via dropna(returns), so the df has 199 rows.
        # zero_vol_indices=[100] sets iloc 100 to zero volume in the post-dropna df.
        val = result[col].iloc[100]
        assert np.isfinite(val), f"Expected finite, got {val}"
        assert val < -5, f"Expected large negative (low volume signal), got {val}"


class TestRSI:
    def test_rsi14_no_excessive_nan_on_normal_data(self):
        registry = get_registry()
        df = _make_df(300)
        func = registry.get("rsi")
        result = func(df, period=14)
        col = result.columns[0]
        nan_pct = result[col].isna().mean()
        assert nan_pct < 0.10, f"RSI-14 NaN rate {nan_pct:.1%} exceeds 10%"

    def test_rsi2_no_catastrophic_nan_on_low_vol(self):
        registry = get_registry()
        np.random.seed(42)
        n = 300
        dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
        close = 91.5 + np.cumsum(
            np.random.choice([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, -0.01], n)
        )
        df = pd.DataFrame({"close": close, "price": close}, index=dates)
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        df["direction"] = np.where(df["returns"] > 0, 1, -1)
        df.dropna(inplace=True)
        func = registry.get("rsi2")
        result = func(df)
        col = result.columns[0]
        nan_pct = result[col].isna().mean()
        assert nan_pct < 0.10, (
            f"RSI-2 NaN rate {nan_pct:.1%} on low-vol data (was 86% before fix)"
        )

    def test_rsi2_all_up_returns_overbought(self):
        registry = get_registry()
        df = _make_df(200, all_up=True)
        func = registry.get("rsi2")
        result = func(df)
        col = result.columns[0]
        valid = result[col].dropna()
        assert valid.mean() > 0.8, (
            f"Expected mean > 0.8 for all-up data, got {valid.mean():.2f}"
        )

    def test_connors_rsi_no_catastrophic_nan(self):
        registry = get_registry()
        df = _make_df(300)
        func = registry.get("connors_rsi")
        result = func(df, rsi_period=3, streak_period=2, pctrank_period=100)
        col = result.columns[0]
        nan_pct = result[col].isna().mean()
        assert nan_pct < 0.40, (
            f"Connors RSI NaN rate {nan_pct:.1%} exceeds 40% (was 43% before fix)"
        )


class TestVixTermSlope:
    def test_does_not_crash_with_series_data(self):
        registry = get_registry()
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "close": np.random.uniform(90, 110, n),
                "returns": np.random.normal(0, 0.01, n),
            },
            index=dates,
        )
        ext_data = {
            "vix_close": pd.Series(np.random.uniform(15, 30, n), index=dates),
            "vix3m_close": pd.Series(np.random.uniform(18, 35, n), index=dates),
        }
        func = registry.get("vix_term_slope")
        result = func(df, external_data=ext_data)
        assert not result.empty
        assert "vix_term_slope" in result.columns
        assert result["vix_term_slope"].notna().sum() > 0

    def test_fallback_to_ext_prefix_keys(self):
        registry = get_registry()
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "close": np.random.uniform(90, 110, n),
                "returns": np.random.normal(0, 0.01, n),
            },
            index=dates,
        )
        ext_data = {
            "ext_vix_close": pd.Series(np.random.uniform(15, 30, n), index=dates),
            "ext_vix3m_close": pd.Series(np.random.uniform(18, 35, n), index=dates),
        }
        func = registry.get("vix_term_slope")
        result = func(df, external_data=ext_data)
        assert not result.empty


class TestComputeFeatures:
    def test_no_inf_in_output(self):
        from algos.common.feature_engine import FeatureConfig, FeatureEngine

        df = _make_df(300, zero_vol_indices=[100, 150, 200])
        fc = FeatureConfig(model_name="svm_optimized", ticker="TEST")
        engine = FeatureEngine()
        aug, fcols = engine.compute_features(df.copy(), fc, external_data={})
        for col in fcols:
            inf_count = np.isinf(aug[col]).sum()
            assert inf_count == 0, f"Found {inf_count} inf in {col}"
