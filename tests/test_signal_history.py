"""Tests for signal_history (Phase 1.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_history import (  # noqa: E402
    SignalHistoryWriter,
    SIGNAL_HISTORY_SCHEMA,
    hash_features,
    log_signal,
)


@pytest.fixture
def writer_path(tmp_path) -> Path:
    return tmp_path / "signal_history.parquet"


class TestHashFeatures:
    def test_stable_hash_for_same_array(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        h1, n1 = hash_features(a)
        h2, n2 = hash_features(a.copy())
        assert h1 == h2 != ""
        assert n1 == n2 == 5

    def test_different_arrays_differ(self):
        h1, _ = hash_features(np.array([1.0, 2.0]))
        h2, _ = hash_features(np.array([1.0, 2.000001]))
        assert h1 != h2

    def test_handles_2d(self):
        h, n = hash_features(np.zeros((4, 3)))
        assert n == 12

    def test_graceful_failure_returns_empty(self):
        # Something that cannot be coerced.
        h, n = hash_features(object())
        assert h == "" and n == 0


class TestSignalHistoryWriter:
    def test_append_creates_file(self, writer_path):
        w = SignalHistoryWriter(path=writer_path)
        w.append(
            timestamp=datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc),
            region="US", ticker="AAA",
            model_type="gnb", strategy_type="ml_signal",
            raw_score=-1.0, signal=-1,
            features_hash="abc123", n_features=5,
            target_weight=0.045, kelly_fraction_used=1.0,
        )
        df = pd.read_parquet(writer_path)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAA"
        assert df.iloc[0]["signal"] == -1

    def test_schema_columns_present(self, writer_path):
        w = SignalHistoryWriter(path=writer_path)
        w.append(
            timestamp=datetime.now(timezone.utc),
            region="US", ticker="X",
            model_type="gnb", strategy_type="ml_signal",
            raw_score=0.5, signal=1, features_hash="", n_features=0,
            target_weight=0.1, kelly_fraction_used=1.0,
        )
        df = pd.read_parquet(writer_path)
        assert set(SIGNAL_HISTORY_SCHEMA.keys()).issubset(set(df.columns))

    def test_same_day_same_ticker_replaces(self, writer_path):
        """Idempotency: re-running signal generation on the same day for the
        same ticker should replace, not duplicate."""
        w = SignalHistoryWriter(path=writer_path)
        ts1 = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
        w.append(timestamp=ts1, region="US", ticker="AAA",
                 model_type="gnb", strategy_type="ml_signal",
                 raw_score=-1.0, signal=-1, features_hash="h1", n_features=5,
                 target_weight=0.045, kelly_fraction_used=1.0)
        w.append(timestamp=ts2, region="US", ticker="AAA",
                 model_type="gnb", strategy_type="ml_signal",
                 raw_score=+1.0, signal=+1, features_hash="h2", n_features=5,
                 target_weight=0.045, kelly_fraction_used=1.0)
        df = pd.read_parquet(writer_path)
        assert len(df) == 1
        assert df.iloc[0]["signal"] == +1  # last write wins
        assert df.iloc[0]["features_hash"] == "h2"

    def test_different_days_accumulate(self, writer_path):
        w = SignalHistoryWriter(path=writer_path)
        for day_offset in range(3):
            ts = datetime(2026, 5, 14 + day_offset, 14, 0, tzinfo=timezone.utc)
            w.append(timestamp=ts, region="US", ticker="AAA",
                     model_type="gnb", strategy_type="ml_signal",
                     raw_score=0.5, signal=1, features_hash="", n_features=0,
                     target_weight=0.045, kelly_fraction_used=1.0)
        df = pd.read_parquet(writer_path)
        assert len(df) == 3

    def test_different_tickers_same_day_accumulate(self, writer_path):
        w = SignalHistoryWriter(path=writer_path)
        ts = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        for tkr in ["AAA", "TLT", "GLD"]:
            w.append(timestamp=ts, region="US", ticker=tkr,
                     model_type="gnb", strategy_type="ml_signal",
                     raw_score=0.5, signal=1, features_hash="", n_features=0,
                     target_weight=0.1, kelly_fraction_used=1.0)
        df = pd.read_parquet(writer_path)
        assert len(df) == 3
        assert set(df["ticker"].tolist()) == {"AAA", "TLT", "GLD"}


class TestLogSignal:
    def test_convenience_function(self, writer_path):
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        log_signal(
            region="US", ticker="AAA",
            model_type="gnb", strategy_type="ml_signal",
            raw_score=-0.7, signal=-1, features=features,
            target_weight=0.045, kelly_fraction_used=1.0,
            path=writer_path,
        )
        df = pd.read_parquet(writer_path)
        assert len(df) == 1
        assert df.iloc[0]["features_hash"] != ""
        assert df.iloc[0]["n_features"] == 5

    def test_never_raises_on_bad_input(self, writer_path):
        # No features, weird ticker, NaN score — should not raise.
        log_signal(
            region="US", ticker="???",
            model_type="unknown", strategy_type="ml_signal",
            raw_score=float("nan"), signal=0,
            features=None,
            target_weight=0.0, kelly_fraction_used=1.0,
            path=writer_path,
        )
        # And the row should be in the file.
        df = pd.read_parquet(writer_path)
        assert len(df) == 1
