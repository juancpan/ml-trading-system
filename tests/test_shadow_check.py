"""Tests for shadow_check (Phase 1.2).

We mock the StrategyExecutor builder so these tests run without IBKR,
TensorFlow models, or live data. The point is to verify the comparison
logic, not the model loading path (which is exercised by the live system).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import shadow_check  # noqa: E402
from signal_history import SignalHistoryWriter  # noqa: E402


@pytest.fixture
def populated_signal_history(tmp_path) -> Path:
    p = tmp_path / "signal_history.parquet"
    w = SignalHistoryWriter(path=p)
    ts = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)
    for ticker, model, sig, raw in [
        ("AAA", "gnb", -1, -1.0),
        ("TLT", "lstm", +1, 0.6),
        ("GLD", "lstm", +1, 0.7),
    ]:
        w.append(timestamp=ts, region="US", ticker=ticker,
                 model_type=model, strategy_type="ml_signal",
                 raw_score=raw, signal=sig, features_hash="h1", n_features=5,
                 target_weight=0.1, kelly_fraction_used=1.0)
    return p


def _make_fake_executor(predictions: dict[str, int]):
    """Build a fake StrategyExecutor whose generate_signal returns canned values."""
    fake = mock.MagicMock()
    fake.current_region = "SHADOW"
    fake.data_manager = mock.MagicMock()
    fake.generate_signal.side_effect = lambda symbol: predictions[symbol]
    return fake


class TestShadowCheck:
    def test_all_match_produces_no_divergences(self, tmp_path, populated_signal_history):
        shadow_out = tmp_path / "shadow_history.parquet"

        with mock.patch.object(
            shadow_check, "_build_strategy_executor",
            return_value=_make_fake_executor({"AAA": -1, "TLT": +1, "GLD": +1}),
        ):
            df = shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        assert len(df) == 3
        assert df["divergence_flag"].sum() == 0

    def test_signal_disagreement_flags_divergence(self, tmp_path, populated_signal_history):
        shadow_out = tmp_path / "shadow_history.parquet"

        with mock.patch.object(
            shadow_check, "_build_strategy_executor",
            return_value=_make_fake_executor({"AAA": +1, "TLT": +1, "GLD": +1}),
        ):
            df = shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        assert df["divergence_flag"].sum() == 1
        bk_row = df[df["ticker"] == "AAA"].iloc[0]
        assert bk_row["delta_signal"] == 2  # +1 - (-1) = 2
        assert "signal_diff" in bk_row["divergence_reason"]

    def test_writes_shadow_history(self, tmp_path, populated_signal_history):
        shadow_out = tmp_path / "shadow_history.parquet"

        with mock.patch.object(
            shadow_check, "_build_strategy_executor",
            return_value=_make_fake_executor({"AAA": -1, "TLT": +1, "GLD": +1}),
        ):
            shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        assert shadow_out.exists()
        df = pd.read_parquet(shadow_out)
        assert len(df) == 3
        assert set(df["ticker"].tolist()) == {"AAA", "TLT", "GLD"}

    def test_idempotent_on_replay(self, tmp_path, populated_signal_history):
        """Re-running should NOT duplicate rows for the same (date, region, ticker)."""
        shadow_out = tmp_path / "shadow_history.parquet"

        with mock.patch.object(
            shadow_check, "_build_strategy_executor",
            return_value=_make_fake_executor({"AAA": -1, "TLT": +1, "GLD": +1}),
        ):
            shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )
            shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        df = pd.read_parquet(shadow_out)
        assert len(df) == 3  # not 6

    def test_executor_exception_marks_error(self, tmp_path, populated_signal_history):
        shadow_out = tmp_path / "shadow_history.parquet"
        fake = mock.MagicMock()
        fake.generate_signal.side_effect = RuntimeError("model load failed")
        fake.current_region = "SHADOW"

        with mock.patch.object(
            shadow_check, "_build_strategy_executor", return_value=fake,
        ):
            df = shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        # All three rows should be flagged with error.
        assert df["divergence_flag"].all()
        assert all("model load failed" in e for e in df["error"].tolist())

    def test_no_signal_history_returns_empty(self, tmp_path):
        empty_sig = tmp_path / "nothing.parquet"
        df = shadow_check.run_shadow_for_today(
            signal_history_path=empty_sig,
            shadow_history_path=tmp_path / "shadow.parquet",
        )
        assert df.empty

    def test_executor_unavailable_returns_empty(self, tmp_path, populated_signal_history):
        with mock.patch.object(
            shadow_check, "_build_strategy_executor", return_value=None,
        ):
            df = shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=tmp_path / "shadow.parquet",
            )
        assert df.empty

    def test_fetches_history_before_shadow_signal_generation(self, tmp_path, populated_signal_history):
        shadow_out = tmp_path / "shadow_history.parquet"
        fake = _make_fake_executor({"AAA": -1, "TLT": +1, "GLD": +1})

        with mock.patch.object(shadow_check, "_build_strategy_executor", return_value=fake):
            shadow_check.run_shadow_for_today(
                signal_history_path=populated_signal_history,
                shadow_history_path=shadow_out,
            )

        fetched = [call.args[0] for call in fake.data_manager.fetch_and_store_historical_data.call_args_list]
        assert fetched == ["AAA", "TLT", "GLD"]
