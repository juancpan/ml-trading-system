"""Tests for attribution.py (Phase 1.3)."""

from __future__ import annotations

from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import attribution  # noqa: E402
from equity_history import EquityHistoryWriter  # noqa: E402
from signal_history import SignalHistoryWriter  # noqa: E402


@pytest.fixture
def populated_dirs(tmp_path):
    eq_path = tmp_path / "equity_history.parquet"
    sig_path = tmp_path / "signal_history.parquet"
    journals = tmp_path / "execution_journals"
    journals.mkdir()
    db_path = tmp_path / "attribution.db"

    w = EquityHistoryWriter(path=eq_path)
    base = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
    w.append(timestamp=base, region="US", nav_usd=10_000.0,
             cash_usd=0, gross_exposure=0, leverage=1.0, event="start")
    w.append(timestamp=base + timedelta(hours=4), region="US", nav_usd=10_050.0,
             cash_usd=0, gross_exposure=0, leverage=1.0, event="post_rebalance")
    w.append(timestamp=base + timedelta(hours=8), region="US", nav_usd=10_120.0,
             cash_usd=0, gross_exposure=0, leverage=1.0, event="eod")

    s = SignalHistoryWriter(path=sig_path)
    for tkr, sig in [("AAA", -1), ("TLT", +1), ("GLD", +1)]:
        s.append(timestamp=base, region="US", ticker=tkr,
                 model_type="gnb", strategy_type="ml_signal",
                 raw_score=float(sig), signal=sig, features_hash="h",
                 n_features=5, target_weight=0.1, kelly_fraction_used=1.0)

    return {"eq": eq_path, "sig": sig_path, "journals": journals, "db": db_path,
            "as_of": Date(2026, 5, 16)}


class TestComputeDailyAttribution:
    def test_total_pnl_matches_nav_delta(self, populated_dirs):
        attr = attribution.compute_daily_attribution(
            region="US", as_of_date=populated_dirs["as_of"],
            equity_history_path=populated_dirs["eq"],
            signal_history_path=populated_dirs["sig"],
            execution_journals_dir=populated_dirs["journals"],
        )
        assert attr is not None
        assert attr.total_pnl == pytest.approx(120.0)
        assert attr.nav_open == pytest.approx(10_000.0)
        assert attr.nav_close == pytest.approx(10_120.0)

    def test_returns_none_if_no_equity(self, tmp_path):
        attr = attribution.compute_daily_attribution(
            region="US", as_of_date=Date(2026, 5, 16),
            equity_history_path=tmp_path / "nothing.parquet",
            signal_history_path=tmp_path / "nothing.parquet",
            execution_journals_dir=tmp_path / "nojournals",
        )
        assert attr is None


class TestPersistence:
    def test_round_trip_attribution(self, populated_dirs):
        attr = attribution.compute_daily_attribution(
            region="US", as_of_date=populated_dirs["as_of"],
            equity_history_path=populated_dirs["eq"],
            signal_history_path=populated_dirs["sig"],
            execution_journals_dir=populated_dirs["journals"],
        )
        attribution.persist_daily_attribution(attr, db_path=populated_dirs["db"])
        history = attribution.get_attribution_history(
            region="US", db_path=populated_dirs["db"], limit=10,
        )
        assert len(history) == 1
        row = history.iloc[0]
        assert row["region"] == "US"
        assert float(row["total_pnl"]) == pytest.approx(120.0)

    def test_insert_or_replace(self, populated_dirs):
        attr = attribution.compute_daily_attribution(
            region="US", as_of_date=populated_dirs["as_of"],
            equity_history_path=populated_dirs["eq"],
            signal_history_path=populated_dirs["sig"],
            execution_journals_dir=populated_dirs["journals"],
        )
        attribution.persist_daily_attribution(attr, db_path=populated_dirs["db"])
        # Replace with a different value.
        from dataclasses import replace
        attr2 = replace(attr, total_pnl=999.0)
        attribution.persist_daily_attribution(attr2, db_path=populated_dirs["db"])
        h = attribution.get_attribution_history(region="US", db_path=populated_dirs["db"])
        assert len(h) == 1  # not 2
        assert float(h.iloc[0]["total_pnl"]) == 999.0


class TestHitRates:
    def test_hit_rate_with_synthetic_returns(self, tmp_path):
        """Build a signal history and synthetic next-day returns; verify
        hit/miss counts."""
        sig_path = tmp_path / "signal_history.parquet"
        s = SignalHistoryWriter(path=sig_path)
        base = datetime(2026, 5, 1, tzinfo=timezone.utc)
        # 5 days of (+1, -1, +1, -1, +1) for ticker X. Returns:
        # (+0.01, -0.02, +0.005, +0.001, -0.01) -- so:
        # day1 +1 vs +0.01 = HIT
        # day2 -1 vs -0.02 = HIT
        # day3 +1 vs +0.005 = HIT
        # day4 -1 vs +0.001 = MISS
        # day5 +1 vs -0.01 = MISS
        signals = [+1, -1, +1, -1, +1]
        returns = [+0.01, -0.02, +0.005, +0.001, -0.01]
        for i, sig in enumerate(signals):
            s.append(
                timestamp=base + timedelta(days=i),
                region="US", ticker="X", model_type="gnb",
                strategy_type="ml_signal", raw_score=float(sig), signal=sig,
                features_hash="", n_features=0,
                target_weight=0.1, kelly_fraction_used=1.0,
            )

        synth_returns = pd.DataFrame([
            {"date": (base + timedelta(days=i)).date(),
             "ticker": "X", "next_day_return": r}
            for i, r in enumerate(returns)
        ])

        hr = attribution.compute_model_hit_rates(
            as_of_date=(base + timedelta(days=4)).date(),
            window_days=10,
            signal_history_path=sig_path,
            price_returns=synth_returns,
        )
        assert len(hr) == 1
        r = hr.iloc[0]
        assert int(r["hits"]) == 3
        assert int(r["misses"]) == 2
        assert float(r["hit_rate"]) == pytest.approx(0.6)

    def test_hit_rate_empty_when_no_signals(self, tmp_path):
        hr = attribution.compute_model_hit_rates(
            as_of_date=Date(2026, 5, 16), window_days=20,
            signal_history_path=tmp_path / "nothing.parquet",
        )
        assert hr.empty
