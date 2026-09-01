"""Tests for the kill-switch pure logic (Phase 0.1).

These tests cover decision logic and file-side-effects only. The IBKR-side
flatten path is not covered here; it's covered by a mock-based integration
test in `tests/test_kill_switch_integration.py` (to be added in a later
iteration).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from kill_switch import (  # noqa: E402 - test imports module under test
    KillSwitchConfig,
    KillSwitchTier,
    KillSwitchDecision,
    compute_mtd_drawdown,
    compute_daily_move,
    evaluate_kill_switch,
    write_sentinel,
    clear_sentinel,
    sentinel_active,
    HARD_KILL_SENTINEL,
    SOFT_HALT_SENTINEL,
    DAILY_MOVE_SENTINEL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> KillSwitchConfig:
    """KILL_SWITCH.md placeholder thresholds."""
    return KillSwitchConfig(
        hard_dd=0.08,
        soft_dd=0.05,
        daily_move=0.04,
        retain_tickers=frozenset({"BIL", "TLT", "GLD"}),
    )


@pytest.fixture
def synthetic_equity_series():
    """Builds a daily equity DataFrame for a synthetic month."""

    def _build(values_per_day: list[float], start: datetime | None = None) -> pd.DataFrame:
        if start is None:
            start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        rows = []
        for i, nav in enumerate(values_per_day):
            ts = start + timedelta(days=i)
            rows.append(
                {
                    "timestamp": ts,
                    "region": "US",
                    "nav_usd": float(nav),
                    "cash_usd": 0.0,
                    "gross_exposure": float(nav) * 1.3,
                    "leverage": 1.3,
                    "kill_switch_active": False,
                }
            )
        return pd.DataFrame(rows)

    return _build


# ---------------------------------------------------------------------------
# compute_mtd_drawdown
# ---------------------------------------------------------------------------


class TestComputeMtdDrawdown:
    def test_flat_returns_zero(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000] * 10)
        assert compute_mtd_drawdown(df, as_of=df["timestamp"].iloc[-1]) == pytest.approx(0.0)

    def test_pure_decline_reports_negative_dd(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_500, 9_000])
        # MTD drawdown is (last / month_start - 1).
        dd = compute_mtd_drawdown(df, as_of=df["timestamp"].iloc[-1])
        assert dd == pytest.approx((9_000 / 10_000) - 1.0)  # -0.10

    def test_recovery_after_decline(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_000, 10_200])
        dd = compute_mtd_drawdown(df, as_of=df["timestamp"].iloc[-1])
        assert dd == pytest.approx((10_200 / 10_000) - 1.0)  # +0.02 (not a drawdown)

    def test_uses_first_nav_of_month_as_anchor(self, synthetic_equity_series):
        # Series spans two months. anchor must reset at month boundary.
        start_april = datetime(2026, 4, 28, tzinfo=timezone.utc)
        df = synthetic_equity_series([10_000, 10_500, 11_000, 10_000, 9_000], start=start_april)
        # rows: Apr 28 (10000), Apr 29 (10500), Apr 30 (11000), May 1 (10000), May 2 (9000).
        # MTD as of May 2: anchor = May 1's 10_000, last = 9000 → -10%.
        dd = compute_mtd_drawdown(df, as_of=df["timestamp"].iloc[-1])
        assert dd == pytest.approx(-0.10)

    def test_empty_series_returns_zero(self):
        df = pd.DataFrame(
            columns=["timestamp", "region", "nav_usd", "cash_usd", "gross_exposure", "leverage"]
        )
        assert compute_mtd_drawdown(df, as_of=datetime.now(timezone.utc)) == 0.0


# ---------------------------------------------------------------------------
# compute_daily_move
# ---------------------------------------------------------------------------


class TestComputeDailyMove:
    def test_zero_move(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 10_000])
        assert compute_daily_move(df, as_of=df["timestamp"].iloc[-1]) == pytest.approx(0.0)

    def test_positive_4_percent(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 10_400])
        move = compute_daily_move(df, as_of=df["timestamp"].iloc[-1])
        assert move == pytest.approx(0.04)

    def test_negative_4_percent(self, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_600])
        move = compute_daily_move(df, as_of=df["timestamp"].iloc[-1])
        assert move == pytest.approx(-0.04)

    def test_uses_prior_day_close(self, synthetic_equity_series):
        # When today has multiple rows, use yesterday's last vs. today's last.
        start = datetime(2026, 5, 14, tzinfo=timezone.utc)
        df = pd.DataFrame(
            [
                # day 1 — three intraday rows ending at 10_000
                {"timestamp": start, "region": "US", "nav_usd": 10_200,
                 "cash_usd": 0, "gross_exposure": 0, "leverage": 1.0, "kill_switch_active": False},
                {"timestamp": start + timedelta(hours=4), "region": "US", "nav_usd": 10_100,
                 "cash_usd": 0, "gross_exposure": 0, "leverage": 1.0, "kill_switch_active": False},
                {"timestamp": start + timedelta(hours=8), "region": "US", "nav_usd": 10_000,
                 "cash_usd": 0, "gross_exposure": 0, "leverage": 1.0, "kill_switch_active": False},
                # day 2 — last row at 9_500
                {"timestamp": start + timedelta(days=1), "region": "US", "nav_usd": 9_500,
                 "cash_usd": 0, "gross_exposure": 0, "leverage": 1.0, "kill_switch_active": False},
            ]
        )
        move = compute_daily_move(df, as_of=df["timestamp"].iloc[-1])
        # (9500 / 10000) - 1 = -0.05
        assert move == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# evaluate_kill_switch — the decision function
# ---------------------------------------------------------------------------


class TestEvaluateKillSwitch:
    def test_normal_returns_ok_tier(self, default_config, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 10_050, 10_020])
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.OK
        assert decision.flatten_tickers == frozenset()
        assert decision.block_entries is False

    def test_soft_halt_at_exactly_5_percent_dd(self, default_config, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_500])  # MTD -5.0%
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.SOFT_HALT
        assert decision.block_entries is True
        assert decision.flatten_tickers == frozenset()
        # Soft halt is reversible — block entries but hold positions.

    def test_hard_kill_at_exactly_8_percent_dd(self, default_config, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_200])  # MTD -8.0%
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.HARD_KILL
        assert decision.block_entries is True
        # Must flatten everything except retain set.
        assert "BIL" not in decision.flatten_tickers
        assert "TLT" not in decision.flatten_tickers
        assert "GLD" not in decision.flatten_tickers

    def test_hard_kill_dominates_soft_halt(self, default_config, synthetic_equity_series):
        # If both fire, hard kill wins.
        df = synthetic_equity_series([10_000, 9_100])  # -9%
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.HARD_KILL

    def test_daily_move_alarm_independent_of_dd(self, default_config, synthetic_equity_series):
        # Big single-day move but no MTD drawdown.
        df = synthetic_equity_series([10_000, 10_500])  # +5% in one day
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.DAILY_MOVE_ALARM
        # No flatten — just block rebalance.
        assert decision.flatten_tickers == frozenset()
        assert decision.block_rebalance is True

    def test_hard_kill_overrides_daily_move_alarm(self, default_config, synthetic_equity_series):
        # Both conditions met: hard kill takes precedence.
        df = synthetic_equity_series([10_000, 9_100])  # -9% MTD AND -9% daily
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        assert decision.tier == KillSwitchTier.HARD_KILL

    def test_retain_tickers_flatten_correctly(self, default_config, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_200])
        current_positions = {"BIL": 100, "TLT": 50, "GLD": 30, "AAA": 10, "GLX": 5, "TUV.TO": 8}
        decision = evaluate_kill_switch(
            df,
            default_config,
            as_of=df["timestamp"].iloc[-1],
            current_positions=current_positions,
        )
        assert decision.tier == KillSwitchTier.HARD_KILL
        assert decision.flatten_tickers == frozenset({"AAA", "GLX", "TUV.TO"})

    def test_custom_retain_set(self, synthetic_equity_series):
        custom_cfg = KillSwitchConfig(
            hard_dd=0.08,
            soft_dd=0.05,
            daily_move=0.04,
            retain_tickers=frozenset({"BIL"}),
        )
        df = synthetic_equity_series([10_000, 9_100])
        current_positions = {"BIL": 1, "TLT": 1, "GLD": 1}
        decision = evaluate_kill_switch(
            df, custom_cfg, as_of=df["timestamp"].iloc[-1],
            current_positions=current_positions,
        )
        assert decision.flatten_tickers == frozenset({"TLT", "GLD"})


# ---------------------------------------------------------------------------
# Sentinel files
# ---------------------------------------------------------------------------


class TestSentinels:
    def test_write_then_detect(self, tmp_path):
        sentinel = tmp_path / "TEST_ACTIVE"
        assert sentinel_active(sentinel) is False
        write_sentinel(sentinel, reason="test", details={"tier": "hard_kill"})
        assert sentinel_active(sentinel) is True

    def test_write_includes_payload(self, tmp_path):
        sentinel = tmp_path / "TEST_ACTIVE"
        write_sentinel(sentinel, reason="MTD -9%", details={"tier": "hard_kill", "nav": 9100.0})
        payload = json.loads(sentinel.read_text())
        assert payload["reason"] == "MTD -9%"
        assert payload["details"]["tier"] == "hard_kill"
        assert "written_at" in payload

    def test_clear_removes_file(self, tmp_path):
        sentinel = tmp_path / "TEST_ACTIVE"
        write_sentinel(sentinel, reason="x", details={})
        clear_sentinel(sentinel)
        assert sentinel_active(sentinel) is False

    def test_clear_is_idempotent(self, tmp_path):
        sentinel = tmp_path / "DOES_NOT_EXIST"
        # Should not raise
        clear_sentinel(sentinel)
        assert sentinel_active(sentinel) is False

    def test_sentinel_constants_distinct(self):
        # Three sentinels must have different filenames.
        names = {HARD_KILL_SENTINEL.name, SOFT_HALT_SENTINEL.name, DAILY_MOVE_SENTINEL.name}
        assert len(names) == 3


# ---------------------------------------------------------------------------
# Decision → sentinel writing integration
# ---------------------------------------------------------------------------


class TestDecisionWriting:
    """Validates that a Decision can serialize cleanly to a sentinel payload."""

    def test_decision_is_dataclass_with_required_fields(self, default_config, synthetic_equity_series):
        df = synthetic_equity_series([10_000, 9_200])
        decision = evaluate_kill_switch(df, default_config, as_of=df["timestamp"].iloc[-1])
        # The fields needed by the alerting and runtime layers.
        assert hasattr(decision, "tier")
        assert hasattr(decision, "mtd_drawdown")
        assert hasattr(decision, "daily_move")
        assert hasattr(decision, "flatten_tickers")
        assert hasattr(decision, "block_entries")
        assert hasattr(decision, "block_rebalance")
        assert hasattr(decision, "reason")
        # `reason` must be a non-empty string when tier != OK.
        assert isinstance(decision.reason, str) and decision.reason
