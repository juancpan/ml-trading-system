"""Tests for the per-position circuit breaker (Tier B risk-of-ruin protection).

Per REVISION_POLICY.md amendment 2026-07-07: if a single position's realized
loss exceeds 2% of NAV within a rolling 10-trading-day window, auto-flip that
position to buy_and_hold/flat. This is a hard, non-negotiable gate (Tier B),
not advisory.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "execution"))

from position_circuit_breaker import (
    PositionBreakerResult,
    evaluate_position_breakers,
)


def test_position_breaks_when_loss_exceeds_2pct_of_nav():
    """A position with -2.5% of NAV realized loss in the window should break."""
    results = evaluate_position_breakers(
        position_pnl={"GLD": -280.0, "TLT": -50.0, "BBB": +100.0},
        nav=11218.0,
        threshold_pct=0.02,
    )
    assert len(results) == 1
    assert results[0].ticker == "GLD"
    assert results[0].should_break is True
    assert abs(results[0].nav_pct - (-280.0 / 11218.0)) < 1e-9


def test_position_does_not_break_when_loss_below_threshold():
    """A position with -1.5% of NAV should NOT break (below 2% threshold)."""
    results = evaluate_position_breakers(
        position_pnl={"TLT": -168.0, "BBB": +100.0},
        nav=11218.0,
        threshold_pct=0.02,
    )
    assert len(results) == 0


def test_multiple_positions_can_break_simultaneously():
    """If two positions both exceed the threshold, both should break."""
    results = evaluate_position_breakers(
        position_pnl={"GLD": -300.0, "TLT": -250.0, "BBB": +50.0},
        nav=11218.0,
        threshold_pct=0.02,
    )
    assert len(results) == 2
    broken_tickers = {r.ticker for r in results}
    assert broken_tickers == {"GLD", "TLT"}


def test_zero_or_positive_pnl_never_breaks():
    """Positions with zero or positive P&L should never trigger the breaker."""
    results = evaluate_position_breakers(
        position_pnl={"BBB": 0.0, "TELEKOM": +500.0},
        nav=11218.0,
        threshold_pct=0.02,
    )
    assert len(results) == 0


def test_empty_positions_returns_empty():
    """No positions = no breaks."""
    results = evaluate_position_breakers(
        position_pnl={},
        nav=11218.0,
        threshold_pct=0.02,
    )
    assert len(results) == 0


def test_custom_threshold_respected():
    """A 5% threshold should not break a 3% loss."""
    results = evaluate_position_breakers(
        position_pnl={"GLD": -336.0},  # ~3% of 11218
        nav=11218.0,
        threshold_pct=0.05,
    )
    assert len(results) == 0
