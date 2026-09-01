"""Per-position circuit breaker (Tier B risk-of-ruin protection).

If a single position's realized loss exceeds 2% of NAV within a rolling
10-trading-day window, auto-flip that position to buy_and_hold/flat.

This is a HARD, non-negotiable gate per REVISION_POLICY.md Tier B.
It is NOT advisory — it fires automatically, same class as the portfolio
kill-switch. Not gated by cooldown, because it is defense, not a revision.

Interim data note: proper per-position attribution wants Phase 1.4
(decision_price logging, currently deferred to Dec 2026). This module
accepts pre-computed per-position P&L from the caller; the caller is
responsible for computing it from execution journals + mark-to-market
data until Phase 1.4 lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent


@dataclass
class PositionBreakerResult:
    """Result of a per-position circuit breaker check."""

    ticker: str
    realized_loss_usd: float
    nav_pct: float
    should_break: bool


def evaluate_position_breakers(
    position_pnl: Dict[str, float],
    nav: float,
    threshold_pct: float = 0.02,
) -> List[PositionBreakerResult]:
    """Check if any position's realized loss exceeds the threshold.

    Args:
        position_pnl: Dict of {ticker: realized_pnl_in_window_usd}.
            Positive = profit, negative = loss. Only the rolling-window
            P&L should be passed, not cumulative.
        nav: Current Net Liquidation Value in USD.
        threshold_pct: Fraction of NAV. Default 0.02 (2%). A position
            breaks if its realized loss is <= -(threshold_pct * nav).

    Returns:
        List of PositionBreakerResult for positions that should be broken.
        Empty list if none.
    """
    if nav <= 0 or not position_pnl:
        return []

    threshold_usd = threshold_pct * nav
    results: List[PositionBreakerResult] = []

    for ticker, pnl in position_pnl.items():
        if pnl <= -threshold_usd:
            nav_pct = pnl / nav
            results.append(
                PositionBreakerResult(
                    ticker=ticker,
                    realized_loss_usd=pnl,
                    nav_pct=nav_pct,
                    should_break=True,
                )
            )
            LOGGER.warning(
                "POSITION CIRCUIT BREAKER: %s realized loss $%.2f "
                "(%.2f%% of NAV) exceeds threshold %.2f%% ($%.2f). "
                "Auto-flipping to buy_and_hold/flat.",
                ticker,
                pnl,
                nav_pct * 100,
                threshold_pct * 100,
                threshold_usd,
            )

    return results


def write_breaker_sentinels(results: List[PositionBreakerResult]) -> None:
    """Write sentinel files for broken positions.

    Each broken position gets a file ``POSITION_BREAK_{TICKER}`` in
    the execution directory. ``main.py`` should check for these
    before placing trades and skip/reverse the broken ticker.
    """
    for r in results:
        sentinel = _THIS_DIR / f"POSITION_BREAK_{r.ticker}"
        sentinel.write_text(
            f'ticker={r.ticker}\n'
            f'realized_loss_usd={r.realized_loss_usd:.2f}\n'
            f'nav_pct={r.nav_pct:.4f}\n'
            f'threshold=0.02\n'
        )
        LOGGER.info("Wrote position breaker sentinel: %s", sentinel)


def clear_breaker_sentinel(ticker: str) -> None:
    """Remove a position breaker sentinel (manual, after operator review)."""
    sentinel = _THIS_DIR / f"POSITION_BREAK_{ticker}"
    if sentinel.exists():
        sentinel.unlink()
        LOGGER.info("Cleared position breaker sentinel for %s", ticker)


def check_existing_breakers() -> List[str]:
    """Return list of tickers with active breaker sentinels."""
    return [
        f.stem.replace("POSITION_BREAK_", "")
        for f in _THIS_DIR.glob("POSITION_BREAK_*")
    ]
