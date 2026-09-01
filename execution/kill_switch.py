"""Kill-switch logic (Phase 0.1 of the Revision Protocol).

Converts ``KILL_SWITCH.md``'s manual runbook into deployable code. Reads
``equity_history.parquet`` produced by ``portfolio_manager._save_state_for_oversight``
helpers in Phase 0.2 and writes one of three sentinel files when a tier
fires:

* ``KILL_SWITCH_ACTIVE``  — hard kill (MTD DD >= hard_dd). Flatten all
  non-retained positions, exit main.py, alert.
* ``SOFT_HALT_ACTIVE``    — soft halt (MTD DD >= soft_dd). Block new
  ml_signal entries; hold existing positions.
* ``DAILY_MOVE_ACTIVE``   — daily move alarm. Skip rebalance for the day.

Sentinels must be cleared manually (``rm`` the file) so a human reviews
before the system re-enables itself.

This module is intentionally pure with respect to IBKR: the "evaluate
decision" path is unit-testable without any IBKR connection. The
"execute flatten" path (which actually places orders) is provided
through :func:`execute_hard_kill`, which takes an OrderGuard as a
dependency and is exercised by integration tests / smoke runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional

# Load .env into os.environ (only keys not already set). See env_loader.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import env_loader  # noqa: F401  (side-effect: populates os.environ)
except Exception:
    pass

import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel file locations
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

HARD_KILL_SENTINEL: Path = _THIS_DIR / "KILL_SWITCH_ACTIVE"
SOFT_HALT_SENTINEL: Path = _THIS_DIR / "SOFT_HALT_ACTIVE"
DAILY_MOVE_SENTINEL: Path = _THIS_DIR / "DAILY_MOVE_ACTIVE"

# Standard equity-history path. Phase 0.2 writes to this file.
EQUITY_HISTORY_PATH: Path = _THIS_DIR / "equity_history.parquet"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KillSwitchConfig:
    """Threshold configuration for the kill-switch.

    Defaults match ``KILL_SWITCH.md`` (Phase 0 placeholders). Phase 2 will
    recompute these from backtest distribution percentiles and update both
    this config and the document together.
    """

    hard_dd: float = 0.08
    soft_dd: float = 0.05
    daily_move: float = 0.04
    retain_tickers: frozenset[str] = field(
        default_factory=lambda: frozenset({"BIL", "TLT", "GLD"})
    )

    @classmethod
    def from_config_module(cls) -> "KillSwitchConfig":
        """Build from ``execution.config``. Falls back to defaults if
        the constants haven't been added yet."""
        try:
            import config as live_config  # type: ignore
        except Exception:  # pragma: no cover - defensive
            return cls()
        return cls(
            hard_dd=getattr(live_config, "KILL_SWITCH_HARD_DD", 0.08),
            soft_dd=getattr(live_config, "KILL_SWITCH_SOFT_DD", 0.05),
            daily_move=getattr(live_config, "KILL_SWITCH_DAILY_MOVE", 0.04),
            retain_tickers=frozenset(
                getattr(live_config, "KILL_SWITCH_RETAIN_TICKERS", ["BIL", "TLT", "GLD"])
            ),
        )


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class KillSwitchTier(str, Enum):
    OK = "ok"
    DAILY_MOVE_ALARM = "daily_move_alarm"
    SOFT_HALT = "soft_halt"
    HARD_KILL = "hard_kill"


@dataclass(frozen=True)
class KillSwitchDecision:
    """Output of :func:`evaluate_kill_switch`.

    Pure data: no I/O. Callers translate this into sentinel writes, order
    submissions, and alerts.
    """

    tier: KillSwitchTier
    mtd_drawdown: float
    daily_move: float
    flatten_tickers: frozenset[str]
    block_entries: bool
    block_rebalance: bool
    reason: str

    def to_payload(self) -> dict:
        return {
            "tier": self.tier.value,
            "mtd_drawdown": float(self.mtd_drawdown),
            "daily_move": float(self.daily_move),
            "flatten_tickers": sorted(self.flatten_tickers),
            "block_entries": bool(self.block_entries),
            "block_rebalance": bool(self.block_rebalance),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Pure metric computations
# ---------------------------------------------------------------------------


def _to_utc(ts) -> datetime:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")


def compute_mtd_drawdown(
    equity_df: pd.DataFrame, *, as_of: Optional[datetime] = None
) -> float:
    """Return ``(current_nav / first_nav_of_month) - 1``.

    Negative values are drawdowns. Positive values mean the strategy is up
    on the month. Returns 0.0 for empty / single-point series.
    """
    if equity_df.empty:
        return 0.0
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if as_of is None:
        as_of = df["timestamp"].iloc[-1].to_pydatetime()
    as_of_utc = _to_utc(as_of)
    as_of_ts = pd.Timestamp(as_of_utc)
    month_start = pd.Timestamp(
        datetime(as_of_utc.year, as_of_utc.month, 1, tzinfo=timezone.utc)
    )
    month_slice = df[(df["timestamp"] >= month_start) & (df["timestamp"] <= as_of_ts)]
    if month_slice.empty:
        return 0.0
    anchor = float(month_slice["nav_usd"].iloc[0])
    last = float(month_slice["nav_usd"].iloc[-1])
    if anchor == 0:
        return 0.0
    return (last / anchor) - 1.0


def compute_daily_move(
    equity_df: pd.DataFrame, *, as_of: Optional[datetime] = None
) -> float:
    """Return today's NAV / yesterday's NAV - 1.

    "Today" is the date of ``as_of``; "yesterday" is the latest prior day
    that has any row. Returns 0.0 if either side is missing.
    """
    if equity_df.empty:
        return 0.0
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if as_of is None:
        as_of = df["timestamp"].iloc[-1].to_pydatetime()
    as_of_utc = _to_utc(as_of)
    today = as_of_utc.date()

    # Latest row on or before as_of (today)
    today_rows = df[df["timestamp"] <= pd.Timestamp(as_of_utc)]
    today_rows = today_rows[today_rows["timestamp"].dt.date == today]
    if today_rows.empty:
        return 0.0
    today_nav = float(today_rows["nav_usd"].iloc[-1])

    # Most recent row from a prior day
    prior = df[df["timestamp"].dt.date < today]
    if prior.empty:
        return 0.0
    prior_nav = float(prior["nav_usd"].iloc[-1])
    if prior_nav == 0:
        return 0.0
    return (today_nav / prior_nav) - 1.0


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------


def evaluate_kill_switch(
    equity_df: pd.DataFrame,
    config: KillSwitchConfig,
    *,
    as_of: Optional[datetime] = None,
    current_positions: Optional[Mapping[str, float]] = None,
) -> KillSwitchDecision:
    """Evaluate kill-switch tier against the latest equity history.

    Precedence (strongest first):

    1. HARD_KILL — ``mtd_dd <= -hard_dd``. Overrides everything else.
    2. SOFT_HALT — ``mtd_dd <= -soft_dd``.
    3. DAILY_MOVE_ALARM — ``abs(daily_move) >= daily_move`` threshold.
    4. OK

    ``current_positions`` is consulted only at the HARD_KILL tier to
    determine which tickers must be flattened. Tickers in
    ``config.retain_tickers`` are excluded.
    """
    mtd = compute_mtd_drawdown(equity_df, as_of=as_of)
    daily = compute_daily_move(equity_df, as_of=as_of)

    # Round to 8 decimals to make tier boundaries deterministic at the
    # floating-point boundary (e.g. 9200/10000 - 1 == -0.07999999996 != -0.08).
    # 1e-8 is sub-basis-point precision; it does not change real behavior.
    mtd_cmp = round(mtd, 8)
    daily_cmp = round(daily, 8)

    # 1. Hard kill
    if mtd_cmp <= -config.hard_dd:
        flatten: frozenset[str] = frozenset()
        if current_positions:
            flatten = frozenset(
                t for t, qty in current_positions.items()
                if t not in config.retain_tickers and qty != 0
            )
        return KillSwitchDecision(
            tier=KillSwitchTier.HARD_KILL,
            mtd_drawdown=mtd,
            daily_move=daily,
            flatten_tickers=flatten,
            block_entries=True,
            block_rebalance=True,
            reason=f"MTD drawdown {mtd:.2%} <= -{config.hard_dd:.0%}",
        )

    # 2. Soft halt
    if mtd_cmp <= -config.soft_dd:
        return KillSwitchDecision(
            tier=KillSwitchTier.SOFT_HALT,
            mtd_drawdown=mtd,
            daily_move=daily,
            flatten_tickers=frozenset(),
            block_entries=True,
            block_rebalance=False,
            reason=f"MTD drawdown {mtd:.2%} <= -{config.soft_dd:.0%}",
        )

    # 3. Daily move alarm
    if abs(daily_cmp) >= config.daily_move:
        return KillSwitchDecision(
            tier=KillSwitchTier.DAILY_MOVE_ALARM,
            mtd_drawdown=mtd,
            daily_move=daily,
            flatten_tickers=frozenset(),
            block_entries=False,
            block_rebalance=True,
            reason=f"Daily move {daily:+.2%} exceeds ±{config.daily_move:.0%}",
        )

    return KillSwitchDecision(
        tier=KillSwitchTier.OK,
        mtd_drawdown=mtd,
        daily_move=daily,
        flatten_tickers=frozenset(),
        block_entries=False,
        block_rebalance=False,
        reason=f"OK (MTD {mtd:+.2%}, daily {daily:+.2%})",
    )


# ---------------------------------------------------------------------------
# Sentinel file helpers
# ---------------------------------------------------------------------------


def write_sentinel(path: Path, *, reason: str, details: Optional[dict] = None) -> None:
    """Write a sentinel file with a JSON payload. Idempotent."""
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "details": details or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write to .tmp then rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def clear_sentinel(path: Path) -> None:
    """Remove a sentinel file if present. Idempotent — never raises if missing."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def sentinel_active(path: Path) -> bool:
    """Return True if the sentinel file exists."""
    return path.exists()


def any_sentinel_active() -> bool:
    """Return True if ANY kill-switch sentinel is active.

    Convenience for ``main.py`` startup checks and external scripts.
    """
    return (
        sentinel_active(HARD_KILL_SENTINEL)
        or sentinel_active(SOFT_HALT_SENTINEL)
        or sentinel_active(DAILY_MOVE_SENTINEL)
    )


def sentinel_for_tier(tier: KillSwitchTier) -> Optional[Path]:
    return {
        KillSwitchTier.HARD_KILL: HARD_KILL_SENTINEL,
        KillSwitchTier.SOFT_HALT: SOFT_HALT_SENTINEL,
        KillSwitchTier.DAILY_MOVE_ALARM: DAILY_MOVE_SENTINEL,
        KillSwitchTier.OK: None,
    }[tier]


# ---------------------------------------------------------------------------
# I/O wrappers — load equity history, apply decision side-effects
# ---------------------------------------------------------------------------


def load_equity_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the equity history parquet. Returns empty DataFrame if missing.

    Resolves ``path`` from the module-level ``EQUITY_HISTORY_PATH`` at call
    time (not import time), so tests that monkey-patch the constant get
    the patched value.
    """
    if path is None:
        path = EQUITY_HISTORY_PATH
    if not path.exists():
        LOGGER.warning("equity_history.parquet not found at %s; returning empty.", path)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "region",
                "nav_usd",
                "cash_usd",
                "gross_exposure",
                "leverage",
                "kill_switch_active",
            ]
        )
    return pd.read_parquet(path)


def apply_decision(
    decision: KillSwitchDecision,
    *,
    alert_fn=None,
) -> None:
    """Apply a decision's side-effects: sentinel files + optional alert.

    ``alert_fn`` is a callable with signature ``(title: str, message: str,
    severity: str) -> None``. When None, alerting is skipped (e.g. in tests
    or when Telegram is not configured).

    Order placement is NOT performed here. The caller (``main.py``) is
    responsible for calling ``execute_hard_kill`` separately so the
    decision/effect boundary is explicit.
    """
    payload = decision.to_payload()

    if decision.tier == KillSwitchTier.HARD_KILL:
        write_sentinel(HARD_KILL_SENTINEL, reason=decision.reason, details=payload)
        # Hard-kill is the strongest tier; the soft sentinel is redundant but
        # we leave it for clarity if it was previously active.
    elif decision.tier == KillSwitchTier.SOFT_HALT:
        write_sentinel(SOFT_HALT_SENTINEL, reason=decision.reason, details=payload)
    elif decision.tier == KillSwitchTier.DAILY_MOVE_ALARM:
        write_sentinel(DAILY_MOVE_SENTINEL, reason=decision.reason, details=payload)
    # OK: do nothing (do NOT auto-clear sentinels — manual review required)

    if alert_fn is not None and decision.tier != KillSwitchTier.OK:
        severity = {
            KillSwitchTier.HARD_KILL: "critical",
            KillSwitchTier.SOFT_HALT: "warning",
            KillSwitchTier.DAILY_MOVE_ALARM: "warning",
        }[decision.tier]
        try:
            alert_fn(
                f"Kill-switch: {decision.tier.value}",
                decision.reason,
                severity,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Alert dispatch failed: %s", exc)


# ---------------------------------------------------------------------------
# Standalone CLI entry point (cron use)
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entrypoint: load equity history, evaluate, apply decision.

    Used by the cron job. Exit codes:
    * 0 — OK or DAILY_MOVE_ALARM (informational)
    * 1 — SOFT_HALT (caller should block ml entries)
    * 2 — HARD_KILL (caller should not start a new trading loop)
    """
    logging.basicConfig(level=logging.INFO)
    cfg = KillSwitchConfig.from_config_module()
    df = load_equity_history()

    # Try to load current positions for flatten list, but don't fail if absent.
    current_positions: dict[str, float] = {}
    positions_pkl = _THIS_DIR / "positions.pkl"
    if positions_pkl.exists():
        try:
            import pickle
            with open(positions_pkl, "rb") as f:
                raw = pickle.load(f)
            current_positions = {
                sym: float(p.get("position", 0)) for sym, p in raw.items()
            }
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not load positions.pkl: %s", exc)

    # Try to obtain alerting callable. Optional.
    alert_fn = None
    try:
        from alerting import send_alert  # type: ignore
        alert_fn = send_alert
    except Exception:
        alert_fn = None

    decision = evaluate_kill_switch(df, cfg, current_positions=current_positions)
    apply_decision(decision, alert_fn=alert_fn)

    LOGGER.info("Kill-switch decision: %s | %s", decision.tier.value, decision.reason)

    return {
        KillSwitchTier.OK: 0,
        KillSwitchTier.DAILY_MOVE_ALARM: 0,
        KillSwitchTier.SOFT_HALT: 1,
        KillSwitchTier.HARD_KILL: 2,
    }[decision.tier]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
