"""Runtime adapters wrapping kill_switch + equity_history for main.py.

This module exists to keep the (large, fragile) ``main.py`` change-set
small and reversible. ``main.py`` imports a few helpers from here; if the
revision protocol is later removed, only the three call-sites need to
change, not the kill-switch logic.

Public surface:

* :func:`record_equity_event` — write a row to equity_history.parquet
  using the current portfolio_manager state. Safe to call from any
  trading-loop checkpoint.
* :func:`evaluate_and_apply` — run kill_switch.evaluate against the
  latest equity_history; write sentinels; fire alerts. Returns the
  decision (or ``None`` if disabled).
* :func:`is_soft_halt_active` — convenience for ``strategy_executor`` to
  query before generating ml_signal entries.
* :func:`is_hard_kill_active` — convenience for the top of main.py's
  loop to exit cleanly.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

from kill_switch import (
    HARD_KILL_SENTINEL,
    SOFT_HALT_SENTINEL,
    DAILY_MOVE_SENTINEL,
    KillSwitchConfig,
    KillSwitchDecision,
    KillSwitchTier,
    apply_decision,
    evaluate_kill_switch,
    load_equity_history,
    sentinel_active,
)
from equity_history import EquityHistoryWriter

try:
    from alerting import send_kill_switch_alert
except Exception:  # pragma: no cover - alerting always importable
    send_kill_switch_alert = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _kill_switch_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "KILL_SWITCH_ENABLED", True))
    except Exception:
        return True


def record_equity_event(
    portfolio_manager,
    *,
    region: str,
    event: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Append one row to equity_history.parquet from current pm state.

    Never raises — the trading loop must not crash because of telemetry.
    """
    log = logger or LOGGER
    try:
        nav = float(portfolio_manager.get_current_net_liquidation() or 0.0)
        # Cash (USD) approximation: use TotalCashValue if present.
        cash = 0.0
        try:
            cv = portfolio_manager.account_values.get("TotalCashValue") or {}
            if isinstance(cv, dict):
                cash = float(cv.get("value", 0.0))
        except Exception:
            cash = 0.0
        gross = float(portfolio_manager.account_values.get("GrossPositionValue", {}).get("value", 0.0)) \
            if isinstance(portfolio_manager.account_values.get("GrossPositionValue"), dict) else 0.0
        leverage = (gross / nav) if nav > 0 else 0.0

        writer = EquityHistoryWriter()
        writer.append(
            timestamp=datetime.now(timezone.utc),
            region=region,
            nav_usd=nav,
            cash_usd=cash,
            gross_exposure=gross,
            leverage=leverage,
            event=event,
            kill_switch_active=is_any_sentinel_active(),
        )
        log.info(
            "equity_history append: region=%s event=%s nav=$%.2f leverage=%.2fx",
            region, event, nav, leverage,
        )
    except Exception as exc:
        log.warning("record_equity_event failed (event=%s): %s\n%s",
                    event, exc, traceback.format_exc())


def evaluate_and_apply(
    portfolio_manager,
    *,
    region: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[KillSwitchDecision]:
    """Evaluate kill-switch and apply side-effects. Returns the decision.

    Returns None if the kill-switch is disabled by config.

    Never raises.
    """
    log = logger or LOGGER
    if not _kill_switch_enabled():
        log.info("Kill-switch disabled by config (KILL_SWITCH_ENABLED=False).")
        return None

    try:
        cfg = KillSwitchConfig.from_config_module()
        df = load_equity_history()

        # Current positions for the flatten list.
        current_positions: dict[str, float] = {}
        try:
            for symbol, pos in portfolio_manager.current_positions.items():
                qty = float(pos.get("position", 0))
                if qty:
                    current_positions[symbol] = qty
        except Exception:
            pass

        decision = evaluate_kill_switch(df, cfg, current_positions=current_positions)
        apply_decision(decision, alert_fn=_alert_fn_or_none())

        log.info(
            "kill_switch: tier=%s mtd=%+.2f%% daily=%+.2f%% reason=%s",
            decision.tier.value,
            decision.mtd_drawdown * 100,
            decision.daily_move * 100,
            decision.reason,
        )
        return decision
    except Exception as exc:
        log.error("evaluate_and_apply failed: %s\n%s", exc, traceback.format_exc())
        return None


def _alert_fn_or_none():
    if send_kill_switch_alert is None:
        return None

    def _fn(title: str, message: str, severity: str) -> None:
        try:
            from alerting import send_alert
            send_alert(title, message, severity)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Alert dispatch failed: %s", exc)

    return _fn


def is_hard_kill_active() -> bool:
    return sentinel_active(HARD_KILL_SENTINEL)


def is_soft_halt_active() -> bool:
    return sentinel_active(SOFT_HALT_SENTINEL)


def is_daily_move_active() -> bool:
    return sentinel_active(DAILY_MOVE_SENTINEL)


def is_any_sentinel_active() -> bool:
    return is_hard_kill_active() or is_soft_halt_active() or is_daily_move_active()


__all__ = [
    "record_equity_event",
    "evaluate_and_apply",
    "is_hard_kill_active",
    "is_soft_halt_active",
    "is_daily_move_active",
    "is_any_sentinel_active",
    "KillSwitchDecision",
    "KillSwitchTier",
]
