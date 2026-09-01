"""Equity history append helper (Phase 0.2 of the Revision Protocol).

Writes a row per trading-loop event (``start``, ``post_rebalance``, ``eod``)
to ``execution/equity_history.parquet``. ``kill_switch.py`` reads from
this file to compute MTD drawdown and daily moves.

Design choices:

* Append-only. We never edit historical rows; revisions are evidence.
* Crash-tolerant. A corrupt parquet is quarantined rather than crashing
  the live trading loop.
* Idempotent under replay. Re-running the same ``(timestamp, region,
  event)`` triple does NOT duplicate rows. This matters because
  ``main.py`` may restart mid-day.
* Cheap to read. Single file; the equity series for the live universe is
  tiny (~hundreds of rows per month).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


EQUITY_HISTORY_SCHEMA: dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "region": "string",
    "event": "string",          # one of: start, post_rebalance, eod, manual
    "nav_usd": "float64",
    "cash_usd": "float64",
    "gross_exposure": "float64",
    "leverage": "float64",
    "kill_switch_active": "bool",
}


_DEFAULT_PATH = Path(__file__).resolve().parent / "equity_history.parquet"


class EquityHistoryWriter:
    """Append-only writer with idempotency and corruption recovery.

    Usage::

        writer = EquityHistoryWriter()  # uses default path
        writer.append(
            timestamp=datetime.now(timezone.utc),
            region="US",
            nav_usd=portfolio_manager.get_current_net_liquidation(),
            cash_usd=...,
            gross_exposure=...,
            leverage=...,
            event="start",
        )
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def _read_or_empty(self) -> pd.DataFrame:
        if not self.path.exists():
            return self._empty_frame()
        try:
            return pd.read_parquet(self.path)
        except Exception as exc:
            # Corruption: quarantine and start over.
            quarantine = self.path.with_suffix(self.path.suffix + ".corrupt")
            LOGGER.error(
                "equity_history.parquet at %s is corrupt (%s); moving to %s",
                self.path, exc, quarantine,
            )
            try:
                self.path.rename(quarantine)
            except OSError:  # pragma: no cover - filesystem edge case
                pass
            return self._empty_frame()

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in EQUITY_HISTORY_SCHEMA.items()})

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        timestamp: datetime,
        region: str,
        nav_usd: float,
        cash_usd: float,
        gross_exposure: float,
        leverage: float,
        event: str,
        kill_switch_active: bool = False,
    ) -> None:
        """Append one row. Silently skips zero-NAV (likely stale state)."""
        if not nav_usd or nav_usd <= 0:
            LOGGER.warning(
                "equity_history.append called with nav_usd=%r; skipping. "
                "(Likely stale account_values.pkl during startup.)",
                nav_usd,
            )
            return

        ts_utc = self._normalize_ts(timestamp)
        new_row = pd.DataFrame(
            [{
                "timestamp": ts_utc,
                "region": str(region),
                "event": str(event),
                "nav_usd": float(nav_usd),
                "cash_usd": float(cash_usd),
                "gross_exposure": float(gross_exposure),
                "leverage": float(leverage),
                "kill_switch_active": bool(kill_switch_active),
            }]
        )

        existing = self._read_or_empty()

        # Idempotency: if (timestamp, region, event) already present, skip.
        if not existing.empty:
            key_match = (
                (existing["timestamp"] == ts_utc)
                & (existing["region"] == region)
                & (existing["event"] == event)
            )
            if key_match.any():
                LOGGER.debug(
                    "equity_history: skipping duplicate (%s, %s, %s)",
                    ts_utc, region, event,
                )
                return

        combined = pd.concat([existing, new_row], ignore_index=True)
        # Force dtypes consistent with the schema.
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined["region"] = combined["region"].astype("string")
        combined["event"] = combined["event"].astype("string")
        combined["kill_switch_active"] = combined["kill_switch_active"].astype(bool)

        # Atomic write via tmp + replace.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(tmp, index=False)
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ts(ts) -> pd.Timestamp:
        if isinstance(ts, pd.Timestamp):
            return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return pd.Timestamp(ts).tz_convert("UTC")
        return pd.Timestamp(ts, tz="UTC")


# Convenience for ad-hoc callers
def append_equity_event(
    *,
    region: str,
    nav_usd: float,
    cash_usd: float = 0.0,
    gross_exposure: float = 0.0,
    leverage: float = 1.0,
    event: str = "manual",
    kill_switch_active: bool = False,
    timestamp: Optional[datetime] = None,
    path: Optional[Path] = None,
) -> None:
    writer = EquityHistoryWriter(path=path)
    writer.append(
        timestamp=timestamp or datetime.now(timezone.utc),
        region=region,
        nav_usd=nav_usd,
        cash_usd=cash_usd,
        gross_exposure=gross_exposure,
        leverage=leverage,
        event=event,
        kill_switch_active=kill_switch_active,
    )
