"""Structured per-ticker per-day signal log (Phase 1.1 of the Revision Protocol).

The live system already logs signals to ``algo_trading.log``, but only as
free-form text. Attribution analysis (Phase 1.3) and shadow-backtest
comparison (Phase 1.2) need structured data: one row per
``(timestamp, region, ticker)`` triple with the raw model output, the
discretized signal, a hash of the features that produced it, and the
target weight at that moment.

Schema mirrors the live trade decision: anything that influenced the
order placed for that ticker on that day should be retrievable from this
file.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


SIGNAL_HISTORY_SCHEMA: dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "region": "string",
    "ticker": "string",
    "model_type": "string",
    "strategy_type": "string",       # ml_signal | buy_and_hold
    "raw_score": "float64",          # pre-thresholding model output
    "signal": "int64",               # -1, 0, +1
    "features_hash": "string",       # sha256(features.tobytes())[:16]
    "n_features": "int64",
    "target_weight": "float64",
    "kelly_fraction_used": "float64",
}


_DEFAULT_PATH = Path(__file__).resolve().parent / "signal_history.parquet"


def hash_features(features: Any) -> tuple[str, int]:
    """Return (hash_hex, n_features) for a features array.

    Accepts numpy arrays, lists, tuples, or anything with ``.tobytes()``.
    Returns ("", 0) on failure rather than raising — telemetry must not
    crash the trading loop.
    """
    try:
        arr = np.asarray(features, dtype=np.float64)
        h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        return h, int(arr.size)
    except Exception as exc:
        LOGGER.debug("hash_features failed: %s", exc)
        return "", 0


class SignalHistoryWriter:
    """Append-only writer for the structured signal log.

    Idempotent on ``(timestamp_date, region, ticker)`` — if a row for the
    same ticker on the same date in the same region exists, the new row
    REPLACES it (last-write-wins). This handles the case where main.py
    restarts mid-day and re-generates a signal.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH

    def append(
        self,
        *,
        timestamp: datetime,
        region: str,
        ticker: str,
        model_type: str,
        strategy_type: str,
        raw_score: float,
        signal: int,
        features_hash: str,
        n_features: int,
        target_weight: float,
        kelly_fraction_used: float,
    ) -> None:
        ts_utc = _to_utc(timestamp)
        new_row = pd.DataFrame([{
            "timestamp": ts_utc,
            "region": str(region),
            "ticker": str(ticker),
            "model_type": str(model_type),
            "strategy_type": str(strategy_type),
            "raw_score": float(raw_score) if raw_score is not None else float("nan"),
            "signal": int(signal),
            "features_hash": str(features_hash),
            "n_features": int(n_features),
            "target_weight": float(target_weight),
            "kelly_fraction_used": float(kelly_fraction_used),
        }])

        try:
            existing = pd.read_parquet(self.path) if self.path.exists() else self._empty_frame()
        except Exception as exc:
            LOGGER.error(
                "signal_history.parquet corrupt at %s: %s; quarantining",
                self.path, exc,
            )
            try:
                self.path.rename(self.path.with_suffix(self.path.suffix + ".corrupt"))
            except OSError:
                pass
            existing = self._empty_frame()

        if not existing.empty:
            existing_ts = pd.to_datetime(existing["timestamp"], utc=True)
            same_day = (
                (existing_ts.dt.date == ts_utc.date())
                & (existing["region"] == region)
                & (existing["ticker"] == ticker)
            )
            if same_day.any():
                existing = existing[~same_day].reset_index(drop=True)

        combined = pd.concat([existing, new_row], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        for str_col in ("region", "ticker", "model_type", "strategy_type", "features_hash"):
            combined[str_col] = combined[str_col].astype("string")

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(tmp, index=False)
        tmp.replace(self.path)

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in SIGNAL_HISTORY_SCHEMA.items()}
        )


def _to_utc(ts) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return pd.Timestamp(ts).tz_convert("UTC")
    return pd.Timestamp(ts, tz="UTC")


# Convenience function for callers who don't want to instantiate the writer.
def log_signal(
    *,
    region: str,
    ticker: str,
    model_type: str,
    strategy_type: str,
    raw_score: float,
    signal: int,
    features: Any = None,
    target_weight: float = 0.0,
    kelly_fraction_used: float = 1.0,
    timestamp: Optional[datetime] = None,
    path: Optional[Path] = None,
) -> None:
    """Single-call entry point for ``strategy_executor`` and friends.

    Never raises.
    """
    try:
        h, n = hash_features(features) if features is not None else ("", 0)
        writer = SignalHistoryWriter(path=path)
        writer.append(
            timestamp=timestamp or datetime.now(timezone.utc),
            region=region,
            ticker=ticker,
            model_type=model_type,
            strategy_type=strategy_type,
            raw_score=raw_score,
            signal=signal,
            features_hash=h,
            n_features=n,
            target_weight=target_weight,
            kelly_fraction_used=kelly_fraction_used,
        )
    except Exception as exc:
        LOGGER.warning("log_signal failed for %s: %s", ticker, exc)
