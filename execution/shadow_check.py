"""Shadow backtest / live-vs-backtest reconciliation (Phase 1.2).

For each ticker in the live universe, re-run signal generation using the
same model artifacts and feature pipeline the live system uses, then
compare the resulting signal to whatever was recorded in
``signal_history.parquet``. Differences flag suspected drift.

Decision to NOT replicate ``OptimizedBacktester``:
    The full backtest runner is batch-oriented and slow. Its purpose is
    historical evaluation. The shadow's purpose is parity verification:
    "did the live process produce the same signal it would have produced
    if I re-ran the same code on the same data five minutes later?"
    Reusing the live loaders catches the failure modes that actually
    matter (model load mismatch, scaler-version mismatch,
    feature-pipeline drift).

The output is ``shadow_history.parquet`` with one row per
``(date, region, ticker)`` triple, including the recorded signal, the
re-computed signal, and a ``divergence_flag``.

Intended invocation: nightly via cron after US close. See
``crontab_regions.txt``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Local imports (run from execution/ on sys.path).
from signal_history import SIGNAL_HISTORY_SCHEMA, hash_features  # noqa: E402

LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SIGNAL_HISTORY = _THIS_DIR / "signal_history.parquet"
DEFAULT_SHADOW_HISTORY = _THIS_DIR / "shadow_history.parquet"


SHADOW_SCHEMA: dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "region": "string",
    "ticker": "string",
    "model_type": "string",
    "live_raw_score": "float64",
    "live_signal": "int64",
    "live_features_hash": "string",
    "shadow_raw_score": "float64",
    "shadow_signal": "int64",
    "shadow_features_hash": "string",
    "delta_raw": "float64",
    "delta_signal": "int64",
    "features_match": "bool",
    "divergence_flag": "bool",
    "divergence_reason": "string",
    "error": "string",
}


def _build_strategy_executor():
    """Build a real StrategyExecutor for the live universe.

    This wires up the same DataManager + StrategyExecutor used by main.py,
    minus the IBKR connection. Returns ``None`` if heavy deps (TF, ML
    libraries) are missing — the shadow check will log a warning and
    skip the cycle rather than crash.
    """
    try:
        from utils import setup_logger
        from data_manager import DataManager
        from strategy_executor import StrategyExecutor
    except Exception as exc:
        LOGGER.error("shadow_check could not import live modules: %s", exc)
        return None

    # utils.setup_logger(log_file, level=..., zmq_pub_socket=...) — the first
    # positional is the LOG FILE, not a logger name. Passing a name here made
    # the filename land in the `level` slot ("Unknown level: 'shadow_check.log'")
    # and crashed every run. Correct order: file first, then level.
    log = setup_logger("shadow_check.log", logging.INFO)
    dm = DataManager(log)
    # DataManager is normally fed by IBKR + parquet store. For the shadow
    # check we rely on its yfinance fallback to fetch the most recent bar,
    # which mirrors `compare_with_backtest.py`'s historical approach.
    se = StrategyExecutor(dm, log, lags=5)
    se.current_region = "SHADOW"
    # CRITICAL: the shadow check only READS/COMPARES — it must not write back
    # into signal_history.parquet. Without this, every shadow run appended a
    # synthetic "SHADOW"-region row, polluting the audited data (and the
    # attribution/hit-rate inputs) and breaking (date, region, ticker) dedup.
    se.log_signals = False
    return se


def run_shadow_for_today(
    *,
    signal_history_path: Path = DEFAULT_SIGNAL_HISTORY,
    shadow_history_path: Path = DEFAULT_SHADOW_HISTORY,
    as_of: Optional[datetime] = None,
    tickers: Optional[list[str]] = None,
    raw_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Run the shadow check, append results to shadow_history.parquet.

    Parameters
    ----------
    signal_history_path
        Where to find the LIVE recorded signals.
    shadow_history_path
        Where to write the comparison output.
    as_of
        Reference date. Defaults to today UTC.
    tickers
        Restrict to these tickers (otherwise use all in the live config).
    raw_tolerance
        Below this absolute difference in ``raw_score``, the run is
        considered concordant on raw values.

    Returns the shadow rows for inspection.
    """
    as_of_utc = as_of if (as_of and as_of.tzinfo) else datetime.now(timezone.utc)
    today = as_of_utc.date()

    if not signal_history_path.exists():
        LOGGER.warning("No signal_history.parquet at %s; nothing to shadow.",
                       signal_history_path)
        return pd.DataFrame()

    live = pd.read_parquet(signal_history_path)
    live["timestamp"] = pd.to_datetime(live["timestamp"], utc=True)
    today_live = live[live["timestamp"].dt.date == today]
    if today_live.empty:
        LOGGER.warning("signal_history.parquet has no rows for %s; "
                       "(did the live loop run today?)", today)
        return pd.DataFrame()

    if tickers:
        today_live = today_live[today_live["ticker"].isin(tickers)]

    se = _build_strategy_executor()
    if se is None:
        LOGGER.error("Could not build a StrategyExecutor; skipping shadow run.")
        return pd.DataFrame()

    rows: list[dict] = []
    for _, live_row in today_live.iterrows():
        ticker = str(live_row["ticker"])
        try:
            if hasattr(se, "data_manager") and se.data_manager is not None:
                se.data_manager.fetch_and_store_historical_data(ticker, today)
            # Re-run the same signal generation. This consults the same
            # model artifacts and feature pipeline as the live loop.
            shadow_signal_int = se.generate_signal(ticker)
            # generate_signal returns binary_signal; raw is in the log.
            # We don't currently expose raw; treat NaN as "unavailable".
            shadow_raw = float("nan")
            shadow_hash = ""  # could re-hash via dm.create_sequence_data if needed
            error = ""
        except Exception as exc:
            LOGGER.warning("Shadow failed for %s: %s", ticker, exc)
            shadow_signal_int = int(live_row["signal"])  # match → no false alarm
            shadow_raw = float("nan")
            shadow_hash = ""
            error = f"{type(exc).__name__}: {exc}"

        delta_signal = int(shadow_signal_int) - int(live_row["signal"])
        live_raw = float(live_row["raw_score"]) if pd.notna(live_row["raw_score"]) else float("nan")
        delta_raw = (shadow_raw - live_raw) if (pd.notna(shadow_raw) and pd.notna(live_raw)) else float("nan")

        features_match = (live_row["features_hash"] == shadow_hash) if shadow_hash else None
        divergence_reasons = []
        if delta_signal != 0:
            divergence_reasons.append(f"signal_diff={delta_signal:+d}")
        if pd.notna(delta_raw) and abs(delta_raw) > raw_tolerance:
            divergence_reasons.append(f"raw_diff={delta_raw:+.6f}")
        if features_match is False:
            divergence_reasons.append("features_hash_mismatch")
        if error:
            divergence_reasons.append("error")
        divergence_flag = bool(divergence_reasons)

        rows.append({
            "timestamp": pd.Timestamp(as_of_utc),
            "region": str(live_row["region"]),
            "ticker": ticker,
            "model_type": str(live_row["model_type"]),
            "live_raw_score": live_raw,
            "live_signal": int(live_row["signal"]),
            "live_features_hash": str(live_row["features_hash"]),
            "shadow_raw_score": shadow_raw,
            "shadow_signal": int(shadow_signal_int),
            "shadow_features_hash": shadow_hash,
            "delta_raw": delta_raw,
            "delta_signal": delta_signal,
            "features_match": bool(features_match) if features_match is not None else False,
            "divergence_flag": divergence_flag,
            "divergence_reason": ", ".join(divergence_reasons),
            "error": error,
        })

    if not rows:
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

    # Append idempotently on (date, region, ticker)
    if shadow_history_path.exists():
        try:
            existing = pd.read_parquet(shadow_history_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            mask = ~(
                (existing["timestamp"].dt.date == today)
                & (existing["region"].isin(new_df["region"]))
                & (existing["ticker"].isin(new_df["ticker"]))
            )
            existing = existing[mask]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:
            LOGGER.error("shadow_history.parquet corrupt: %s; rewriting.", exc)
            combined = new_df
    else:
        combined = new_df

    tmp = shadow_history_path.with_suffix(shadow_history_path.suffix + ".tmp")
    shadow_history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(tmp, index=False)
    tmp.replace(shadow_history_path)

    # Summary log
    n_total = len(new_df)
    n_divergent = int(new_df["divergence_flag"].sum())
    LOGGER.info("Shadow check: %d / %d divergent", n_divergent, n_total)
    return new_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow backtest checker (Phase 1.2)")
    parser.add_argument("--tickers", nargs="*", help="Restrict to these tickers")
    parser.add_argument("--signal-history", type=Path, default=DEFAULT_SIGNAL_HISTORY)
    parser.add_argument("--shadow-history", type=Path, default=DEFAULT_SHADOW_HISTORY)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        df = run_shadow_for_today(
            signal_history_path=args.signal_history,
            shadow_history_path=args.shadow_history,
            tickers=args.tickers,
        )
        if df.empty:
            print("No shadow rows produced (no live signals for today, or executor unavailable).")
            return 0

        n_div = int(df["divergence_flag"].sum())
        if n_div:
            print(f"DIVERGENCES: {n_div} / {len(df)} tickers")
            print(df[df["divergence_flag"]][[
                "ticker", "live_signal", "shadow_signal",
                "delta_signal", "divergence_reason"
            ]].to_string(index=False))
            return 1
        print(f"All {len(df)} signals match live recording.")
        return 0
    except Exception as exc:
        LOGGER.error("shadow_check failed: %s\n%s", exc, traceback.format_exc())
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
