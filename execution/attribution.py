"""Daily PnL attribution (Phase 1.3 of the Revision Protocol).

Decomposes a day's PnL into:
    - total_pnl              : region-window NAV delta (NOT TWS full-day P&L)
    - execution_drag         : (fill_price - decision_price) * quantity summed across fills
    - signal_contribution    : lower bound on what signals were worth
    - weighting_contribution : counterfactual: equal-weighted same signals
    - sizing_contribution    : residual after the above

Per-model 20-day rolling directional hit-rate is also computed from
``signal_history.parquet`` joined to next-day returns.

Outputs go to a small SQLite DB ``execution/attribution.db`` with two
tables: ``attribution_daily`` and ``model_hit_rate``. We intentionally
keep this SEPARATE from ``portfolio_oversight/data/oversight.db`` so the
revision-protocol observability stack can evolve independently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, date as Date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = _THIS_DIR / "attribution.db"
DEFAULT_EQUITY_HISTORY = _THIS_DIR / "equity_history.parquet"
DEFAULT_SIGNAL_HISTORY = _THIS_DIR / "signal_history.parquet"
DEFAULT_EXECUTION_JOURNALS = _THIS_DIR / "execution_journals"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attribution_daily (
    date TEXT NOT NULL,
    region TEXT NOT NULL,
    total_pnl REAL,
    execution_drag REAL,
    signal_contribution REAL,
    weighting_contribution REAL,
    sizing_contribution REAL,
    nav_open REAL,
    nav_close REAL,
    notes TEXT,
    PRIMARY KEY (date, region)
);

CREATE TABLE IF NOT EXISTS model_hit_rate (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    model_type TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    hits INTEGER,
    misses INTEGER,
    total INTEGER,
    unresolved INTEGER,
    hit_rate REAL,
    PRIMARY KEY (date, ticker, model_type, window_days)
);
"""

# Columns added after the table first shipped; _open_db backfills them on
# existing databases via ALTER TABLE (SQLite has no IF NOT EXISTS for columns).
_MIGRATIONS = [
    ("model_hit_rate", "total", "ALTER TABLE model_hit_rate ADD COLUMN total INTEGER"),
    ("model_hit_rate", "unresolved", "ALTER TABLE model_hit_rate ADD COLUMN unresolved INTEGER"),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a table first shipped (idempotent)."""
    for table, column, ddl in _MIGRATIONS:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(ddl)


@contextmanager
def _open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA_SQL)
        _apply_migrations(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class DailyAttribution:
    date: Date
    region: str
    total_pnl: float
    execution_drag: float
    signal_contribution: float
    weighting_contribution: float
    sizing_contribution: float
    nav_open: float
    nav_close: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_equity_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date"] = df["timestamp"].dt.date
    return df


def _load_signal_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["date"] = df["timestamp"].dt.date
    return df


def _load_execution_journals(dirpath: Path, as_of_date: Date) -> list[dict]:
    """Read the day's execution journal entries."""
    fp = dirpath / f"{as_of_date.isoformat()}_journal.json"
    if not fp.exists():
        return []
    try:
        return json.loads(fp.read_text())
    except Exception as exc:
        LOGGER.warning("Could not parse execution journal %s: %s", fp, exc)
        return []


# ---------------------------------------------------------------------------
# Attribution maths
# ---------------------------------------------------------------------------


def compute_daily_attribution(
    *,
    region: str,
    as_of_date: Date,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY,
    signal_history_path: Path = DEFAULT_SIGNAL_HISTORY,
    execution_journals_dir: Path = DEFAULT_EXECUTION_JOURNALS,
) -> Optional[DailyAttribution]:
    """Compute attribution numbers for one (region, date).

    Returns None if equity history insufficient.
    """
    eq = _load_equity_history(equity_history_path)
    if eq.empty:
        return None

    eq_today = eq[(eq["region"] == region) & (eq["date"] == as_of_date)].copy()
    if eq_today.empty:
        return None

    nav_open = float(eq_today["nav_usd"].iloc[0])
    nav_close = float(eq_today["nav_usd"].iloc[-1])
    total_pnl = nav_close - nav_open

    # Execution drag: extracted from execution journal (rough proxy — we
    # treat avg_price - intended price as drag; intended price absent
    # in journal today, so we use 0 as a placeholder. This is improved in
    # Phase 1.4 once we hook decision-price logging.)
    journal = _load_execution_journals(execution_journals_dir, as_of_date)
    execution_drag = 0.0
    for order in journal:
        events = order.get("events", []) if isinstance(order, dict) else []
        if not events:
            continue
        last = events[-1]
        if last.get("status") == "Filled" and "avg_price" in last:
            # No "intended price" in journal — leave drag at 0 for the
            # placeholder. Phase 1.4 will hook in decision_price.
            pass
    # TODO(Phase 1.4): wire actual execution drag once decision_price is logged.

    # Counterfactual: equal-weighted across active ml_signal tickers.
    sig = _load_signal_history(signal_history_path)
    weighting_contribution = float("nan")
    signal_contribution = float("nan")
    if not sig.empty:
        sig_today = sig[(sig["region"] == region) & (sig["date"] == as_of_date)]
        if not sig_today.empty:
            # Without per-ticker daily returns we cannot complete the
            # counterfactual today; we record placeholders. Phase 1.4
            # extends data_manager to feed per-ticker pct returns.
            signal_contribution = float("nan")
            weighting_contribution = float("nan")

    # Sizing contribution defined as residual.
    if all(np.isnan(x) for x in (signal_contribution, weighting_contribution)):
        sizing_contribution = float("nan")
    else:
        sizing_contribution = total_pnl - (
            (signal_contribution if not np.isnan(signal_contribution) else 0.0)
            + (weighting_contribution if not np.isnan(weighting_contribution) else 0.0)
            + execution_drag
        )

    return DailyAttribution(
        date=as_of_date,
        region=region,
        total_pnl=total_pnl,
        execution_drag=execution_drag,
        signal_contribution=signal_contribution,
        weighting_contribution=weighting_contribution,
        sizing_contribution=sizing_contribution,
        nav_open=nav_open,
        nav_close=nav_close,
        notes=(
            "phase1_placeholder: total_pnl is region-window NAV delta, not "
            "TWS full-day P&L; signal/weighting counterfactuals need "
            "per-ticker returns"
        ),
    )


# ---------------------------------------------------------------------------
# Per-model rolling hit-rate
# ---------------------------------------------------------------------------


# Below this many resolved samples a hit-rate is statistically meaningless.
# We still record hits/misses but emit hit_rate=NaN so the dashboard shows
# "n too small" instead of a misleading 1.000 / 0.000.
MIN_HITRATE_SAMPLES = 5


def compute_model_hit_rates(
    *,
    as_of_date: Date,
    window_days: int = 20,
    signal_history_path: Path = DEFAULT_SIGNAL_HISTORY,
    price_returns: Optional[pd.DataFrame] = None,
    min_samples: int = MIN_HITRATE_SAMPLES,
) -> pd.DataFrame:
    """Compute rolling directional hit-rate per (ticker, model_type).

    A "hit" is signal=+1 followed by next-day return > 0, OR
    signal=-1 followed by next-day return < 0.

    ``window_days`` is honoured as a count of the most recent *signal-emitting
    trading days that have a resolved next-day return*, per (ticker,
    model_type). Signals whose next-day return is not yet available (the last
    day or two, plus the first day before the price series starts) are NOT
    counted — they are tracked separately and never silently inflate/deflate
    the rate.

    ``price_returns`` must be a DataFrame with columns (date, ticker,
    next_day_return). If None, we attempt to derive it from yfinance; if that
    fails, returns an empty frame (caller logs a warning).

    Returns columns:
        date, ticker, model_type, window_days, hits, misses, total,
        unresolved, hit_rate
    where ``hit_rate`` is NaN when ``total < min_samples`` (insufficient data).
    """
    sig = _load_signal_history(signal_history_path)
    if sig.empty:
        return pd.DataFrame()

    # Pull a generous calendar span to source candidate signals; the actual
    # window is enforced later as a count of resolved trading days, not a
    # calendar slice (calendar slicing conflated holidays/weekends with the
    # intended trading-day window and produced "20-day" rows spanning ~40
    # calendar days).
    span_start = as_of_date - pd.Timedelta(days=window_days * 3)
    sig_span = sig[(sig["date"] >= span_start) & (sig["date"] <= as_of_date)].copy()
    if sig_span.empty:
        return pd.DataFrame()

    if price_returns is None:
        price_returns = _fetch_next_day_returns(
            tickers=sorted(sig_span["ticker"].unique().tolist()),
            start=span_start, end=as_of_date,
        )
    if price_returns.empty:
        LOGGER.warning("No next-day returns available; hit rate cannot be computed.")
        return pd.DataFrame()

    merged = sig_span.merge(price_returns, on=["date", "ticker"], how="left")

    # Track how many recent signals could not be scored (no next-day return
    # yet) per group, so the absence of the freshest days is visible.
    unresolved = (
        merged[merged["next_day_return"].isna()]
        .groupby(["ticker", "model_type"]).size()
        .rename("unresolved")
    )

    resolved = merged.dropna(subset=["next_day_return"]).copy()
    if resolved.empty:
        return pd.DataFrame()

    # Enforce the trading-day window: keep only the most recent `window_days`
    # resolved signal dates within each (ticker, model_type) group.
    resolved = resolved.sort_values("date")
    resolved = (
        resolved.groupby(["ticker", "model_type"], group_keys=False)
        .apply(lambda g: g.tail(window_days))
    )

    resolved["hit"] = (
        ((resolved["signal"] > 0) & (resolved["next_day_return"] > 0))
        | ((resolved["signal"] < 0) & (resolved["next_day_return"] < 0))
    ).astype(int)

    grouped = resolved.groupby(["ticker", "model_type"])["hit"].agg(["sum", "count"]).reset_index()
    grouped.columns = ["ticker", "model_type", "hits", "total"]
    grouped["misses"] = grouped["total"] - grouped["hits"]
    grouped = grouped.merge(unresolved, on=["ticker", "model_type"], how="left")
    grouped["unresolved"] = grouped["unresolved"].fillna(0).astype(int)

    # Suppress statistically meaningless rates (n too small) to NaN.
    raw_rate = grouped["hits"] / grouped["total"].replace(0, np.nan)
    grouped["hit_rate"] = raw_rate.where(grouped["total"] >= min_samples, other=np.nan)

    grouped["date"] = as_of_date
    grouped["window_days"] = window_days
    return grouped[[
        "date", "ticker", "model_type", "window_days",
        "hits", "misses", "total", "unresolved", "hit_rate",
    ]]


def _fetch_next_day_returns(
    *, tickers: list[str], start: Date, end: Date
) -> pd.DataFrame:
    """Best-effort fetch of next-day returns via yfinance.

    Returns columns (date, ticker, next_day_return). Empty on failure.
    """
    try:
        import yfinance as yf
    except Exception:
        LOGGER.warning("yfinance not available; cannot compute next-day returns.")
        return pd.DataFrame()

    out: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=str(start), end=str(end + pd.Timedelta(days=2)),
                             interval="1d", auto_adjust=False)
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["Date"]).dt.date
            hist["next_day_return"] = hist["Close"].pct_change().shift(-1)
            hist["ticker"] = ticker
            out.append(hist[["date", "ticker", "next_day_return"]])
        except Exception as exc:
            LOGGER.debug("yfinance fetch failed for %s: %s", ticker, exc)
            continue

    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    return df.dropna(subset=["next_day_return"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_daily_attribution(
    attribution: DailyAttribution, *, db_path: Path = DEFAULT_DB_PATH,
) -> None:
    with _open_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO attribution_daily
            (date, region, total_pnl, execution_drag, signal_contribution,
             weighting_contribution, sizing_contribution, nav_open, nav_close, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attribution.date.isoformat(),
                attribution.region,
                attribution.total_pnl,
                attribution.execution_drag,
                attribution.signal_contribution,
                attribution.weighting_contribution,
                attribution.sizing_contribution,
                attribution.nav_open,
                attribution.nav_close,
                attribution.notes,
            ),
        )


def persist_hit_rates(df: pd.DataFrame, *, db_path: Path = DEFAULT_DB_PATH) -> None:
    if df.empty:
        return
    rows = df.to_dict("records")
    with _open_db(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO model_hit_rate
            (date, ticker, model_type, window_days, hits, misses, total,
             unresolved, hit_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(r["date"].isoformat() if hasattr(r["date"], "isoformat") else r["date"],
              r["ticker"], r["model_type"], int(r["window_days"]),
              int(r["hits"]), int(r["misses"]),
              int(r["total"]) if pd.notna(r.get("total")) else int(r["hits"]) + int(r["misses"]),
              int(r["unresolved"]) if pd.notna(r.get("unresolved")) else 0,
              float(r["hit_rate"]) if pd.notna(r["hit_rate"]) else None)
             for r in rows],
        )


def get_attribution_history(
    region: Optional[str] = None,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 60,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = "SELECT * FROM attribution_daily"
    args: tuple = ()
    if region:
        sql += " WHERE region = ?"
        args = (region,)
    sql += " ORDER BY date DESC LIMIT ?"
    args = args + (limit,)
    with _open_db(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=args)


def get_hit_rate_history(
    ticker: Optional[str] = None,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 60,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    sql = "SELECT * FROM model_hit_rate"
    args: tuple = ()
    if ticker:
        sql += " WHERE ticker = ?"
        args = (ticker,)
    sql += " ORDER BY date DESC LIMIT ?"
    args = args + (limit,)
    with _open_db(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Exit codes (consumed by run_region.sh / cron so failures are LOUD):
#   0  success — attribution row persisted (hit-rates persisted unless --no-hit-rates)
#   1  hard failure — could not compute/persist attribution for the region/date
#   2  no equity history for region/date AND --allow-missing was given (soft skip)
#   3  attribution persisted, but hit-rates were requested and came back EMPTY
#      (e.g. yfinance unavailable / returned nothing) — partial, needs eyes
EXIT_OK = 0
EXIT_HARD_FAIL = 1
EXIT_SOFT_SKIP = 2
EXIT_HITRATES_EMPTY = 3


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daily PnL attribution (Phase 1.3)")
    parser.add_argument("--region", default="US")
    parser.add_argument("--date", default=None, help="ISO date; defaults to today UTC")
    parser.add_argument("--no-hit-rates", action="store_true")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Treat 'no equity history for this region/date' as a soft skip "
             "(exit 2) rather than a hard failure (exit 1). Used by the "
             "multi-region cron loop where not every region trades every day.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    as_of = (pd.to_datetime(args.date).date() if args.date
             else datetime.now(timezone.utc).date())

    attr = compute_daily_attribution(region=args.region, as_of_date=as_of)
    if attr is None:
        msg = f"No equity history for region={args.region} date={as_of}."
        if args.allow_missing:
            LOGGER.warning("%s Soft-skipping (--allow-missing).", msg)
            print(msg + " (soft skip)")
            return EXIT_SOFT_SKIP
        LOGGER.error("%s", msg)
        print(msg)
        return EXIT_HARD_FAIL
    persist_daily_attribution(attr)
    print(f"Attribution (region={args.region}, date={as_of}):")
    print(f"  region_nav_delta       = ${attr.total_pnl:+.2f}")
    print(f"  execution_drag         = ${attr.execution_drag:+.2f}")
    print(f"  signal_contribution    = {attr.signal_contribution}")
    print(f"  weighting_contribution = {attr.weighting_contribution}")
    print(f"  sizing_contribution    = {attr.sizing_contribution}")

    rc = EXIT_OK
    if not args.no_hit_rates:
        hr = compute_model_hit_rates(as_of_date=as_of)
        persist_hit_rates(hr)
        if hr.empty:
            # FAIL LOUDLY: attribution landed but hit-rates silently produced
            # nothing. Most common cause is yfinance unavailable/empty in the
            # cron environment. Surface it instead of a silent no-op.
            LOGGER.warning(
                "Hit-rate computation returned NO rows for date=%s "
                "(yfinance unavailable/empty, or no signals in window). "
                "model_hit_rate was NOT advanced.", as_of,
            )
            print("WARNING: per-model hit rates EMPTY — model_hit_rate not advanced.")
            rc = EXIT_HITRATES_EMPTY
        else:
            print("\nPer-model hit rates (20d):")
            for _, r in hr.iterrows():
                print(f"  {r['ticker']:>15} [{r['model_type']:>10}]: "
                      f"{r['hits']}/{r['hits']+r['misses']} = {r['hit_rate']:.3f}")
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
