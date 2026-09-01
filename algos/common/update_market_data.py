"""
Market data store updater. Populates and incrementally updates the
parquet-based market data store.

Usage:
    # Initial population (one-time, takes a while for many tickers)
    python -m algos.common.update_market_data --init --lookback-days 1825

    # Daily incremental update (fast, only fetches missing days)
    python -m algos.common.update_market_data

    # Update specific tickers
    python -m algos.common.update_market_data --tickers SPY NVDA 8058.T EURUSD=X

    # Weekly full-refresh last 5 days (catch split/dividend corrections)
    python -m algos.common.update_market_data --full-refresh 5

    # Seed from existing CSVs (migrate without re-downloading)
    python -m algos.common.update_market_data --seed-from-csv

    # Export portfolio CSV (replaces yfinance_downloader_v5.py CSV output)
    python -m algos.common.update_market_data --export-csv output.csv --start 2021-01-01 --end 2026-02-25

    # Force yfinance only (skip IBKR even if available)
    python -m algos.common.update_market_data --source yfinance

    # Fetch FRED macro data
    python -m algos.common.update_market_data --tickers "FRED:FEDFUNDS" "FRED:IRLTLT01JPM156N"

    # Status check
    python -m algos.common.update_market_data --status
"""

import argparse
from collections import Counter
import json
import logging
import os
import random
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Project root setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algos.common.market_data_store import MarketDataStore, StoreWriteTemporaryError
from algos.common.yf_downloader import resilient_download_single

logger = logging.getLogger(__name__)

_TRANSIENT_ERROR_PATTERNS = re.compile(
    r"timeout|timed out|connection|ssl|tls|temporary failure|"
    r"unable to open database|database is locked|curl|dns|nonetype.*subscriptable",
    re.IGNORECASE,
)

# Try to import IBKR downloader
try:
    from algos.common.ibkr_downloader import (
        IBKRDataDownloader,
        is_gateway_available,
        PacingViolation,
    )

    _HAS_IBKR = True
except ImportError:
    _HAS_IBKR = False
    logger.debug("ibkr_downloader not available; IBKR source disabled")

# Try to import FRED downloader
try:
    from algos.common.fred_downloader import (
        is_fred_ticker,
        parse_fred_ticker,
        download_fred_series,
        is_fred_available,
    )

    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False
    logger.debug("fred_downloader not available; FRED source disabled")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

UPDATER_CONFIG = {
    "max_retries": 50,  # Per ticker, before giving up
    "retry_wait_min": 60.0,  # Seconds to wait on all-sources-failed
    "retry_wait_max": 120.0,  # Upper bound for retry wait (with jitter)
    "max_workers": 2,  # Threads for batch updates
    "delay_between_tickers": 0.5,  # Seconds between sequential downloads
    "default_lookback_days": 1825,  # 5 years
}

LOCKFILE_PATH = Path(_PROJECT_ROOT) / "data" / "market_data" / ".updater.lock"

# ===========================================================================
# PID LOCK
# ===========================================================================


class PIDLock:
    """Simple PID-file lock to prevent concurrent updater instances."""

    def __init__(self, path: Path):
        self.path = path

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns False if another instance is running."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                old_pid = int(self.path.read_text().strip())
                # Check if old process is still running
                os.kill(old_pid, 0)
                logger.error(
                    f"Another updater instance is running (PID {old_pid}). "
                    f"Delete {self.path} if this is stale."
                )
                return False
            except (ProcessLookupError, ValueError):
                # Process is gone, stale lockfile
                logger.info(
                    f"Removing stale lockfile (PID was {self.path.read_text().strip()})"
                )
                self.path.unlink()
            except PermissionError:
                logger.error(f"Cannot check lockfile PID. Delete {self.path} manually.")
                return False

        self.path.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        """Release the lock."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


# ===========================================================================
# TICKER LIST EXTRACTION
# ===========================================================================


def load_ticker_universe() -> dict:
    """
    Load the ticker universe from the JSON config file.

    The ticker universe is stored in data/market_data/ticker_universe.json.
    This file is the single source of truth for what tickers the market data
    store tracks. It maps yfinance ticker -> output column name.

    Falls back to regex-parsing yfinance_downloader_v5.py if the JSON doesn't exist.

    Returns:
        Dict mapping yfinance ticker -> output column name.
    """
    json_path = _PROJECT_ROOT / "data" / "market_data" / "ticker_universe.json"

    # Primary: read from JSON config
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            tickers = data.get("tickers", {})
            if tickers:
                logger.info(f"Loaded {len(tickers)} tickers from {json_path.name}")
                return tickers
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {json_path}: {e}")

    # Fallback: parse yfinance_downloader_v5.py
    logger.info(
        "ticker_universe.json not found, falling back to parsing yfinance_downloader_v5.py"
    )
    return _load_ticker_universe()


def _load_ticker_universe() -> dict:
    """
    Fallback: extract tickers by regex-parsing yfinance_downloader_v5.py.

    Finds the FIRST current_tickers_map definition (the active one)
    and parses all uncommented entries.

    Returns:
        Dict mapping yfinance ticker -> output column name.
    """
    downloader_path = (
        _PROJECT_ROOT / "algos" / "backtest_code" / "yfinance_downloader_v5.py"
    )
    if not downloader_path.exists():
        logger.error(f"Cannot find {downloader_path}")
        return {}

    try:
        content = downloader_path.read_text()
        # Find ALL current_tickers_map blocks and use the LAST one
        # (the file may have multiple definitions; the last assignment wins)
        matches = list(
            re.finditer(
                r"current_tickers_map\s*=\s*\{(.*?)\n    \}",
                content,
                re.DOTALL,
            )
        )
        if not matches:
            logger.error(
                "Could not find current_tickers_map in yfinance_downloader_v5.py"
            )
            return {}

        block = matches[-1].group(1)
        # Parse active (non-commented) entries
        tickers = {}
        pattern = re.compile(r"""['"]([^'"]+)['"]\s*:\s*['"]([^'"]+)['"]""")
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            entry_match = pattern.search(line)
            if entry_match:
                yf_ticker = entry_match.group(1)
                col_name = entry_match.group(2).strip()
                tickers[yf_ticker] = col_name

        logger.info(f"Extracted {len(tickers)} tickers from yfinance_downloader_v5.py")
        return tickers
    except Exception as e:
        logger.error(f"Error parsing yfinance_downloader_v5.py: {e}")
        return {}


def _expand_ticker_args(raw_tickers: list) -> list:
    """Expand comma/space-separated ticker CLI args into a clean list."""
    expanded = []
    for item in raw_tickers:
        if item is None:
            continue
        text = str(item).strip().strip('"').strip("'")
        if not text:
            continue
        parts = re.split(r"[\s,;]+", text)
        for part in parts:
            token = part.strip().strip('"').strip("'")
            if token:
                expanded.append(token)

    deduped = []
    seen = set()
    for ticker in expanded:
        normalized = ticker.upper()
        if normalized not in seen:
            deduped.append(ticker)
            seen.add(normalized)
    return deduped


# ===========================================================================
# CSV SEEDER
# ===========================================================================


def seed_from_csv(store: MarketDataStore, csv_dir: str = None) -> int:
    """
    Seed the parquet store from existing CSV files in the data/ directory.

    Reads multi-ticker CSVs (financial_data_combined_prices_*.csv) and
    converts each ticker column to a per-ticker parquet file.

    Args:
        store: MarketDataStore instance
        csv_dir: Directory containing CSVs. Defaults to data/

    Returns:
        Number of tickers seeded.
    """
    csv_dir = Path(csv_dir) if csv_dir else _PROJECT_ROOT / "data"
    csv_files = sorted(csv_dir.glob("financial_data_combined_prices_*.csv"))

    if not csv_files:
        logger.warning(
            f"No financial_data_combined_prices_*.csv files found in {csv_dir}"
        )
        return 0

    # Use the most recent CSV (latest end date)
    latest_csv = csv_files[-1]
    logger.info(f"Seeding from {latest_csv.name} ...")

    try:
        df = pd.read_csv(latest_csv, index_col="Date", parse_dates=True)
    except Exception as e:
        logger.error(f"Could not read {latest_csv}: {e}")
        return 0

    seeded = 0
    tickers_map = load_ticker_universe()
    # Build reverse map: column_name -> yfinance_ticker
    reverse_map = {v: k for k, v in tickers_map.items()} if tickers_map else {}

    for col in df.columns:
        if col in ("Date", "date"):
            continue

        price_series = df[col].dropna()
        if price_series.empty:
            continue

        # Create a minimal OHLCV DataFrame (we only have price data from the CSV)
        ticker_df = pd.DataFrame(index=price_series.index)
        ticker_df["open"] = price_series.values
        ticker_df["high"] = price_series.values
        ticker_df["low"] = price_series.values
        ticker_df["close"] = price_series.values
        ticker_df["volume"] = 0.0
        ticker_df["adj_close"] = price_series.values
        ticker_df["source"] = "csv_seed"
        ticker_df.index.name = "date"

        # Determine the ticker name for the parquet file
        # Use the yfinance ticker if available, otherwise use the column name
        yf_ticker = reverse_map.get(col, col)
        normalized = store.normalize_ticker(yf_ticker)

        store.write_ticker(normalized, ticker_df, source="csv_seed")
        seeded += 1
        if seeded % 50 == 0:
            logger.info(f"  Seeded {seeded} tickers...")

    logger.info(f"Seeded {seeded} tickers from {latest_csv.name}")
    return seeded


# ===========================================================================
# MARKET DATA UPDATER
# ===========================================================================


class MarketDataUpdater:
    """
    Updates the parquet-based market data store with fresh data.

    Phase 1: yfinance only.
    Phase 2: IBKR primary, yfinance fallback.

    The updater never quits on transient failures. It retries with exponential
    backoff up to max_retries, then logs the failure and moves on. A ticker
    is never silently skipped -- every failure is logged.
    """

    def __init__(
        self,
        store: MarketDataStore,
        ibkr_downloader=None,
        max_retries: int = None,
        max_workers: int = None,
        source: str = "auto",
        skip_fresh: bool = False,
    ):
        self.store = store
        self.ibkr = ibkr_downloader
        self.max_retries = max_retries or UPDATER_CONFIG["max_retries"]
        self.max_workers = max_workers or UPDATER_CONFIG["max_workers"]
        self.source = source  # "auto", "yfinance", "ibkr"
        self._skip_fresh = skip_fresh
        self._stats = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "quarantined": 0,
        }
        self._stats_lock = threading.Lock()
        self._failure_signatures = Counter()
        self._failure_lock = threading.Lock()
        self._yf_attempts = 0
        self._yf_failures = 0
        self._yf_mode_lock = threading.Lock()
        self._yf_concurrency_guard = threading.Semaphore(2)
        self._yf_serial_lock = threading.Lock()
        self._write_retry_wait = (2.0, 5.0)

    def _record_failure_signature(self, source: str, error: Exception) -> None:
        """Record grouped failure signatures for run-end diagnostics."""
        message = str(error).strip().lower() if error is not None else "unknown"
        if not message:
            message = "unknown"
        message = " ".join(message.split())
        message = message[:160]
        signature = f"{source}:{message}"
        with self._failure_lock:
            self._failure_signatures[signature] += 1

    def _record_yf_outcome(self, success: bool) -> None:
        """Track yfinance health to switch into degraded serial mode."""
        with self._yf_mode_lock:
            self._yf_attempts += 1
            if not success:
                self._yf_failures += 1

    def _in_yf_degraded_mode(self) -> bool:
        """True when yfinance failure rate indicates throttling is needed."""
        with self._yf_mode_lock:
            if self._yf_attempts < 20:
                return False
            return (self._yf_failures / self._yf_attempts) >= 0.35

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        """Classify likely transient/network/cache failures for retry behavior."""
        return bool(_TRANSIENT_ERROR_PATTERNS.search(str(error)))

    @staticmethod
    def _extract_latest_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract a 1D adjusted-close/close series from a downloaded frame."""
        if df is None or df.empty:
            return None

        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = work.columns.get_level_values(0)
        if work.columns.duplicated().any():
            work = work.loc[:, ~work.columns.duplicated(keep="last")]

        for col in ["Adj Close", "Close", "adj_close", "close"]:
            if col in work.columns:
                series = work[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, -1]
                if isinstance(series, pd.Series):
                    return series.dropna()
        return None

    def _has_scale_mismatch(
        self, ticker: str, fetched_df: pd.DataFrame, fetch_start: str, end: str
    ) -> bool:
        """Check for severe price-scale mismatch between store and fresh fetch.

        This catches legacy corrupted/scaled ticker histories by comparing
        overlapping adjusted-close series medians.
        """
        existing_df = self.store.get_ohlcv_raw(ticker, start=fetch_start, end=end)
        if existing_df is None or existing_df.empty:
            return False

        fetched_prices = self._extract_latest_price_series(fetched_df)
        if fetched_prices is None or fetched_prices.empty:
            return False

        existing_col = "adj_close" if "adj_close" in existing_df.columns else "close"
        if existing_col not in existing_df.columns:
            return False

        existing_prices = existing_df[existing_col].dropna()
        if existing_prices.empty:
            return False

        common_index = existing_prices.index.intersection(fetched_prices.index)
        if len(common_index) < 5:
            return False

        old_med = float(existing_prices.loc[common_index].median())
        new_med = float(fetched_prices.loc[common_index].median())
        if old_med <= 0.0 or new_med <= 0.0:
            return False

        ratio = old_med / new_med
        return ratio < 0.5 or ratio > 2.0

    def update_ticker(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        full_refresh_days: int = 0,
    ) -> bool:
        """
        Update a single ticker in the parquet store.

        1. Check parquet for latest date
        2. If up-to-date, skip
        3. If full_refresh_days > 0, re-download last N days (overwrites)
        4. Otherwise, fetch only missing date range (append)
        5. Try data sources with failover
        6. On failure, retry with backoff up to max_retries

        Args:
            ticker: Ticker symbol (yfinance format)
            start: Start date override (YYYY-MM-DD). Used for initial population.
            end: End date override (YYYY-MM-DD). Defaults to today.
            full_refresh_days: If > 0, re-download last N trading days.

        Returns:
            True if updated (or already up-to-date), False if failed after all retries.
        """
        normalized = self.store.normalize_ticker(ticker)
        today = datetime.now().strftime("%Y-%m-%d")
        end = end or today
        end_ts = pd.Timestamp(end)

        # --skip-fresh: skip tickers that already have recent IBKR-sourced data.
        # This enables resuming after a Gateway crash without re-downloading
        # the ~868 tickers that were already successfully fetched.
        if self._skip_fresh and self.store.has_ticker(normalized):
            try:
                ticker_path = self.store._ticker_path(normalized)
                raw_df = pd.read_parquet(ticker_path)
                if "source" in raw_df.columns and len(raw_df) > 0:
                    last_source = raw_df["source"].iloc[-1]
                    if last_source == "ibkr":
                        last_date = raw_df.index[-1]
                        days_old = (pd.Timestamp.now() - pd.Timestamp(last_date)).days
                        if days_old <= 7:
                            logger.debug(
                                "[%s] Fresh IBKR data (%dd old) — skipping (--skip-fresh)",
                                ticker,
                                days_old,
                            )
                            with self._stats_lock:
                                self._stats["skipped"] += 1
                            return True
            except Exception:
                pass  # Can't determine freshness — proceed with download

        # Pre-flight: check if IBKR contract map marks this ticker as unfetchable.
        # This prevents ANY retry attempts for tickers that reqContractDetails
        # already confirmed don't exist on IBKR (indices, ETPs, mutual funds, etc.).
        if self.source == "ibkr" and self.ibkr is not None:
            contract_map = self.ibkr._load_contract_map()
            if ticker in contract_map and contract_map[ticker] is None:
                logger.debug(
                    "[%s] Unfetchable via IBKR (null in contract map) — skipping",
                    normalized,
                )
                with self._stats_lock:
                    self._stats["skipped"] += 1
                return True  # Not an error — permanently unfetchable

        if self.source == "ibkr" and self.ibkr is None:
            logger.error(
                f"[{normalized}] IBKR-only mode requested but IBKR connection is unavailable"
            )
            with self._stats_lock:
                self._stats["failed"] += 1
            return False

        # Determine what date range we need
        if full_refresh_days > 0 and self.store.has_ticker(ticker):
            # Re-download last N calendar days for correction coverage
            refresh_start = (end_ts - timedelta(days=full_refresh_days * 2)).strftime(
                "%Y-%m-%d"
            )
            fetch_start = refresh_start
            logger.info(
                f"[{normalized}] Full refresh: re-downloading from {fetch_start}"
            )
        elif self.store.has_ticker(ticker):
            # If --init was specified with a start date and source is ibkr,
            # check if existing data is from yfinance. If so, do a full
            # re-download from start instead of incremental. This replaces
            # stale yfinance data with authoritative IBKR data.
            _force_full = False
            if start and self.source == "ibkr":
                try:
                    ticker_path = self.store._ticker_path(normalized)
                    raw_df = pd.read_parquet(ticker_path)
                    if "source" in raw_df.columns and len(raw_df) > 0:
                        last_source = raw_df["source"].iloc[-1]
                        if last_source != "ibkr":
                            # Source is yfinance — replace with IBKR
                            fetch_start = start
                            _force_full = True
                            logger.info(
                                f"[{normalized}] Re-downloading from {fetch_start} "
                                f"(replacing {last_source} data with IBKR)"
                            )
                        else:
                            # Source is IBKR but check if data is too short.
                            # If --init requested 1825 days (~1200 bars) but
                            # we only have 250 bars, re-download the full range.
                            start_ts = pd.Timestamp(start)
                            first_date = raw_df.index[0]
                            if first_date > start_ts + timedelta(days=30):
                                fetch_start = start
                                _force_full = True
                                logger.info(
                                    f"[{normalized}] IBKR data too short "
                                    f"(starts {first_date.date()}, want {start_ts.date()}). "
                                    f"Re-downloading from {fetch_start}"
                                )
                except Exception:
                    pass

            if not _force_full:
                # Incremental: fetch from last date + 1 day
                date_range = self.store.get_date_range(ticker)
                if date_range and date_range[1]:
                    last_date = pd.Timestamp(date_range[1])
                    days_behind = (end_ts - last_date).days
                    if days_behind <= 1:
                        with self._stats_lock:
                            self._stats["skipped"] += 1
                            skipped_count = self._stats["skipped"]
                        if skipped_count == 1:
                            logger.info(
                                f"[{normalized}] Already up-to-date (last: {date_range[1]}). "
                                f"Skipping tickers with data through {date_range[1]} or later..."
                            )
                        elif skipped_count % 200 == 0:
                            logger.info(
                                f"  ...{skipped_count} tickers already up-to-date, skipping"
                            )
                        else:
                            logger.debug(
                                f"[{normalized}] Already up-to-date (last: {date_range[1]})"
                            )
                        return True
                    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(
                        f"[{normalized}] Incremental: fetching from {fetch_start} ({days_behind} days behind)"
                    )
                else:
                    fetch_start = start or (
                        pd.Timestamp(end)
                        - timedelta(days=UPDATER_CONFIG["default_lookback_days"])
                    ).strftime("%Y-%m-%d")
        else:
            # New ticker -- full download
            fetch_start = start or (
                pd.Timestamp(end)
                - timedelta(days=UPDATER_CONFIG["default_lookback_days"])
            ).strftime("%Y-%m-%d")
            logger.info(f"[{normalized}] New ticker: downloading from {fetch_start}")

        # Nothing to fetch if range is inverted/empty after incremental math.
        if pd.Timestamp(fetch_start) >= end_ts:
            with self._stats_lock:
                self._stats["skipped"] += 1
            logger.debug(
                f"[{normalized}] No fetch needed (fetch_start={fetch_start}, end={end})"
            )
            return True

        # Fetch with retry (cap retries when no fallback source is available)
        effective_max_retries = self.max_retries
        retry_wait_min = UPDATER_CONFIG["retry_wait_min"]
        retry_wait_max = UPDATER_CONFIG["retry_wait_max"]
        if self.ibkr is None and self.source in ("auto", "yfinance"):
            # yfinance-only: limited retries with longer waits
            effective_max_retries = min(self.max_retries, 6)
            retry_wait_min = 8.0
            retry_wait_max = 20.0
        elif self.source == "ibkr":
            # IBKR-only: cap retries — if IBKR says "no security definition",
            # retrying 50 times with 60-120s backoff wastes hours.
            effective_max_retries = min(self.max_retries, 3)
            retry_wait_min = 2.0
            retry_wait_max = 5.0

        for attempt in range(effective_max_retries):
            df, source = self._fetch_with_failover(ticker, fetch_start, end)
            if df is not None and not df.empty:
                force_overwrite = False
                is_forex = False
                if _HAS_IBKR:
                    from algos.common.ibkr_downloader import is_forex_ticker

                    is_forex = is_forex_ticker(ticker)

                if (
                    source == "yfinance"
                    and not is_forex
                    and self._has_scale_mismatch(ticker, df, fetch_start, end)
                ):
                    logger.warning(
                        f"[{normalized}] Detected historical scale mismatch; performing full yfinance rebuild"
                    )
                    deep_start = (
                        end_ts - timedelta(days=UPDATER_CONFIG["default_lookback_days"])
                    ).strftime("%Y-%m-%d")
                    if deep_start != fetch_start:
                        refreshed_df, refreshed_source = self._try_yfinance(
                            ticker, deep_start, end
                        )
                        if refreshed_df is not None and not refreshed_df.empty:
                            df = refreshed_df
                            source = refreshed_source
                            force_overwrite = True

                try:
                    write_ok = self.store.write_ticker(
                        ticker,
                        df,
                        source=source,
                        overwrite=force_overwrite,
                    )
                except StoreWriteTemporaryError as e:
                    self._record_failure_signature("store", e)
                    if attempt < effective_max_retries - 1:
                        wait = random.uniform(*self._write_retry_wait)
                        logger.warning(
                            f"[{normalized}] Temporary write failure ({e}). Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                        continue
                    logger.error(
                        f"[{normalized}] FAILED due to repeated temporary store write failures"
                    )
                    with self._stats_lock:
                        self._stats["failed"] += 1
                    return False
                if not write_ok:
                    with self._stats_lock:
                        self._stats["quarantined"] += 1
                        self._stats["failed"] += 1
                    logger.error(
                        f"[{normalized}] Data rejected by validator; quarantined and skipped write"
                    )
                    self._record_failure_signature(
                        source or "unknown",
                        ValueError("malformed dataframe during write"),
                    )
                    return False
                with self._stats_lock:
                    self._stats["updated"] += 1
                logger.info(f"[{normalized}] Updated: {len(df)} bars from {source}")
                return True

            if attempt < effective_max_retries - 1:
                wait = random.uniform(retry_wait_min, retry_wait_max)
                logger.warning(
                    f"[{normalized}] All sources failed (attempt {attempt + 1}/{effective_max_retries}). "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)

        logger.error(f"[{normalized}] FAILED after {effective_max_retries} attempts")
        with self._stats_lock:
            self._stats["failed"] += 1
        return False

    def update_all(
        self,
        tickers_map: dict = None,
        start: str = None,
        end: str = None,
        full_refresh_days: int = 0,
    ) -> dict:
        """
        Update multiple tickers.

        Args:
            tickers_map: Dict of yfinance_ticker -> col_name. If None, extracted
                         from yfinance_downloader_v5.py.
            start: Start date override for initial population.
            end: End date override (YYYY-MM-DD). Defaults to today.
            full_refresh_days: If > 0, re-download last N days.

        Returns:
            Dict of stats: {updated, skipped, failed}.
        """
        if tickers_map is None:
            tickers_map = load_ticker_universe()

        if not tickers_map:
            logger.error("No tickers to update")
            return self._stats

        tickers = list(dict(tickers_map).keys())
        total = len(tickers)
        logger.info(f"Updating {total} tickers...")

        self._stats = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "quarantined": 0,
        }

        if self.max_workers <= 1:
            # Sequential mode
            for i, ticker in enumerate(tickers, 1):
                logger.debug(f"[{i}/{total}] Processing {ticker}...")
                success = self.update_ticker(
                    ticker,
                    start=start,
                    end=end,
                    full_refresh_days=full_refresh_days,
                )

                # Connection health check: if IBKR is in use and a ticker
                # failed, verify the connection is still alive.
                if not success and self.ibkr is not None:
                    try:
                        if not self.ibkr.isConnected():
                            logger.warning(
                                "IBKR Gateway disconnected after [%d/%d] %s. "
                                "Attempting reconnect...",
                                i,
                                total,
                                ticker,
                            )
                            reconnected = False
                            for attempt in range(1, 4):
                                logger.info("Reconnect attempt %d/3...", attempt)
                                if self.ibkr.connect_gateway():
                                    logger.info("Reconnected to IBKR Gateway.")
                                    reconnected = True
                                    break
                                time.sleep(5 * attempt)
                            if not reconnected:
                                logger.error(
                                    "Could not reconnect to IBKR Gateway. "
                                    "Continuing with yfinance only for "
                                    "remaining %d tickers.",
                                    total - i,
                                )
                                self.ibkr = None
                    except Exception:
                        pass  # isConnected() may not exist on all objects

                done = (
                    self._stats["updated"]
                    + self._stats["skipped"]
                    + self._stats["failed"]
                )
                if done % 100 == 0 and done > 0:
                    logger.info(
                        f"Progress: {done}/{total} "
                        f"(updated={self._stats['updated']}, "
                        f"skipped={self._stats['skipped']}, "
                        f"failed={self._stats['failed']}, "
                        f"quarantined={self._stats['quarantined']})"
                    )
                time.sleep(UPDATER_CONFIG["delay_between_tickers"])
        else:
            # Threaded mode
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, ticker in enumerate(tickers):
                    future = executor.submit(
                        self.update_ticker,
                        ticker,
                        start=start,
                        end=end,
                        full_refresh_days=full_refresh_days,
                    )
                    futures[future] = (i + 1, ticker)

                last_logged_progress = 0
                for future in as_completed(futures):
                    idx, ticker = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"[{idx}/{total}] {ticker}: Unexpected error: {e}")
                        with self._stats_lock:
                            self._stats["failed"] += 1

                    with self._stats_lock:
                        done = (
                            self._stats["updated"]
                            + self._stats["skipped"]
                            + self._stats["failed"]
                        )
                    # Log progress at 25-ticker intervals, avoiding duplicates
                    progress_bucket = done // 25 if done < total else total
                    if progress_bucket > last_logged_progress:
                        last_logged_progress = progress_bucket
                        logger.info(
                            f"Progress: {done}/{total} "
                            f"(updated={self._stats['updated']}, "
                            f"skipped={self._stats['skipped']}, "
                            f"failed={self._stats['failed']}, "
                            f"quarantined={self._stats['quarantined']})"
                        )

        # Summary
        if self._stats["skipped"] > 0 and self._stats["updated"] == 0:
            logger.info(
                f"Update complete: All {self._stats['skipped']} tickers already up-to-date. "
                f"Nothing to download."
            )
        else:
            logger.info(
                f"Update complete: {self._stats['updated']} updated, "
                f"{self._stats['skipped']} skipped, {self._stats['failed']} failed, "
                f"{self._stats['quarantined']} quarantined"
            )

        if self._failure_signatures:
            top_errors = self._failure_signatures.most_common(5)
            logger.info("Top failure signatures this run:")
            for signature, count in top_errors:
                logger.info(f"  [{count}x] {signature}")

        return self._stats

    def _fetch_with_failover(self, ticker: str, start: str, end: str) -> tuple:
        """
        Fetch data from available sources with source priority based on asset type.

        Source priority:
        - **Stocks/ETFs**: yfinance primary (provides Adj Close with split/dividend
          adjustments that IBKR does not). IBKR fallback if yfinance fails.
        - **Forex**: IBKR primary (IDEALPRO MIDPOINT, free, high quality -- yfinance
          forex data has gaps and stale quotes). yfinance fallback if IBKR unavailable.

        The --source flag can override this:
        - --source yfinance: always use yfinance (skip IBKR entirely)
        - --source ibkr: always use IBKR (skip yfinance entirely)
        - --source auto: asset-type-based priority (default)

        Args:
            ticker: yfinance-format ticker
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            (DataFrame, source_name) or (None, None) on failure.
        """
        # FRED tickers always route to FRED regardless of --source flag.
        if _HAS_FRED and is_fred_ticker(ticker):
            return self._try_fred(ticker, start, end)

        # Detect if this is a forex pair (e.g., EURUSD=X, USDJPY=X)
        forex = False
        if _HAS_IBKR:
            from algos.common.ibkr_downloader import is_forex_ticker

            forex = is_forex_ticker(ticker)

        # Forced source overrides
        if self.source == "yfinance":
            return self._try_yfinance(ticker, start, end)
        if self.source == "ibkr":
            # Skip ALL tickers that IBKR cannot serve.
            # These would fail with "No security definition" and waste retries.
            #
            # Categories:
            #   ^VIX, ^SPX     — Indices (not securities, use secType=IND)
            #   =F             — Futures (need different contract spec)
            #   .VN, .AAA, .TW  — No IBKR access (Vietnam, Thailand, Taiwan)
            #   .JK, .IL       — No IBKR subscription (Indonesia, India)
            #   .RO, .KL       — No IBKR access (Romania, Malaysia numerics)
            #   GMET, etc.     — Mutual funds (secType=FUND, not STK)
            #   Note: =X forex is NOW handled via contract map (secType=CASH)
            _IBKR_SKIP_PREFIXES = ("^",)
            _IBKR_SKIP_SUFFIXES = (
                "=F",
                ".VN",
                ".AAA",
                ".TW",
                ".JK",
                ".IL",
                ".RO",
                ".KL",
                ".IR",
            )
            _IBKR_SKIP_EXACT = {"GMLGX", "SWISX"}

            skip = False
            for pfx in _IBKR_SKIP_PREFIXES:
                if ticker.startswith(pfx):
                    skip = True
                    break
            if not skip:
                for sfx in _IBKR_SKIP_SUFFIXES:
                    if ticker.endswith(sfx):
                        skip = True
                        break
            if not skip and ticker in _IBKR_SKIP_EXACT:
                skip = True

            if skip:
                logger.info(
                    "[%s] Skipping — not available via IBKR. "
                    "Use --source auto to fall back to yfinance.",
                    ticker,
                )
                return (None, None)
            return self._try_ibkr(ticker, start, end)
        if self.source == "fred":
            return self._try_fred(ticker, start, end)

        # Auto mode: route by asset type
        if forex:
            # Forex: IBKR primary (high quality), yfinance fallback
            result = self._try_ibkr(ticker, start, end)
            if result[0] is not None:
                return result
            if self.ibkr is not None:
                logger.info(f"[{ticker}] IBKR forex failed, falling back to yfinance")
            return self._try_yfinance(ticker, start, end)
        else:
            # Stocks/ETFs: yfinance primary (has Adj Close), IBKR fallback
            result = self._try_yfinance(ticker, start, end)
            if result[0] is not None:
                return result
            if self.ibkr is not None:
                logger.info(f"[{ticker}] yfinance failed, falling back to IBKR")
            return self._try_ibkr(ticker, start, end)

    def _try_yfinance(self, ticker: str, start: str, end: str) -> tuple:
        """Try downloading from yfinance with auto_adjust=False (preserves Adj Close)."""
        local_attempts = 2
        for local_attempt in range(local_attempts):
            try:
                degraded_mode = self._in_yf_degraded_mode()
                lock_obj = (
                    self._yf_serial_lock
                    if degraded_mode
                    else self._yf_concurrency_guard
                )
                if degraded_mode and local_attempt == 0:
                    logger.info(
                        "High yfinance error rate detected; using serial yfinance mode"
                    )

                with lock_obj:
                    df = resilient_download_single(
                        ticker,
                        start=start,
                        end=end,
                        auto_adjust=False,  # Keep both Close (unadjusted) and Adj Close
                        max_retries=2,
                        progress=False,
                        ignore_tz=True,
                    )
                if df is not None and not df.empty:
                    self._record_yf_outcome(success=True)
                    return (df, "yfinance")
                self._record_yf_outcome(success=False)
            except Exception as e:
                self._record_yf_outcome(success=False)
                self._record_failure_signature("yfinance", e)
                logger.debug(f"yfinance failed for {ticker}: {e}")
                if local_attempt < local_attempts - 1 and self._is_transient_error(e):
                    backoff = random.uniform(1.0, 3.0) * (local_attempt + 1)
                    time.sleep(backoff)
                else:
                    break
        return (None, None)

    def _try_fred(self, ticker: str, start: str, end: str) -> tuple:
        """Try downloading from FRED API for macro/rate series."""
        if not _HAS_FRED:
            logger.debug("FRED downloader not available")
            return (None, None)
        if not is_fred_available():
            logger.warning(
                "FRED_API_KEY not set. Cannot fetch FRED data. "
                "Set FRED_API_KEY env var with your free key from "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return (None, None)
        try:
            series_id = parse_fred_ticker(ticker) if is_fred_ticker(ticker) else ticker
            df = download_fred_series(series_id, start=start, end=end)
            if df is not None and not df.empty:
                return (df, "fred")
            logger.debug(f"FRED returned no data for {ticker}")
        except Exception as e:
            self._record_failure_signature("fred", e)
            logger.debug(f"FRED failed for {ticker}: {e}")
        return (None, None)

    def _try_ibkr(self, ticker: str, start: str, end: str) -> tuple:
        """Try downloading from IBKR. Note: IBKR does not provide Adj Close for stocks."""
        if self.ibkr is None:
            return (None, None)
        try:
            df = self.ibkr.download_single(ticker, start, end)
            if df is not None and not df.empty:
                return (df, "ibkr")
            logger.debug(f"IBKR returned no data for {ticker}")
        except PacingViolation:
            self._record_failure_signature("ibkr", RuntimeError("pacing violation"))
            logger.info(f"IBKR pacing limit for {ticker}")
        except Exception as e:
            self._record_failure_signature("ibkr", e)
            logger.debug(f"IBKR failed for {ticker}: {e}")
        return (None, None)


# ===========================================================================
# STATUS DISPLAY
# ===========================================================================


def show_status(store: MarketDataStore) -> None:
    """Print a status summary of the parquet store."""
    tickers = store.list_tickers()
    if not tickers:
        print("Parquet store is empty. Run with --init to populate.")
        return

    print(f"\nMarket Data Store: {store.data_dir}")
    print(f"Total tickers: {len(tickers)}")

    # Collect stats
    stale = []
    fresh = []
    for t in tickers:
        if store.check_freshness(t, max_age_hours=48):
            fresh.append(t)
        else:
            stale.append(t)

    print(f"Fresh (<48h): {len(fresh)}")
    print(f"Stale (>48h): {len(stale)}")

    if stale and len(stale) <= 20:
        print(f"Stale tickers: {', '.join(stale)}")
    elif stale:
        print(f"Stale tickers (first 20): {', '.join(stale[:20])}...")

    # Show date range summary
    earliest = None
    latest = None
    for t in tickers[:50]:  # Sample first 50
        dr = store.get_date_range(t)
        if dr:
            if earliest is None or dr[0] < earliest:
                earliest = dr[0]
            if latest is None or dr[1] > latest:
                latest = dr[1]

    if earliest and latest:
        print(f"Date range (sampled): {earliest} to {latest}")

    print()


# ===========================================================================
# CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Market data store updater. Populates and maintains the parquet store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial population (one-time)
  python -m algos.common.update_market_data --init --lookback-days 1825

  # Daily incremental update
  python -m algos.common.update_market_data

  # Update specific tickers
  python -m algos.common.update_market_data --tickers SPY NVDA AAPL

  # Weekly full-refresh (catch corrections)
  python -m algos.common.update_market_data --full-refresh 5

  # Seed from existing CSV data
  python -m algos.common.update_market_data --seed-from-csv

  # Export portfolio CSV for portimization.py
  python -m algos.common.update_market_data --export-csv data/financial_data_combined_prices.csv

  # Export one ticker raw OHLCV for validation
  python -m algos.common.update_market_data --export-ticker-ohlc EURUSD=X --start 2025-01-01

  # Show store status
  python -m algos.common.update_market_data --status
        """,
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Initial population mode. Downloads full history for all tickers.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to update (space-separated).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Number of days of history (default: 1825 = 5 years).",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Explicit start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=str, default=None, help="Explicit end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--full-refresh",
        type=int,
        default=0,
        metavar="DAYS",
        help="Re-download last N trading days to catch corrections.",
    )
    parser.add_argument(
        "--seed-from-csv",
        action="store_true",
        help="Seed parquet store from existing CSV files in data/.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Export multi-ticker prices to CSV for portimization.py.",
    )
    parser.add_argument(
        "--export-ticker-ohlc",
        type=str,
        default=None,
        metavar="TICKER",
        help=("Export raw OHLCV parquet data for one ticker to CSV (e.g., EURUSD=X)."),
    )
    parser.add_argument(
        "--ohlc-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional output path for --export-ticker-ohlc.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        metavar="FRAC",
        help=(
            "Min fraction of non-NaN trading days required per ticker when exporting. "
            "Tickers below this are excluded (e.g., IPO'd after start date). "
            "Default: 0.8 (80%%)."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "yfinance", "ibkr", "fred"],
        default="ibkr",
        help=(
            "Data source: 'ibkr' (default, post-migration) uses IBKR for all "
            "tickers via conId-based lookup. 'auto' uses yfinance for stocks + "
            "IBKR for forex (legacy). 'yfinance' or 'fred' force a specific source."
        ),
    )
    # Default IBKR port: read from IBKR_PORT env var, otherwise 4002.
    # IB Gateway: paper=4002, live=4001. TWS: paper=7497, live=7496.
    _default_ibkr_port = int(os.environ.get("IBKR_PORT", "4002"))
    parser.add_argument(
        "--ibkr-port",
        "--port",
        type=int,
        default=_default_ibkr_port,
        help=(
            f"IBKR API port (default: {_default_ibkr_port}, override with "
            f"IBKR_PORT env var). IB Gateway paper=4002, live=4001. "
            f"TWS paper=7497, live=7496."
        ),
    )
    parser.add_argument(
        "--ibkr-client-id",
        type=int,
        default=None,
        help="Explicit IBKR client id (default: auto-allocated by the shared "
        "client-id rotator, which guarantees a unique, collision-checked id).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of download threads (default: 4).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Max retries per ticker on all-sources-failed.",
    )
    parser.add_argument(
        "--skip-fresh",
        action="store_true",
        help=(
            "Skip tickers that already have IBKR-sourced data within the last "
            "7 days. Useful for resuming after a Gateway crash mid-download."
        ),
    )
    parser.add_argument(
        "--status", action="store_true", help="Show store status and exit."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override parquet store directory."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging (DEBUG level)."
    )

    args = parser.parse_args()

    # Setup logging -- ensure directory exists before creating FileHandler
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = _PROJECT_ROOT / "data" / "market_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                str(log_dir / "updater.log"),
                mode="a",
            ),
        ],
    )

    store = MarketDataStore(data_dir=args.data_dir)

    # Status check
    if args.status:
        show_status(store)
        return

    # Seed from CSV
    if args.seed_from_csv:
        count = seed_from_csv(store)
        print(f"Seeded {count} tickers from CSV files.")
        return

    # Export one ticker raw OHLCV
    if args.export_ticker_ohlc:
        ticker = args.export_ticker_ohlc.strip()
        if not ticker:
            print("ERROR: --export-ticker-ohlc requires a ticker symbol")
            sys.exit(1)

        start = args.start
        end = args.end or datetime.now().strftime("%Y-%m-%d")
        if not start:
            lookback = args.lookback_days or UPDATER_CONFIG["default_lookback_days"]
            start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        df = store.get_ohlcv_raw(ticker, start=start, end=end)
        if df is None or df.empty:
            print(f"No data found for {ticker} in range {start} to {end}.")
            sys.exit(1)

        out_path = args.ohlc_out
        if not out_path:
            safe_ticker = store.normalize_ticker(ticker).replace("/", "_")
            out_path = str(_PROJECT_ROOT / "data" / f"{safe_ticker}_ohlc_raw.csv")

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=True)

        required_cols = ["open", "high", "low", "close", "volume", "adj_close"]
        available_cols = [col for col in required_cols if col in df.columns]
        nan_counts = df[available_cols].isna().sum().to_dict() if available_cols else {}

        print(f"Exported {ticker}: rows={len(df)}, cols={len(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Saved: {output_path}")
        if nan_counts:
            print(f"NaN counts: {nan_counts}")
        return

    # Export CSV
    if args.export_csv:
        tickers_map = load_ticker_universe()
        if not tickers_map:
            print("No tickers found. Cannot export.")
            sys.exit(1)

        start = args.start
        end = args.end or datetime.now().strftime("%Y-%m-%d")
        if not start:
            lookback = args.lookback_days or UPDATER_CONFIG["default_lookback_days"]
            start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        df = store.export_portfolio_csv(
            tickers_map,
            start,
            end,
            args.export_csv,
            min_coverage=args.min_coverage,
        )
        if df.empty:
            print("WARNING: Export produced empty DataFrame. Is the store populated?")
        else:
            print(f"Exported {df.shape} to {args.export_csv}")
        return

    # Acquire PID lock
    lock = PIDLock(LOCKFILE_PATH)
    if not lock.acquire():
        sys.exit(1)

    try:
        # Determine tickers
        if args.tickers:
            expanded_tickers = _expand_ticker_args(args.tickers)
            tickers_map = {t: t for t in expanded_tickers}
        else:
            tickers_map = load_ticker_universe()

        if not tickers_map:
            logger.error(
                "No tickers to update. Use --tickers or ensure yfinance_downloader_v5.py has active tickers."
            )
            sys.exit(1)

        # Determine start date
        start = args.start
        if args.init or start:
            if not start:
                lookback = args.lookback_days or UPDATER_CONFIG["default_lookback_days"]
                start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
            logger.info(f"Init mode: downloading from {start}")

        # Create IBKR downloader if available and requested
        ibkr = None
        if _HAS_IBKR and args.source in ("auto", "ibkr"):
            if is_gateway_available(port=args.ibkr_port):
                if args.source == "ibkr":
                    logger.info(
                        f"IB Gateway/TWS detected on port {args.ibkr_port} -- "
                        f"using IBKR as primary source for all tickers "
                        f"(conId-based lookup, forex via IDEALPRO MIDPOINT)."
                    )
                else:
                    logger.info(
                        f"IB Gateway/TWS detected on port {args.ibkr_port} -- "
                        f"auto mode: IBKR for forex pairs, yfinance for stocks "
                        f"(provides Adj Close)."
                    )
                connect_attempts = 3
                for attempt in range(1, connect_attempts + 1):
                    # Each attempt gets a FRESH downloader. With no explicit
                    # --ibkr-client-id, the downloader auto-allocates a unique,
                    # collision-checked id from the shared rotator (registry +
                    # live 326 probe), so we never reuse an id another session
                    # holds. The old "base + (attempt-1)" scheme could collide on
                    # all three attempts if the base id was busy -> error 326.
                    ibkr = IBKRDataDownloader(
                        port=args.ibkr_port, client_id=args.ibkr_client_id
                    )
                    if ibkr.connect_gateway():
                        logger.info(
                            f"IBKR handshake successful on attempt "
                            f"{attempt}/{connect_attempts} with clientId={ibkr.client_id}"
                        )
                        break
                    # Release the id this failed attempt reserved before retrying.
                    try:
                        ibkr.disconnect_gateway()
                    except Exception:
                        pass
                    ibkr = None
                    if attempt < connect_attempts:
                        logger.warning(
                            f"IBKR handshake failed (attempt {attempt}/{connect_attempts}); "
                            f"retrying in 5s with a fresh client id..."
                        )
                        time.sleep(5)

                if ibkr is None:
                    if args.source == "ibkr":
                        logger.error(
                            f"Could not connect to IB Gateway/TWS on port {args.ibkr_port} in --source ibkr mode. "
                            "Start IB Gateway/TWS API and retry."
                        )
                        sys.exit(1)
                    logger.warning(
                        f"Could not connect to IB Gateway/TWS on port {args.ibkr_port}. "
                        "Forex pairs will fall back to yfinance (lower quality)."
                    )
                    ibkr = None
            else:
                if args.source == "ibkr":
                    logger.error(
                        f"IB Gateway/TWS not detected on port {args.ibkr_port} in --source ibkr mode. "
                        "Start IB Gateway/TWS API (or update port handling) and retry."
                    )
                    sys.exit(1)
                logger.info(
                    f"IB Gateway/TWS not detected on port {args.ibkr_port} -- all tickers use yfinance. "
                    "Start IB Gateway/TWS API for higher quality forex data."
                )
        elif args.source == "ibkr":
            logger.error(
                "IBKR downloader module is unavailable but --source ibkr was requested."
            )
            sys.exit(1)

        # Create updater
        updater = MarketDataUpdater(
            store=store,
            ibkr_downloader=ibkr,
            max_retries=args.max_retries,
            max_workers=args.workers,
            source=args.source,
            skip_fresh=getattr(args, "skip_fresh", False),
        )

        # Run updates
        stats = updater.update_all(
            tickers_map=tickers_map,
            start=start,
            end=args.end,
            full_refresh_days=args.full_refresh,
        )

        # Final summary
        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} tickers failed. Check logs for details.")
            sys.exit(1)

    finally:
        # Disconnect IBKR if connected
        if ibkr is not None:
            try:
                ibkr.disconnect_gateway()
            except Exception:
                pass
        lock.release()


if __name__ == "__main__":
    main()
