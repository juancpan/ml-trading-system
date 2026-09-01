"""
External Data Loader for feature engineering.
Fetches and caches external market data (VIX, benchmarks, yields, etc.)
that can be used as features alongside primary ticker data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Import resilient downloader with yfinance fallback
try:
    from algos.common.yf_downloader import resilient_download_single

    _HAS_RESILIENT_DOWNLOADER = True
except ImportError:
    try:
        import sys
        import os

        _script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _project_root = _script_dir.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from algos.common.yf_downloader import resilient_download_single

        _HAS_RESILIENT_DOWNLOADER = True
    except ImportError:
        try:
            import yfinance as yf
        except ImportError:
            yf = None
        _HAS_RESILIENT_DOWNLOADER = False


# Default cache directory
_CACHE_DIR = None

# In-memory session cache for external data (avoids repeated parquet reads in WFOV loops)
# Key: (ticker, interval) -> DataFrame. Cleared between sessions via clear_session_cache().
_SESSION_CACHE: dict[str, pd.DataFrame] = {}


def clear_session_cache():
    """Clear the in-memory session cache (call between WFOV sessions or workflow runs)."""
    global _SESSION_CACHE
    _SESSION_CACHE.clear()


def _get_cache_dir() -> Path:
    """Get or create the external data cache directory."""
    global _CACHE_DIR
    if _CACHE_DIR is not None:
        return _CACHE_DIR

    # Find project root
    try:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
    except NameError:
        project_root = Path.cwd()
        for _ in range(5):
            if (project_root / "algos").is_dir():
                break
            if project_root == project_root.parent:
                break
            project_root = project_root.parent

    _CACHE_DIR = project_root / "data" / "external"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _try_parquet_store(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Try loading ticker data from local parquet store (handles FRED: prefix)."""
    try:
        from algos.common.market_data_store import MarketDataStore

        store = MarketDataStore()
        df = store.get_ohlcv(ticker, start=start, end=end, use_adj_close=True)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def _is_fred_ticker(ticker: str) -> bool:
    """Check if ticker uses the FRED: prefix convention."""
    return ticker.strip().upper().startswith("FRED:")


def _download_ticker_data(
    ticker: str, start: str, end: str, interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a ticker.

    Resolution order:
    1. Local parquet store (always tried first; required for FRED tickers).
    2. yfinance resilient downloader (for market tickers).
    """
    # Always try parquet store first (required path for FRED tickers).
    store_df = _try_parquet_store(ticker, start, end)
    if store_df is not None:
        return store_df

    # FRED tickers must come from the store; don't attempt yfinance.
    if _is_fred_ticker(ticker):
        print(
            f"Warning: FRED ticker {ticker} not found in parquet store. "
            f'Run: python -m algos.common.update_market_data --tickers "{ticker}"'
        )
        return None

    # Respect MARKET_DATA_STORE_ONLY mode — no network downloads.
    if os.environ.get("MARKET_DATA_STORE_ONLY"):
        print(
            f"Warning: External ticker {ticker} not in parquet store "
            f"and MARKET_DATA_STORE_ONLY is set. Skipping download."
        )
        return None

    try:
        if _HAS_RESILIENT_DOWNLOADER:
            df = resilient_download_single(
                ticker, start=start, end=end, interval=interval, progress=False
            )
        elif yf is not None:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        else:
            print(
                f"Warning: No download backend available for external data ({ticker})"
            )
            return None

        if df is None or df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    except Exception as e:
        print(f"Warning: Failed to download external data for {ticker}: {e}")
        return None


def fetch_external_series(
    ticker: str,
    start: str,
    end: str,
    column: str = "close",
    interval: str = "1d",
    cache_ttl_hours: int = 24,
    buffer_days: int = 100,
) -> Optional[pd.Series]:
    """
    Fetch a single price series for an external ticker, with caching.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., '^VIX', 'SPY', 'TLT')
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD'
        column: Which column to return ('close', 'open', 'high', 'low', 'volume')
        interval: Data interval (default '1d')
        cache_ttl_hours: Cache time-to-live in hours (default 24)
        buffer_days: Extra days to fetch before start for indicator warmup

    Returns:
        pd.Series with DatetimeIndex, or None if fetching fails
    """
    # Handle COT: prefixed tickers (Commitment of Traders data)
    if ticker.upper().startswith("COT:"):
        from algos.common.cot_downloader import fetch_cot_net_positioning

        currency = ticker.split(":")[1].strip()
        return fetch_cot_net_positioning(currency=currency, start=start, end=end)

    cache_dir = _get_cache_dir()

    # Sanitize ticker for filename (^VIX -> _VIX, FRED:XYZ -> FRED_XYZ)
    safe_ticker = (
        ticker.replace("^", "_").replace("/", "-").replace(".", "_").replace(":", "_")
    )
    cache_file = cache_dir / f"{safe_ticker}_{interval}.parquet"

    # L0: Check in-memory session cache first (avoids all disk I/O in WFOV loops)
    session_key = f"{safe_ticker}_{interval}"
    if session_key in _SESSION_CACHE:
        cached_df = _SESSION_CACHE[session_key]
        if cached_df.index.min() <= pd.Timestamp(
            start
        ) and cached_df.index.max() >= pd.Timestamp(end) - timedelta(days=5):
            result_df = cached_df.loc[start:end]
            if not result_df.empty and column in result_df.columns:
                return result_df[column].rename(f"ext_{safe_ticker}_{column}")

    # L1: Check disk cache freshness
    use_cache = False
    if cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age < timedelta(hours=cache_ttl_hours):
            use_cache = True

    if use_cache:
        try:
            cached_df = pd.read_parquet(cache_file)
            cached_df.index = pd.to_datetime(cached_df.index)

            # Store in session cache for subsequent calls
            _SESSION_CACHE[session_key] = cached_df

            # Check if cached data covers our date range
            if cached_df.index.min() <= pd.Timestamp(
                start
            ) and cached_df.index.max() >= pd.Timestamp(end) - timedelta(days=5):
                # Slice to requested range
                result_df = cached_df.loc[start:end]
                if not result_df.empty and column in result_df.columns:
                    return result_df[column].rename(f"ext_{safe_ticker}_{column}")
        except Exception:
            pass  # Fall through to download

    # Download with buffer for indicator warmup
    buffered_start = (pd.Timestamp(start) - timedelta(days=buffer_days)).strftime(
        "%Y-%m-%d"
    )
    # Add small buffer at end to handle timezone/market close timing
    buffered_end = (pd.Timestamp(end) + timedelta(days=5)).strftime("%Y-%m-%d")

    df = _download_ticker_data(ticker, buffered_start, buffered_end, interval)
    if df is None or df.empty:
        return None

    # Standardize column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Handle 'adj_close' -> 'close' if auto_adjust wasn't used
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]

    # Save to disk cache (full buffered range)
    try:
        df.to_parquet(cache_file)
    except Exception:
        pass  # Non-fatal: caching is optional

    # Store in session cache for subsequent calls within the same process
    _SESSION_CACHE[session_key] = df

    # Slice to requested range and return
    result_df = df.loc[start:end]
    if result_df.empty or column not in result_df.columns:
        # Try case-insensitive column match
        col_lower_map = {c.lower(): c for c in result_df.columns}
        if column.lower() in col_lower_map:
            column = col_lower_map[column.lower()]
        else:
            print(
                f"Warning: Column '{column}' not found in external data for {ticker}. "
                f"Available: {list(result_df.columns)}"
            )
            return None

    return result_df[column].rename(f"ext_{safe_ticker}_{column}")


def fetch_multiple_external(
    external_configs: list[dict],
    start: str,
    end: str,
    interval: str = "1d",
    primary_index: Optional[pd.DatetimeIndex] = None,
    cache_ttl_hours: int = 24,
) -> dict[str, pd.Series]:
    """
    Fetch multiple external data series and align to a primary index.

    Args:
        external_configs: List of dicts with keys:
            - ticker: str (e.g., '^VIX')
            - column: str (e.g., 'close')
            - name: str (optional, custom feature name)
        start: Start date
        end: End date
        interval: Data interval
        primary_index: DatetimeIndex to align to (forward-fills gaps)
        cache_ttl_hours: Cache TTL

    Returns:
        Dict mapping feature_name -> pd.Series (aligned to primary_index if provided)
    """
    result = {}

    for ext_cfg in external_configs:
        ticker = ext_cfg["ticker"]
        column = ext_cfg.get("column", "close")
        name = ext_cfg.get("name", None)

        series = fetch_external_series(
            ticker=ticker,
            start=start,
            end=end,
            column=column,
            interval=interval,
            cache_ttl_hours=cache_ttl_hours,
        )

        if series is None:
            print(
                f"Warning: Could not fetch external data for {ticker} ({column}). "
                f"Feature will be skipped."
            )
            continue

        # Custom name or auto-generated
        if name is None:
            safe_ticker = (
                ticker.replace("^", "").replace("/", "-").replace(".", "_").lower()
            )
            name = f"ext_{safe_ticker}_{column}"

        # Align to primary index if provided
        if primary_index is not None:
            series = series.reindex(primary_index, method="ffill")

        result[name] = series.rename(name)

    return result


class ExternalDataCache:
    """
    Manages external data for live trading.
    Pre-fetches and maintains a rolling cache of external data series.
    """

    def __init__(
        self,
        external_configs: list[dict],
        lookback_days: int = 300,
        cache_ttl_hours: int = 12,
        logger=None,
    ):
        """
        Args:
            external_configs: List of external data configurations
            lookback_days: Days of history to maintain
            cache_ttl_hours: How often to refresh
            logger: Optional logger instance
        """
        self.external_configs = external_configs
        self.lookback_days = lookback_days
        self.cache_ttl_hours = cache_ttl_hours
        self.logger = logger
        self._cache: dict[str, pd.Series] = {}
        self._last_refresh: Optional[datetime] = None

    def refresh(self, primary_index: Optional[pd.DatetimeIndex] = None) -> None:
        """Refresh all external data series."""
        now = datetime.now()

        # Skip if recently refreshed
        if self._last_refresh is not None and (now - self._last_refresh) < timedelta(
            hours=self.cache_ttl_hours
        ):
            if self.logger:
                self.logger.debug("External data cache still fresh, skipping refresh")
            return

        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        if self.logger:
            self.logger.info(
                f"Refreshing external data cache ({len(self.external_configs)} series)"
            )

        self._cache = fetch_multiple_external(
            external_configs=self.external_configs,
            start=start_date,
            end=end_date,
            primary_index=primary_index,
            cache_ttl_hours=self.cache_ttl_hours,
        )

        self._last_refresh = now

        if self.logger:
            self.logger.info(
                f"External data cache refreshed: {list(self._cache.keys())}"
            )

    def get_data(self) -> dict[str, pd.Series]:
        """Get cached external data. Auto-refreshes if stale."""
        if self._last_refresh is None:
            self.refresh()
        return self._cache

    def get_series(self, name: str) -> Optional[pd.Series]:
        """Get a single cached series by name."""
        return self._cache.get(name)
