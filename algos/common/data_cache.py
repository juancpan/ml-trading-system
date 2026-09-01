"""
Centralized data caching system for improved performance.
Reduces redundant API calls and file I/O operations.
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Any
import threading
import sys
import os

# Import resilient downloader
try:
    from algos.common.yf_downloader import resilient_download_single, resilient_download

    _HAS_RESILIENT_DOWNLOADER = True
except ImportError:
    # Fallback: try relative path for when running from within algos/
    try:
        _script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _project_root = _script_dir.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from algos.common.yf_downloader import (
            resilient_download_single,
            resilient_download,
        )

        _HAS_RESILIENT_DOWNLOADER = True
    except ImportError:
        import yfinance as yf

        _HAS_RESILIENT_DOWNLOADER = False
        print(
            "Warning: yf_downloader not available, falling back to bare yfinance (no retry logic)"
        )


class DataCache:
    """
    Thread-safe singleton cache for market data.
    Implements LRU-style caching with TTL support.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.cache_dir = Path("data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache with TTL
        self._memory_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl = timedelta(hours=24)  # Default TTL

        # Configuration
        self.max_memory_items = 100
        self.enable_disk_cache = True

    def _generate_cache_key(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> str:
        """Generate unique cache key for data request."""
        key_str = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid based on TTL."""
        return datetime.now() - timestamp < self._cache_ttl

    def get(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache (memory first, then disk).
        Returns None if not found or expired.
        """
        cache_key = self._generate_cache_key(ticker, start, end, interval)

        # Check memory cache first
        if cache_key in self._memory_cache:
            data, timestamp = self._memory_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return data.copy()
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]

        # Check disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        data, timestamp = pickle.load(f)

                    if self._is_cache_valid(timestamp):
                        # Load into memory cache for faster access
                        self._add_to_memory_cache(cache_key, data)
                        return data.copy()
                    else:
                        # Remove expired file
                        cache_file.unlink()
                except Exception:
                    # Corrupted cache file, remove it
                    cache_file.unlink(missing_ok=True)

        return None

    def set(
        self, ticker: str, start: str, end: str, interval: str, data: pd.DataFrame
    ) -> None:
        """Store data in cache (both memory and disk)."""
        cache_key = self._generate_cache_key(ticker, start, end, interval)
        timestamp = datetime.now()

        # Add to memory cache
        self._add_to_memory_cache(cache_key, data)

        # Save to disk cache
        if self.enable_disk_cache:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump((data.copy(), timestamp), f)
            except Exception as e:
                print(f"Warning: Could not save to disk cache: {e}")

    def _add_to_memory_cache(self, key: str, data: pd.DataFrame) -> None:
        """Add item to memory cache with LRU eviction."""
        # Implement simple LRU by removing oldest items when at capacity
        if len(self._memory_cache) >= self.max_memory_items:
            # Remove oldest entry (first item in dict)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = (data.copy(), datetime.now())

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()

        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for monitoring."""
        disk_files = (
            list(self.cache_dir.glob("*.pkl")) if self.enable_disk_cache else []
        )

        return {
            "memory_items": len(self._memory_cache),
            "disk_files": len(disk_files),
            "memory_size_mb": sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024
                for df, _ in self._memory_cache.values()
            ),
            "ttl_hours": self._cache_ttl.total_seconds() / 3600,
        }


class OptimizedDataLoader:
    """
    Optimized data loader with caching and batch processing.
    """

    def __init__(self, cache_ttl_hours: float = 24):
        self.cache = DataCache()
        self.cache._cache_ttl = timedelta(hours=cache_ttl_hours)

    @staticmethod
    def _store_only_mode() -> bool:
        """Return True when network downloads should be disabled."""
        value = os.getenv("MARKET_DATA_STORE_ONLY", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def load_data(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load data with caching support.
        """
        # Try cache first
        if use_cache:
            cached_data = self.cache.get(ticker, start, end, interval)
            if cached_data is not None:
                return cached_data

        # Try parquet store first (local, fast, no network)
        try:
            from algos.common.market_data_store import MarketDataStore

            _store = MarketDataStore()
            if _store.has_ticker(ticker):
                parquet_data = _store.get_ohlcv(ticker, start, end, use_adj_close=True)
                if parquet_data is not None and not parquet_data.empty:
                    if use_cache:
                        self.cache.set(ticker, start, end, interval, parquet_data)
                    return parquet_data
        except ImportError:
            pass
        except Exception as e:
            print(
                f"Parquet store read failed for {ticker}, falling back to download: {e}"
            )

        if self._store_only_mode():
            print(
                f"Store-only mode enabled; no parquet data available for {ticker} in {start} to {end}"
            )
            return None

        # Download with resilient retry logic (fallback)
        try:
            if _HAS_RESILIENT_DOWNLOADER:
                data = resilient_download_single(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
            else:
                import yfinance as yf

                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    ignore_tz=True,
                )

            if data is not None and not data.empty:
                # Store in cache
                if use_cache:
                    self.cache.set(ticker, start, end, interval, data)
                return data

        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

        return None

    def load_multiple(
        self,
        tickers: list,
        start: str,
        end: str,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers efficiently.
        Uses batch download when possible.
        """
        if self._store_only_mode():
            results = {}
            for ticker in tickers:
                data = self.load_data(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    use_cache=use_cache,
                )
                if data is not None and not data.empty:
                    results[ticker] = data
            return results

        results = {}
        uncached_tickers = []

        # Check cache first
        if use_cache:
            for ticker in tickers:
                cached_data = self.cache.get(ticker, start, end, interval)
                if cached_data is not None:
                    results[ticker] = cached_data
                else:
                    uncached_tickers.append(ticker)
        else:
            uncached_tickers = tickers

        # Batch download uncached tickers with resilient retry
        if uncached_tickers:
            try:
                if _HAS_RESILIENT_DOWNLOADER:
                    batch_data = resilient_download(
                        uncached_tickers,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                    )
                else:
                    import yfinance as yf

                    batch_data = yf.download(
                        uncached_tickers,
                        start=start,
                        end=end,
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                        group_by="ticker",
                        ignore_tz=True,
                    )

                # Process batch results
                if batch_data is not None and not batch_data.empty:
                    if len(uncached_tickers) == 1:
                        # Single ticker returns DataFrame directly
                        results[uncached_tickers[0]] = batch_data
                        if use_cache:
                            self.cache.set(
                                uncached_tickers[0], start, end, interval, batch_data
                            )
                    elif isinstance(batch_data.columns, pd.MultiIndex):
                        # Multiple tickers return MultiIndex DataFrame
                        for ticker in uncached_tickers:
                            try:
                                if ticker in batch_data.columns.get_level_values(0):
                                    ticker_data = batch_data[ticker].dropna()
                                elif ticker in batch_data.columns.get_level_values(1):
                                    ticker_data = batch_data.xs(
                                        ticker, axis=1, level=1
                                    ).dropna()
                                else:
                                    continue
                                if not ticker_data.empty:
                                    results[ticker] = ticker_data
                                    if use_cache:
                                        self.cache.set(
                                            ticker, start, end, interval, ticker_data
                                        )
                            except (KeyError, TypeError):
                                continue

            except Exception as e:
                print(f"Error in batch download: {e}")
                # Fall back to individual downloads
                for ticker in uncached_tickers:
                    data = self.load_data(ticker, start, end, interval, use_cache)
                    if data is not None:
                        results[ticker] = data

        return results
