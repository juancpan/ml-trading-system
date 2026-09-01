"""
Parquet-backed market data store. Single source of truth for all
historical daily OHLCV data in the system.

Read path (fast, local-only, no network):
    store = MarketDataStore()
    df = store.get_ohlcv("SPY", "2020-01-01", "2025-01-01")
    multi = store.get_multi_ticker_prices(["SPY", "NVDA", "GLD"], "2020-01-01", "2025-01-01")

Write path (used only by update_market_data.py):
    store.write_ticker("SPY", df, source="yfinance")

The store NEVER fetches from the network. All data retrieval is from
local parquet files. To populate/update, use update_market_data.py.

Parquet schema per ticker file:
    date       (DatetimeIndex)  -- trading date, timezone-naive
    open       (float64)        -- unadjusted open
    high       (float64)        -- unadjusted high
    low        (float64)        -- unadjusted low
    close      (float64)        -- unadjusted close
    volume     (float64)        -- volume
    adj_close  (float64)        -- split+dividend adjusted close
    source     (string)         -- "ibkr" or "yfinance"

Storage layout:
    data/market_data/
    ├── SPY.parquet
    ├── NVDA.parquet
    ├── 8058.T.parquet
    ├── EURUSD.parquet
    └── _metadata.json          -- ticker registry + last update timestamps
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StoreWriteTemporaryError(Exception):
    """Raised when a parquet write fails due to transient system conditions."""


# Standard column names for all parquet files
PARQUET_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close", "source"]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Default store location relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data" / "market_data")


class MarketDataStore:
    """
    Parquet-backed market data store.

    Provides fast, local-only reads for all historical daily OHLCV data.
    Thread-safe for reads. Writes use atomic file replacement (write to
    .tmp then os.rename) to prevent corruption.

    Args:
        data_dir: Path to the market_data directory. Defaults to data/market_data/
                  relative to project root.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
        self._metadata_path = self.data_dir / "_metadata.json"
        self._metadata_lock = threading.RLock()
        self._write_locks = {}
        self._write_locks_lock = threading.Lock()
        self._metadata_cache: Optional[dict] = None
        self._metadata_cache_mtime: float = 0.0

    def _get_ticker_write_lock(self, normalized_ticker: str) -> threading.Lock:
        """Get/create a per-ticker write lock."""
        with self._write_locks_lock:
            if normalized_ticker not in self._write_locks:
                self._write_locks[normalized_ticker] = threading.Lock()
            return self._write_locks[normalized_ticker]

    # =========================================================================
    # TICKER NAME NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """
        Normalize ticker symbol for parquet file naming.

        Strips the yfinance '=X' suffix for forex pairs and the 'FRED:'
        prefix for FRED series so the parquet filename is clean
        (e.g., EURUSD.parquet, FEDFUNDS.parquet).

        Args:
            ticker: Raw ticker symbol (e.g., 'SPY', 'EURUSD=X', 'FRED:FEDFUNDS')

        Returns:
            Normalized ticker for file naming (e.g., 'SPY', 'EURUSD', 'FEDFUNDS')
        """
        text = ticker.strip()
        if text.upper().startswith("FRED:"):
            return text[5:]  # Strip 'FRED:' prefix
        if text.endswith("=X"):
            return text[:-2]
        return text

    @staticmethod
    def to_yfinance_ticker(normalized: str) -> str:
        """
        Convert a normalized ticker back to yfinance format.

        Detects forex pairs (e.g., EURUSD, USDJPY) and appends '=X'.
        FRED tickers are returned with their ``FRED:`` prefix restored.
        All other tickers pass through unchanged.

        Args:
            normalized: Normalized ticker (e.g., 'EURUSD', 'SPY', 'FEDFUNDS')

        Returns:
            yfinance ticker (e.g., 'EURUSD=X', 'SPY', 'FRED:FEDFUNDS')
        """
        # Common forex pairs -- 6-char codes with no dots or digits
        # This heuristic covers the standard majors/crosses
        if (
            len(normalized) == 6
            and normalized.isalpha()
            and normalized.isupper()
            and "." not in normalized
        ):
            # Check if it looks like a currency pair (e.g., EURUSD, USDJPY)
            _FOREX_CURRENCIES = {
                "EUR",
                "USD",
                "GBP",
                "JPY",
                "CHF",
                "AUD",
                "NZD",
                "CAD",
                "HKD",
                "SGD",
                "NOK",
                "SEK",
                "DKK",
                "PLN",
                "CZK",
                "HUF",
                "ILS",
                "MXN",
                "ZAR",
                "TRY",
                "INR",
                "RON",
                "AED",
                "SAR",
                "MYR",
                "BRL",
                "KRW",
                "TWD",
                "THB",
                "PHP",
            }
            base = normalized[:3]
            quote = normalized[3:]
            if base in _FOREX_CURRENCIES and quote in _FOREX_CURRENCIES:
                return f"{normalized}=X"
        return normalized

    def _ticker_path(self, ticker: str) -> Path:
        """Get the parquet file path for a ticker."""
        normalized = self.normalize_ticker(ticker)
        return self.data_dir / f"{normalized}.parquet"

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(self) -> dict:
        """Load metadata from JSON, with file-mtime caching."""
        with self._metadata_lock:
            if not self._metadata_path.exists():
                return {}

            try:
                mtime = self._metadata_path.stat().st_mtime
                if (
                    self._metadata_cache is not None
                    and mtime == self._metadata_cache_mtime
                ):
                    return dict(self._metadata_cache)

                with open(self._metadata_path, "r") as f:
                    data = json.load(f)

                self._metadata_cache = data
                self._metadata_cache_mtime = mtime
                return dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load metadata: {e}")
                return {}

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to JSON atomically."""
        with self._metadata_lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_name = tempfile.mkstemp(
                    prefix="_metadata.", suffix=".json.tmp", dir=self.data_dir
                )
                tmp_path = Path(tmp_name)
                with os.fdopen(tmp_fd, "w") as f:
                    tmp_fd = None
                    json.dump(metadata, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(self._metadata_path)
                self._metadata_cache = dict(metadata)
                self._metadata_cache_mtime = self._metadata_path.stat().st_mtime
            except OSError as e:
                logger.error(f"Could not save metadata: {e}")
            finally:
                if tmp_fd is not None:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_path is not None and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def _update_metadata_for_ticker(
        self, ticker: str, df: pd.DataFrame, source: str
    ) -> None:
        """Update metadata entry for a single ticker after a write."""
        normalized = self.normalize_ticker(ticker)
        first_date = None
        last_date = None
        if len(df) > 0:
            first_date = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
            last_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")

        metadata = self._load_metadata()
        metadata[normalized] = {
            "first_date": first_date,
            "last_date": last_date,
            "rows": len(df),
            "last_updated": datetime.now().isoformat(),
            "source": source,
        }
        self._save_metadata(metadata)

    # =========================================================================
    # READ API
    # =========================================================================

    def has_ticker(self, ticker: str) -> bool:
        """Check if a ticker exists in the store."""
        return self._ticker_path(ticker).exists()

    def list_tickers(self) -> list:
        """List all tickers in the store."""
        if not self.data_dir.exists():
            return []
        return sorted([p.stem for p in self.data_dir.glob("*.parquet")])

    def get_date_range(self, ticker: str) -> Optional[tuple]:
        """
        Get (earliest_date, latest_date) for a ticker.

        Returns:
            Tuple of (first_date, last_date) as strings, or None if ticker not found.
        """
        metadata = self._load_metadata()
        normalized = self.normalize_ticker(ticker)
        if normalized in metadata:
            m = metadata[normalized]
            return (m.get("first_date"), m.get("last_date"))

        # Fallback: read from parquet directly
        path = self._ticker_path(ticker)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=[])
            if df.index.empty:
                return None
            first_date = pd.to_datetime(df.index.min()).strftime("%Y-%m-%d")
            last_date = pd.to_datetime(df.index.max()).strftime("%Y-%m-%d")
            return (first_date, last_date)
        except Exception as e:
            logger.warning(f"Could not read date range for {ticker}: {e}")
            return None

    def get_last_updated(self, ticker: str) -> Optional[datetime]:
        """Get the last update timestamp for a ticker."""
        metadata = self._load_metadata()
        normalized = self.normalize_ticker(ticker)
        if normalized in metadata:
            ts = metadata[normalized].get("last_updated")
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass
        # Fallback: file modification time
        path = self._ticker_path(ticker)
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime)
        return None

    def check_freshness(self, ticker: str, max_age_hours: float = 48.0) -> bool:
        """
        Check if data is fresh enough.

        Args:
            ticker: Ticker symbol
            max_age_hours: Maximum acceptable age in hours

        Returns:
            True if data is fresh enough, False if stale or missing.
        """
        last_updated = self.get_last_updated(ticker)
        if last_updated is None:
            return False
        age = datetime.now() - last_updated
        if age > timedelta(hours=max_age_hours):
            logger.warning(
                f"{ticker} data is {age.total_seconds() / 3600:.1f}h old "
                f"(limit: {max_age_hours}h). Consider running update_market_data.py"
            )
            return False
        return True

    def get_ticker_info(self, ticker: str) -> Optional[dict]:
        """Get metadata for a ticker: date range, row count, source, last update."""
        metadata = self._load_metadata()
        normalized = self.normalize_ticker(ticker)
        return metadata.get(normalized)

    def get_ohlcv(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        use_adj_close: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Read OHLCV data for a single ticker from parquet.

        Args:
            ticker: Ticker symbol (yfinance format accepted: 'SPY', 'EURUSD=X', '8058.T')
            start: Start date (YYYY-MM-DD), inclusive. None = no lower bound.
            end: End date (YYYY-MM-DD), inclusive. None = no upper bound.
            use_adj_close: If True, the returned 'Close' column contains adjusted
                           close values (matching yfinance auto_adjust=True behavior).
                           If False, returns unadjusted close.

        Returns:
            DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
            Column names are capitalized to match yfinance output format.
            Returns None if ticker not found.
        """
        path = self._ticker_path(ticker)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Could not read parquet for {ticker}: {e}")
            return None

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Date filtering
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]

        if df.empty:
            return df

        # Build output DataFrame matching yfinance format (capitalized columns)
        # Column order: Open, High, Low, Close, Adj Close, Volume
        result = pd.DataFrame(index=df.index)
        result.index.name = "Date"

        result["Open"] = df["open"] if "open" in df.columns else None
        result["High"] = df["high"] if "high" in df.columns else None
        result["Low"] = df["low"] if "low" in df.columns else None

        if use_adj_close and "adj_close" in df.columns:
            result["Close"] = df["adj_close"]
        elif "close" in df.columns:
            result["Close"] = df["close"]

        # Also provide Adj Close column for consumers that need it explicitly
        if "adj_close" in df.columns:
            result["Adj Close"] = df["adj_close"]

        result["Volume"] = df["volume"] if "volume" in df.columns else None

        return result

    def get_ohlcv_raw(
        self, ticker: str, start: str = None, end: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Read raw parquet data with lowercase columns and source column.

        Returns the parquet data as-is: [open, high, low, close, volume, adj_close, source].
        Useful for the updater when merging/appending data.

        Args:
            ticker: Ticker symbol
            start: Start date (YYYY-MM-DD), inclusive
            end: End date (YYYY-MM-DD), inclusive

        Returns:
            Raw DataFrame with lowercase columns, or None if not found.
        """
        path = self._ticker_path(ticker)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Could not read parquet for {ticker}: {e}")
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]

        return df

    def get_latest_price(
        self, ticker: str, price_col: str = "adj_close"
    ) -> Optional[float]:
        """
        Get the most recent price for a ticker.

        Args:
            ticker: Ticker symbol
            price_col: Which price column to use ('adj_close', 'close', 'open')

        Returns:
            Latest price as float, or None if not found.
        """
        path = self._ticker_path(ticker)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path, columns=[price_col])
            if df.empty:
                return None
            # Get last non-NaN value
            last_valid = df[price_col].dropna()
            if last_valid.empty:
                return None
            return float(last_valid.iloc[-1])
        except Exception as e:
            logger.warning(f"Could not get latest price for {ticker}: {e}")
            return None

    def get_latest_prices(self, tickers: list, price_col: str = "adj_close") -> dict:
        """
        Get latest prices for multiple tickers.

        Args:
            tickers: List of ticker symbols
            price_col: Which price column to use

        Returns:
            Dict of {ticker: price}. Missing tickers are omitted.
        """
        prices = {}
        for ticker in tickers:
            price = self.get_latest_price(ticker, price_col)
            if price is not None:
                prices[self.normalize_ticker(ticker)] = price
        return prices

    def get_multi_ticker_prices(
        self,
        tickers: list,
        start: str = None,
        end: str = None,
        price_col: str = "adj_close",
        column_names: dict = None,
        min_coverage: float = 0.0,
    ) -> pd.DataFrame:
        """
        Read a single price column for multiple tickers into a wide DataFrame.

        This replaces the CSV output of yfinance_downloader_v5.py for consumers
        like portimization.py that expect a DataFrame with Date index and one
        column per ticker.

        Args:
            tickers: List of ticker symbols (yfinance format accepted)
            start: Start date (YYYY-MM-DD), inclusive
            end: End date (YYYY-MM-DD), inclusive
            price_col: Which price column to use ('adj_close', 'close')
            column_names: Optional dict mapping ticker -> output column name.
                          E.g., {'EURUSD=X': 'EUR=', '8058.T': 'Marubeni'}
            min_coverage: Minimum fraction of non-NaN trading days required in
                          the [start, end] range. Tickers below this threshold
                          are excluded. E.g., 0.8 = must have data for at least
                          80% of trading days. 0.0 = no filtering (default).

        Returns:
            DataFrame with DatetimeIndex and one column per ticker.
            Columns are named using column_names if provided, otherwise
            the normalized ticker name.
        """
        series_list = []
        skipped_coverage = 0
        for ticker in tickers:
            normalized = self.normalize_ticker(ticker)
            path = self._ticker_path(ticker)
            if not path.exists():
                logger.warning(f"Ticker {ticker} not found in store, skipping")
                continue

            try:
                df = pd.read_parquet(path, columns=[price_col])
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df = df.set_index("date")
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                if start is not None:
                    df = df.loc[df.index >= pd.Timestamp(start)]
                if end is not None:
                    df = df.loc[df.index <= pd.Timestamp(end)]

                col_name = normalized
                if column_names and ticker in column_names:
                    col_name = column_names[ticker]
                elif column_names and normalized in column_names:
                    col_name = column_names[normalized]

                s = df[price_col].rename(col_name)

                # Coverage filter: skip tickers with insufficient data
                if min_coverage > 0 and len(s) > 0:
                    non_nan = s.notna().sum()
                    coverage = non_nan / len(s) if len(s) > 0 else 0.0
                    if coverage < min_coverage:
                        skipped_coverage += 1
                        logger.debug(
                            f"Ticker {ticker} excluded: coverage {coverage:.1%} "
                            f"< {min_coverage:.0%} ({non_nan}/{len(s)} trading days)"
                        )
                        continue

                series_list.append(s)
            except Exception as e:
                logger.warning(f"Could not read {ticker} for multi-ticker: {e}")

        if skipped_coverage > 0:
            logger.info(
                f"Excluded {skipped_coverage} tickers with < {min_coverage:.0%} "
                f"data coverage in [{start} to {end}]"
            )

        if not series_list:
            return pd.DataFrame()

        result = pd.concat(series_list, axis=1)
        result.index.name = "Date"
        return result

    # =========================================================================
    # WRITE API (used by update_market_data.py only)
    # =========================================================================

    def write_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        source: str = "yfinance",
        overwrite: bool = False,
    ) -> bool:
        """
        Write OHLCV data for a ticker to parquet, atomically.

        Uses write-to-tmp-then-rename pattern to prevent corruption if the
        process crashes mid-write.

        If the parquet file already exists, merges new data with existing data,
        deduplicating by date (new data takes precedence for overlapping dates).

        Args:
            ticker: Ticker symbol
            df: DataFrame with OHLCV data. Accepts both yfinance-style
                (capitalized: Open, High, Low, Close, Volume, Adj Close)
                and lowercase column names.
            source: Data source identifier ('ibkr', 'yfinance')
            overwrite: If True, replace existing ticker parquet instead of merge.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self._ticker_path(ticker)
        normalized_ticker = self.normalize_ticker(ticker)
        ticker_lock = self._get_ticker_write_lock(normalized_ticker)

        # Normalize input DataFrame to lowercase columns
        normalized_df = self._normalize_dataframe(df, source)
        if normalized_df is None or normalized_df.empty:
            logger.warning(f"No valid data to write for {ticker}")
            return False

        # Merge with existing data if present
        with ticker_lock:
            # Merge with existing data if present
            if path.exists() and not overwrite:
                try:
                    existing = pd.read_parquet(path)
                    if not isinstance(existing.index, pd.DatetimeIndex):
                        if "date" in existing.columns:
                            existing = existing.set_index("date")
                        existing.index = pd.to_datetime(existing.index)

                    # Combine: new data overwrites overlapping dates
                    combined = pd.concat([existing, normalized_df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    normalized_df = combined
                except Exception as e:
                    logger.warning(
                        f"Could not merge with existing data for {ticker}: {e}"
                    )
            elif overwrite and path.exists():
                logger.info(f"Overwriting existing data for {ticker}")

            # Atomic write: write to .tmp then rename
            tmp_path = path.with_suffix(
                f".{os.getpid()}.{threading.get_ident()}.parquet.tmp"
            )
            try:
                normalized_df.to_parquet(tmp_path, engine="pyarrow", index=True)
                tmp_path.replace(path)
            except Exception as e:
                logger.error(f"Failed to write parquet for {ticker}: {e}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                error_text = str(e).lower()
                if "too many open files" in error_text or "errno 24" in error_text:
                    raise StoreWriteTemporaryError(str(e))
                return False

            # Update metadata
            self._update_metadata_for_ticker(ticker, normalized_df, source)
            return True

    def _normalize_dataframe(
        self, df: pd.DataFrame, source: str
    ) -> Optional[pd.DataFrame]:
        """
        Normalize a yfinance-style DataFrame to the standard parquet schema.

        Handles:
        - Capitalized columns (Open, High, Low, Close, Volume, Adj Close)
        - MultiIndex columns from yf.download() (flattens them)
        - Lowercase columns (already normalized)
        - Missing adj_close (copies from close)
        """
        if df is None or df.empty:
            return None

        work = df.copy()

        # Flatten MultiIndex columns if present
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = work.columns.get_level_values(0)

        # Ensure unique column names to prevent 2D assignments when selecting by label
        if work.columns.duplicated().any():
            work = work.loc[:, ~work.columns.duplicated(keep="last")]

        # Ensure DatetimeIndex
        if not isinstance(work.index, pd.DatetimeIndex):
            if "Date" in work.columns:
                work = work.set_index("Date")
            elif "date" in work.columns:
                work = work.set_index("date")
            work.index = pd.to_datetime(work.index)

        # Strip timezone info for consistency
        if work.index.tz is not None:
            work.index = work.index.tz_localize(None)

        work.index.name = "date"

        # Build normalized DataFrame
        result = pd.DataFrame(index=work.index)

        # Column name mapping (case-insensitive lookup)
        col_map = {c.lower().replace(" ", "_"): c for c in work.columns}

        for target, candidates in [
            ("open", ["open"]),
            ("high", ["high"]),
            ("low", ["low"]),
            ("close", ["close"]),
            ("volume", ["volume"]),
            ("adj_close", ["adj_close", "adj close"]),
        ]:
            for candidate in candidates:
                if candidate in col_map:
                    source_col = work[col_map[candidate]]
                    if isinstance(source_col, pd.DataFrame):
                        source_col = source_col.iloc[:, -1]
                    if not isinstance(source_col, pd.Series):
                        logger.warning(
                            f"Malformed column for {target}: expected 1D series, got {type(source_col)}"
                        )
                        continue
                    result[target] = source_col.values
                    break

        # If adj_close is missing, copy from close
        if "adj_close" not in result.columns and "close" in result.columns:
            result["adj_close"] = result["close"]

        # If close is missing but adj_close exists, copy from adj_close
        if "close" not in result.columns and "adj_close" in result.columns:
            result["close"] = result["adj_close"]

        # Add source column
        if "source" not in result.columns:
            result["source"] = source

        # Final shape validation for required 1D numeric columns
        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            if col in result.columns and result[col].ndim != 1:
                logger.warning(f"Rejecting malformed dataframe: column {col} is not 1D")
                return None

        if not result.index.is_unique:
            result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        # Drop rows where all OHLCV columns are NaN
        ohlcv_cols = [
            c for c in ["open", "high", "low", "close", "volume"] if c in result.columns
        ]
        if ohlcv_cols:
            result = result.dropna(subset=ohlcv_cols, how="all")

        return result

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_portfolio_csv(
        self,
        tickers_map: dict,
        start: str,
        end: str,
        output_path: str,
        price_col: str = "adj_close",
        include_calendar_gaps: bool = True,
        min_coverage: float = 0.0,
    ) -> pd.DataFrame:
        """
        Export multi-ticker prices to CSV in the format portimization.py expects.

        Replaces yfinance_downloader_v5.py's CSV generation. Produces a flat CSV
        with Date column + one price column per ticker.

        Args:
            tickers_map: Dict mapping yfinance ticker -> output column name.
                         E.g., {'EURUSD=X': 'EUR=', 'SPY': 'SPY'}
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            output_path: Path to write CSV
            price_col: Which price column to use
            include_calendar_gaps: If True, reindex to full calendar dates with NaN
            min_coverage: Minimum fraction of non-NaN data required per ticker.
                          Tickers below this are excluded from the export.

        Returns:
            The exported DataFrame
        """
        tickers = list(tickers_map.keys())
        data = self.get_multi_ticker_prices(
            tickers,
            start,
            end,
            price_col,
            column_names=tickers_map,
            min_coverage=min_coverage,
        )

        if data.empty:
            logger.warning("No data to export")
            return data

        if include_calendar_gaps:
            full_range = pd.date_range(start=start, end=end, freq="D")
            data = data.reindex(full_range)
            data = data.dropna(how="all")

        data = data.reset_index().rename(columns={"index": "Date"})
        data.to_csv(output_path, encoding="utf-8", index=False)
        logger.info(
            f"Exported {len(data)} rows x {len(data.columns)} cols to {output_path}"
        )
        return data
