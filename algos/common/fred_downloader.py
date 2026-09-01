"""
FRED (Federal Reserve Economic Data) downloader for macro/rate series.

Fetches economic data series from the FRED API and converts them into the
standard OHLCV parquet schema used by MarketDataStore.

Requires:
    - ``fredapi`` package (``pip install fredapi``)
    - ``FRED_API_KEY`` environment variable set to a valid FRED API key.
      Obtain one free at https://fred.stlouisfed.org/docs/api/api_key.html

Ticker convention:
    FRED tickers in our system use a ``FRED:`` prefix.
    Example: ``FRED:FEDFUNDS``, ``FRED:IRLTLT01JPM156N``.

Usage::

    from algos.common.fred_downloader import (
        is_fred_ticker, parse_fred_ticker, download_fred_series,
    )

    df = download_fred_series("FEDFUNDS", start="2020-01-01", end="2026-01-01")
"""

import logging
import os
import random
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRED_PREFIX = "FRED:"

# Publication lag in calendar days for known FRED series.
# These represent the typical delay between the observation date and the date
# the data becomes publicly available. Source: FRED release schedules.
FRED_PUBLICATION_LAGS = {
    "FEDFUNDS": 21,  # Federal Funds Rate: ~3 weeks after month-end
    "IRLTLT01JPM156N": 60,  # Japan 10Y yield (OECD): ~2 months
    "IRSTCB01JPM156N": 60,  # BoJ policy rate (OECD): ~2 months
    "DTWEXBGS": 10,  # Trade-weighted Dollar Index: ~1-2 weeks
    "DGS2": 1,  # 2Y Treasury yield: next business day
    "DGS10": 1,  # 10Y Treasury yield: next business day
    "VIXCLS": 0,  # VIX: same day (market data)
}
DEFAULT_PUBLICATION_LAG_DAYS = 30  # Conservative default for unknown series

_HAS_FREDAPI = False
try:
    from fredapi import Fred

    _HAS_FREDAPI = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------


def is_fred_ticker(ticker: str) -> bool:
    """Return True if *ticker* uses the ``FRED:`` prefix convention."""
    return ticker.strip().upper().startswith(FRED_PREFIX)


def parse_fred_ticker(ticker: str) -> str:
    """Strip the ``FRED:`` prefix and return the raw FRED series ID.

    Raises ``ValueError`` if the ticker does not have the expected prefix.
    """
    text = ticker.strip()
    if text.upper().startswith(FRED_PREFIX):
        return text[len(FRED_PREFIX) :]
    raise ValueError(f"Not a FRED ticker: {ticker}")


# ---------------------------------------------------------------------------
# Date alignment helpers
# ---------------------------------------------------------------------------


def _align_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Shift index dates to calendar month-end for monthly series."""
    if df.empty:
        return df
    df = df.copy()
    df.index = df.index + pd.offsets.MonthEnd(0)
    # Drop duplicates that may arise from alignment.
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _detect_frequency(df: pd.DataFrame) -> str:
    """Guess whether a FRED series is daily, monthly, or quarterly."""
    if df.empty or len(df) < 3:
        return "unknown"
    deltas = df.index.to_series().diff().dropna().dt.days
    median_gap = deltas.median()
    if median_gap <= 5:
        return "daily"
    if median_gap <= 35:
        return "monthly"
    if median_gap <= 100:
        return "quarterly"
    return "unknown"


def shift_for_publication_lag(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Shift FRED data forward by publication lag to prevent lookahead bias.

    FRED data is indexed by observation date, but the data is not publicly
    available until days/weeks/months later. This function shifts the data
    forward so that it is only available at the approximate publication date.

    Args:
        df: DataFrame with DatetimeIndex (observation dates).
        series_id: FRED series ID (e.g., 'FEDFUNDS').

    Returns:
        DataFrame with index shifted forward by the publication lag.
    """
    lag_days = FRED_PUBLICATION_LAGS.get(series_id, DEFAULT_PUBLICATION_LAG_DAYS)
    if lag_days > 0:
        df = df.copy()
        df.index = df.index + pd.Timedelta(days=lag_days)
        logger.info(
            f"Shifted FRED series {series_id} by {lag_days} days for publication lag"
        )
    return df


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------


def download_fred_series(
    series_id: str,
    start: str = None,
    end: str = None,
    align_month_end: bool = True,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch a FRED series and return a store-compatible DataFrame.

    Parameters
    ----------
    series_id : str
        Raw FRED series ID (without ``FRED:`` prefix),
        e.g. ``"FEDFUNDS"``, ``"IRLTLT01JPM156N"``.
    start, end : str, optional
        Date range in ``YYYY-MM-DD`` format.
    align_month_end : bool
        If ``True`` and the series appears monthly, align dates to month-end.
    max_retries : int
        Number of retry attempts on transient failures.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with DatetimeIndex and columns:
        ``[open, high, low, close, volume, adj_close, source]``.
        Returns ``None`` on failure.
    """
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return None

    if not _HAS_FREDAPI:
        logger.error("fredapi package is not installed. Run: pip install fredapi")
        return None

    fred = Fred(api_key=api_key)

    raw_series: Optional[pd.Series] = None
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            kwargs = {}
            if start:
                kwargs["observation_start"] = start
            if end:
                kwargs["observation_end"] = end

            raw_series = fred.get_series(series_id, **kwargs)

            if raw_series is not None and not raw_series.empty:
                break
            logger.debug(f"FRED returned empty for {series_id} (attempt {attempt + 1})")
        except Exception as e:
            last_error = e
            logger.warning(
                f"FRED fetch failed for {series_id} "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                wait = random.uniform(2.0, 5.0) * (attempt + 1)
                time.sleep(wait)

    if raw_series is None or raw_series.empty:
        logger.error(
            f"Failed to fetch FRED series {series_id} after {max_retries} attempts"
            + (f": {last_error}" if last_error else "")
        )
        return None

    # Build store-compatible DataFrame.
    # FRED data is a single value per observation date.
    # Map to OHLCV: open=high=low=close=adj_close=value, volume=0.
    raw_series = raw_series.dropna()
    if raw_series.empty:
        logger.warning(f"FRED series {series_id} has no non-NaN observations")
        return None

    df = pd.DataFrame(
        {
            "open": raw_series.values,
            "high": raw_series.values,
            "low": raw_series.values,
            "close": raw_series.values,
            "volume": 0,
            "adj_close": raw_series.values,
            "source": "fred",
        },
        index=raw_series.index,
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Align monthly/quarterly series to month-end for cleaner joins.
    if align_month_end:
        freq = _detect_frequency(df)
        if freq in ("monthly", "quarterly"):
            df = _align_to_month_end(df)
            logger.debug(f"FRED {series_id}: aligned {freq} dates to month-end")

    df = df.sort_index()

    df = shift_for_publication_lag(df, series_id)

    logger.info(
        f"FRED {series_id}: fetched {len(df)} observations "
        f"({df.index.min().date()} to {df.index.max().date()})"
    )
    return df


# ---------------------------------------------------------------------------
# Convenience: check if FRED is available at runtime
# ---------------------------------------------------------------------------


def is_fred_available() -> bool:
    """Return True if fredapi is installed and FRED_API_KEY is set."""
    return _HAS_FREDAPI and bool(os.environ.get("FRED_API_KEY", "").strip())
