"""
Yahoo Data Download script for multiple tickers with custom formatting.

Goals:
    1. Download ticker data for a defined list of assets.
    2. For each ticker, prioritize 'Adj Close' price; fall back to 'Close' if 'Adj Close' is not available.
    3. Parse the data to a pandas DataFrame, selecting the chosen price type.
    4. Rename columns to match desired custom output format (e.g., AAPL, EUR=).
    5. Ensure the output DataFrame contains a continuous date range (daily),
       with blank (NaN) cells for tickers on their non-trading days.
    6. Save the DataFrame to a CSV file.
    7. Implement robust error handling and logging for production environment.

@author: jcp
@date: 2025-07-17
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import sys
import time
import random
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser

# Import resilient downloader
try:
    from algos.common.yf_downloader import resilient_download

    _HAS_RESILIENT_DOWNLOADER = True
except ImportError:
    try:
        _script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _project_root = _script_dir.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from algos.common.yf_downloader import resilient_download

        _HAS_RESILIENT_DOWNLOADER = True
    except ImportError:
        _HAS_RESILIENT_DOWNLOADER = False
        logging.warning("yf_downloader not available, using built-in retry logic")

# Configure logging for production environment
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# RATE LIMITING CONFIGURATION (Based on Yahoo Finance research - Jan 2026)
# =============================================================================
# Yahoo Finance has NO official documented rate limits for yfinance endpoints.
# These values are based on community experience and November 2024 tightening:
# - ~950 tickers per session before 429 errors (down from ~7000 previously)
# - Recommended: 2 requests per 5 seconds (0.4 req/sec)
# - Random delays help avoid detection patterns
#
# IMPORTANT: As of yfinance 0.2.57+, custom sessions are NOT supported because
# yfinance uses curl_cffi internally for browser impersonation. We use manual
# delays between batches instead of session-based rate limiting.
# =============================================================================

RATE_LIMIT_CONFIG = {
    "batch_size": 30,  # Tickers per batch for large downloads
    "delay_between_batches": 5.0,  # Base delay between batches (seconds)
    "delay_jitter": 3.0,  # Random jitter added to delay (0 to this value)
    "delay_on_error": 10.0,  # Initial delay on 429 error (seconds)
    "max_retries": 3,  # Max retries per batch
    "backoff_multiplier": 2.0,  # Exponential backoff multiplier
}


def download_with_rate_limit(tickers, start_date, end_date, interval="1d"):
    """
    Download data with rate limiting, batching, and retry logic.

    Delegates to the centralized resilient downloader (algos.common.yf_downloader)
    which provides:
    - Batched downloads with delays between batches
    - Exponential backoff with jitter on errors
    - Failed-ticker detection via stderr parsing (retries only tickers that failed)
    - SQLite cache corruption recovery
    - Individual ticker fallback for persistently failed tickers

    Falls back to the legacy built-in retry logic if the resilient downloader
    is not available.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        interval: Data interval (default '1d')

    Returns:
        pd.DataFrame: Combined data for all tickers
    """
    if _HAS_RESILIENT_DOWNLOADER:
        logging.info(f"Using resilient downloader for {len(tickers)} tickers...")
        result = resilient_download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        return result if result is not None else pd.DataFrame()

    # Legacy fallback: built-in retry logic (less robust)
    logging.warning("Resilient downloader not available, using legacy retry logic")
    return _download_with_rate_limit_legacy(tickers, start_date, end_date, interval)


def _download_with_rate_limit_legacy(tickers, start_date, end_date, interval="1d"):
    """
    Legacy download with rate limiting (fallback when resilient downloader unavailable).

    For large ticker lists (>batch_size), downloads in batches with delays.
    Implements exponential backoff on 429 errors.
    NOTE: Does NOT detect per-ticker failures within a batch.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        interval: Data interval (default '1d')

    Returns:
        pd.DataFrame: Combined data for all tickers
    """
    batch_size = RATE_LIMIT_CONFIG["batch_size"]

    # For small ticker lists, download all at once
    if len(tickers) <= batch_size:
        logging.info(f"Downloading {len(tickers)} tickers in single batch...")
        return _download_batch_with_retry(tickers, start_date, end_date, interval)

    # For large lists, download in batches
    logging.info(f"Downloading {len(tickers)} tickers in batches of {batch_size}...")
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        logging.info(
            f"Batch {batch_num}/{total_batches}: Downloading {len(batch)} tickers..."
        )

        batch_data = _download_batch_with_retry(batch, start_date, end_date, interval)

        if batch_data is not None and not batch_data.empty:
            all_data.append(batch_data)

        # Add delay between batches (except for last batch)
        if i + batch_size < len(tickers):
            delay = RATE_LIMIT_CONFIG["delay_between_batches"] + random.uniform(
                0, RATE_LIMIT_CONFIG["delay_jitter"]
            )
            logging.info(f"  Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)

    # Combine all batches
    if not all_data:
        return pd.DataFrame()

    # Merge dataframes on index
    combined = all_data[0]
    for df in all_data[1:]:
        combined = combined.join(df, how="outer")

    return combined


def _download_batch_with_retry(tickers, start_date, end_date, interval):
    """
    Legacy: Download a batch of tickers with exponential backoff retry.

    Note: As of yfinance 0.2.57+, custom sessions are not supported due to
    curl_cffi usage. We use manual delays instead of session-based rate limiting.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        interval: Data interval

    Returns:
        pd.DataFrame: Downloaded data or None on failure
    """
    max_retries = RATE_LIMIT_CONFIG["max_retries"]
    base_delay = RATE_LIMIT_CONFIG["delay_on_error"]
    backoff = RATE_LIMIT_CONFIG["backoff_multiplier"]

    for attempt in range(max_retries):
        try:
            # Note: yfinance 0.2.57+ uses curl_cffi internally and doesn't support
            # custom sessions. We rely on manual delays between batches instead.
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                threads=True,
                auto_adjust=False,
                progress=False,  # Disable progress bar for cleaner logs
                ignore_tz=True,
            )

            # Check for rate limit error in the returned data (yfinance may not raise exception)
            if data is not None and not data.empty:
                return data

            # Empty data might indicate rate limiting, wait and retry
            if attempt < max_retries - 1:
                wait_time = base_delay * (backoff**attempt)
                logging.warning(
                    f"  Empty response (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.0f}s..."
                )
                time.sleep(wait_time)

        except Exception as e:
            error_str = str(e).lower()

            if "429" in error_str or "rate" in error_str or "too many" in error_str:
                wait_time = base_delay * (backoff**attempt)
                logging.warning(
                    f"  Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.0f}s..."
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    f"  Download error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(base_delay)

    logging.error(f"  Failed to download after {max_retries} attempts")
    return None


def download_financial_data(
    tickers_map: dict,
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = None,
    interval: str = "1d",
    output_filename: str = "yfinance_combined_data.csv",
    desired_output_order: list = None,
    timezone: str = None,
) -> pd.DataFrame:
    """
    Downloads historical price data for multiple financial tickers using yfinance.
    Prioritizes 'Adj Close' and falls back to 'Close'. Formats the DataFrame
    to include all calendar days with NaN for non-trading days, and saves to CSV.

    Args:
        tickers_map (dict): A dictionary where keys are yfinance ticker symbols
                            and values are the desired column names in the output DataFrame.
                            Example: {'AAPL': 'AAPL.O', 'EURUSD=X': 'EUR='}.
        start_date (str, optional): The start date for data download (YYYY-MM-DD).
                                   Can be omitted if lookback_days is provided.
        end_date (str, optional): The end date for data download (YYYY-MM-DD).
                                 Can be omitted if lookback_days is provided (defaults to today).
        lookback_days (int, optional): Number of days to look back from end_date (or today).
                                        If provided, start_date will be calculated automatically.
                                        Example: lookback_days=1260 for ~5 years of daily data.
        interval (str): The data interval (e.g., '1d' for daily). Defaults to '1d'.
        output_filename (str): The base name of the CSV file to save the data to.
                               The final filename will include dates and interval.
        desired_output_order (list, optional): A list of column names in the desired order
                                                for the final output. If a column is in this
                                                list but not in the downloaded data, it's skipped.
                                                If None, columns are ordered by ticker map values alphabetically.
        timezone (str, optional): IANA timezone name to convert the output dates to
                                  (e.g., 'America/New_York', 'Asia/Tokyo', 'Europe/London', 'UTC').
                                  If None, dates are left as-is from yfinance.
                                  Note: yfinance does not support fetching data in a specific
                                  timezone natively. Daily bars are always returned timezone-naive,
                                  intraday bars are always UTC. This parameter applies a post-
                                  download conversion: naive dates are assumed UTC before converting.
                                  For daily bars the date portion is preserved (time is dropped).

    Returns:
        pd.DataFrame: The processed DataFrame with downloaded data, or an empty DataFrame
                      if no data is available or an error occurs.
    """
    # Validate timezone early if provided
    if timezone is not None:
        try:
            import zoneinfo

            zoneinfo.ZoneInfo(timezone)
        except (KeyError, Exception):
            logging.error(
                f"Invalid timezone '{timezone}'. Use IANA timezone names "
                f"(e.g., 'America/New_York', 'Asia/Tokyo', 'UTC'). "
                f"Run 'python -c \"import zoneinfo; print(sorted(zoneinfo.available_timezones()))\"' "
                f"to list all available timezones."
            )
            return pd.DataFrame()
    # Calculate dates from lookback_days if provided
    if lookback_days is not None:
        # Get current date (or use provided end_date)
        if end_date is None:
            end_dt = datetime.now()
            end_date = end_dt.strftime("%Y-%m-%d")
            logging.info(f"Using current date as end_date: {end_date}")
        else:
            # Parse and regularize the end_date using dateutil
            try:
                end_dt = dateutil_parser.parse(end_date)
                end_date = end_dt.strftime("%Y-%m-%d")
                logging.info(f"Parsed and regularized end_date: {end_date}")
            except Exception as e:
                logging.error(f"Failed to parse end_date '{end_date}': {e}")
                return pd.DataFrame()

        # Calculate start_date
        start_dt = end_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        logging.info(
            f"Calculated start_date from lookback_days={lookback_days}: {start_date}"
        )

    # Validate that we have both start_date and end_date
    if start_date is None or end_date is None:
        logging.error(
            "Either provide both start_date and end_date, or provide lookback_days parameter."
        )
        return pd.DataFrame()

    # Regularize dates using dateutil if they were provided directly
    if lookback_days is None:
        try:
            start_dt = dateutil_parser.parse(start_date)
            start_date = start_dt.strftime("%Y-%m-%d")
            end_dt = dateutil_parser.parse(end_date)
            end_date = end_dt.strftime("%Y-%m-%d")
            logging.info(f"Regularized dates - start: {start_date}, end: {end_date}")
        except Exception as e:
            logging.error(f"Failed to parse dates: {e}")
            return pd.DataFrame()

    yfinance_tickers = list(tickers_map.keys())

    if not yfinance_tickers:
        logging.warning("No tickers defined in tickers_map. Skipping download.")
        return pd.DataFrame()

    logging.info(
        f"Attempting to download data for {len(yfinance_tickers)} tickers from {start_date} to {end_date} with interval {interval}..."
    )

    try:
        # Download data using rate-limited batched download
        # This handles:
        # - Batching for large ticker lists (50 per batch)
        # - Delays between batches (3-5 seconds with jitter)
        # - Exponential backoff on 429 errors (10s, 20s, 40s)
        # - Automatic retries (3 attempts per batch)
        #
        # Note: yfinance 0.2.57+ uses curl_cffi internally for browser
        # impersonation, so custom sessions/User-Agents are not needed.
        raw_multi_index = download_with_rate_limit(
            yfinance_tickers, start_date, end_date, interval
        )

        if raw_multi_index is None or raw_multi_index.empty:
            logging.warning(
                f"No data downloaded for the specified tickers and date range. Returned an empty DataFrame."
            )
            return pd.DataFrame()

        # Create an empty list to hold the selected series for each ticker
        selected_prices = []

        # Iterate through each ticker to select 'Adj Close' or 'Close'
        for yf_ticker in yfinance_tickers:
            # Check if this specific ticker's data exists in the downloaded MultiIndex
            if yf_ticker not in raw_multi_index.columns.levels[1]:
                logging.warning(
                    f"No data found for ticker '{yf_ticker}'. It will be excluded from the output."
                )
                continue

            # Prioritize 'Adj Close'
            if ("Adj Close", yf_ticker) in raw_multi_index.columns:
                price_series = raw_multi_index["Adj Close"][yf_ticker]
                logging.debug(f"Using 'Adj Close' for ticker: {yf_ticker}")
            # Fallback to 'Close'
            elif ("Close", yf_ticker) in raw_multi_index.columns:
                price_series = raw_multi_index["Close"][yf_ticker]
                logging.debug(f"Using 'Close' for ticker: {yf_ticker}")
            else:
                logging.warning(
                    f"Neither 'Adj Close' nor 'Close' found for ticker '{yf_ticker}'. It will be excluded from the output."
                )
                continue

            # Rename the series to the desired output column name
            price_series.name = tickers_map.get(
                yf_ticker, yf_ticker
            )  # Use map, fallback to original if not found (shouldn't happen with correct map)
            selected_prices.append(price_series)

        if not selected_prices:
            logging.warning(
                "No price series could be successfully extracted for any ticker. Returning empty DataFrame."
            )
            return pd.DataFrame()

        # Concatenate all selected price series into a single DataFrame
        data = pd.concat(selected_prices, axis=1)

        # Ensure index is datetime type
        data.index = pd.to_datetime(data.index)

        # --- Crucial step for the desired output format (blanks for non-trading days) ---
        # Create a full calendar date range from the start_date to end_date
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Reindex the data to the full calendar range. This will insert NaNs for missing dates.
        # This keeps the 'blanks' for non-trading days where a ticker didn't trade.
        data = data.reindex(full_date_range)

        # Drop any rows where ALL columns are NaN. This cleans up dates that are
        # entirely empty (e.g., if start_date is before any ticker's data begins,
        # or a universal market closure not covered by yfinance for any asset).
        initial_rows_after_reindex = data.shape[0]
        data.dropna(how="all", inplace=True)
        if data.shape[0] < initial_rows_after_reindex:
            logging.info(
                f"Dropped {initial_rows_after_reindex - data.shape[0]} rows where all ticker data was NaN after reindexing."
            )

        if data.empty:
            logging.warning(
                "DataFrame is empty after reindexing and dropping all-NaN rows. No valid data to save."
            )
            return pd.DataFrame()

        # --- Timezone conversion ---
        if timezone is not None:
            import zoneinfo

            target_tz = zoneinfo.ZoneInfo(timezone)

            if data.index.tz is None:
                # Naive datetime index (typical for daily bars) -- assume UTC first
                data.index = data.index.tz_localize("UTC").tz_convert(target_tz)
                logging.info(f"Localized naive dates as UTC, converted to {timezone}")
            else:
                # Already timezone-aware -- convert directly
                data.index = data.index.tz_convert(target_tz)
                logging.info(f"Converted timezone-aware dates to {timezone}")

            # For daily bars, strip the time component after conversion so the
            # Date column stays clean (e.g., '2025-01-15' not '2025-01-15 00:00:00-05:00').
            # For intraday intervals, keep the full datetime with tz offset.
            is_daily_or_larger = interval in ("1d", "1wk", "1mo", "5d", "3mo")
            if is_daily_or_larger:
                data.index = data.index.normalize().tz_localize(None)
                logging.info(
                    f"Stripped time component for daily bars (timezone: {timezone})"
                )

        # Reset index to make 'Date' a regular column, as per desired output format
        data = data.reset_index().rename(columns={"index": "Date"})

        # Order columns: 'Date' first, then desired order, then any remaining columns
        final_columns = ["Date"]
        if desired_output_order:
            # Add columns from desired_output_order that are actually present in data
            final_columns.extend(
                [
                    col
                    for col in desired_output_order
                    if col in data.columns and col != "Date"
                ]
            )

        # Add any columns that are in `data` but were not explicitly in `desired_output_order`
        existing_cols_not_ordered = [
            col for col in data.columns if col not in final_columns and col != "Date"
        ]
        if existing_cols_not_ordered:
            logging.info(
                f"Appending columns not specified in desired_output_order: {existing_cols_not_ordered}"
            )
            # Sort them alphabetically for consistent ordering of appended columns
            existing_cols_not_ordered.sort()
            final_columns.extend(existing_cols_not_ordered)

        data = data[final_columns]

        # Construct the full output filename
        full_output_filename = (
            f"{output_filename.split('.csv')[0]}_{start_date}_{end_date}_{interval}.csv"
        )

        # Save to csv file
        data.to_csv(full_output_filename, encoding="utf-8", index=False)
        logging.info(
            f"Successfully downloaded and saved data to '{full_output_filename}'"
        )
        logging.info(f"Final DataFrame shape: {data.shape}")

        return data

    except Exception as e:
        logging.error(
            f"An error occurred during data download or processing: {e}", exc_info=True
        )
        return pd.DataFrame()  # Return empty DataFrame on error


if __name__ == "__main__":
    # --- Configuration for Production Use ---
    # Define the tickers you want to download and their desired output names.
    # The keys must be valid yfinance ticker symbols.
    # The values are the exact column names that will appear in your CSV.
    # Example: 'AAPL': 'AAPL' will result in an 'AAPL' column.
    # If yfinance's ticker is 'EURUSD=X' but you want 'EUR=', map it as 'EURUSD=X': 'EUR='


    current_tickers_map = {  # Example multi-asset, multi-exchange universe
        "SPY": "SPY",
        "QQQ": "QQQ",
        "TLT": "TLT",
        "GLD": "GLD",
        "BIL": "BIL",
        "SAP.DE": "SAP.DE",   # Xetra
        "BP.L": "BP.L",       # LSE
        "7203.T": "7203.T",   # Tokyo
        "BTC-USD": "BTC-USD", # crypto
    }


    # Define the date range and intervals
    # start_date_param = '2023-01-16'
    # end_date_param = '2026-01-23'
    lookback_days_param = (
        1100  # 3 years, to catch up regieme, +5 days when portimization.py
    )
    # lookback_days_param = 1825  # 5 years, +5 days when portimization.py
    # lookback_days_param = 2555 # 7 years
    # lookback_days_param = 7300 # 20 years
    # lookback_days_param = 100 # 20 years
    # interval_param = '1d'
    interval_param = "1d"

    # Optional: Define the exact order of columns for the output CSV.
    # This list should use the *output column names* defined as values in current_tickers_map.
    # Any column names in this list that are not actually downloaded will be skipped.
    # Any downloaded/renamed columns not in this list will be appended at the end.
    desired_output_column_order = [
        "SPY",
        "EUR=",
        "GDX",
        "GLD",
        "NVDA",
        "TSLA",
        "AAPL",
        "MSFT",
        "AMZN",
        "QQQ",
    ]

    # --- CLI Configuration ---
    # Parse command-line flags.
    # Usage:
    #   python yfinance_downloader_v5.py                                              # use hardcoded tickers (default)
    #   python yfinance_downloader_v5.py --json data/hrp_weights.json                 # load tickers from JSON keys
    #   python yfinance_downloader_v5.py --json data/hrp_weights.json --tz UTC        # JSON tickers + timezone
    #   python yfinance_downloader_v5.py --timezone America/New_York                  # hardcoded tickers + timezone
    import argparse

    parser = argparse.ArgumentParser(
        description="Download financial data from yfinance with optional timezone conversion.",
        add_help=False,  # Don't conflict with any existing arg handling
    )
    parser.add_argument(
        "--timezone",
        "--tz",
        type=str,
        default=None,
        metavar="TZ",
        help=(
            "IANA timezone name to convert output dates to "
            "(e.g., 'America/New_York', 'Asia/Tokyo', 'Europe/London', 'UTC'). "
            "If omitted, dates are left as-is from yfinance."
        ),
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file whose keys define the ticker universe to download. "
            "The JSON values are ignored (e.g., weights from HRP output). "
            "Each key is used as both the yfinance ticker symbol and the output "
            "column name. When provided, this overrides the hardcoded tickers_map. "
            "Example: --json algos/backtest_code/data/hrp_weights.json"
        ),
    )
    args, _ = parser.parse_known_args()
    timezone_param = args.timezone
    json_path_param = args.json

    if timezone_param:
        logging.info(f"Timezone override: {timezone_param}")

    # --- JSON ticker universe override ---
    if json_path_param is not None:
        json_file = Path(json_path_param)
        if not json_file.is_file():
            logging.error(f"JSON file not found: {json_file.resolve()}")
            sys.exit(1)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {json_file}: {e}")
            sys.exit(1)

        if not isinstance(json_data, dict) or not json_data:
            logging.error(
                f"JSON file must contain a non-empty object/dict. Got: {type(json_data).__name__}"
            )
            sys.exit(1)

        # Build tickers_map from JSON keys: key is both yfinance ticker and output name
        current_tickers_map = {str(k): str(k) for k in json_data.keys()}
        logging.info(
            f"Loaded {len(current_tickers_map)} tickers from JSON: {json_file.name}"
        )

    # period='5y'  # Not used in this version, but could be integrated for dynamic date ranges
    # --- Execute Download ---
    downloaded_df = download_financial_data(
        tickers_map=current_tickers_map,
        # start_date=start_date_param,
        # end_date=end_date_param,
        # period=period,
        lookback_days=lookback_days_param,
        interval=interval_param,
        output_filename="financial_data_combined_prices.csv",  # Changed filename for clarity
        desired_output_order=desired_output_column_order,
        timezone=timezone_param,
    )

    if not downloaded_df.empty:
        logging.info(
            "Data download and processing completed successfully. Check the generated CSV file."
        )
    else:
        logging.error(
            "Failed to download or process data, or no data was available after filtering. Review logs for specific issues."
        )
