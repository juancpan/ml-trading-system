import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import sys
from typing import Optional

# Import resilient downloader (preferred) with yfinance fallback
try:
    from algos.common.yf_downloader import resilient_download_single

    _HAS_RESILIENT_DOWNLOADER = True
except ImportError:
    try:
        _script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _project_root = _script_dir.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from algos.common.yf_downloader import resilient_download_single

        _HAS_RESILIENT_DOWNLOADER = True
    except ImportError:
        import yfinance as yf

        _HAS_RESILIENT_DOWNLOADER = False
        print(
            "Warning: yf_downloader not available, falling back to bare yfinance (no retry logic)"
        )


# --- Define and create necessary directories relative to project root
try:
    current_execution_dir = Path(os.getcwd())
    project_root_dir = current_execution_dir
    # Traverse up until 'project root' is found or a sensible root is reached.
    for _ in range(5):
        if (project_root_dir / "algos").is_dir():
            break
        if project_root_dir == project_root_dir.parent:
            break
        project_root_dir = project_root_dir.parent
    if not (project_root_dir / "algos").is_dir():
        # Fallback if 'project root' not found, assume current_execution_dir is a good base
        project_root_dir = current_execution_dir
except NameError:
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root_dir = current_script_dir.parent.parent

# Define paths for data, images, logs, and pickle dumps using Pathlib
data_dir = project_root_dir / "data"
logs_dir = project_root_dir / "logs"

# Create directories if they don't exist
for d in [data_dir, logs_dir]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)


class RedirectStdoutToFile:
    def __init__(self, filename="output.txt", mode="a"):
        self.filename = logs_dir / filename
        self.mode = mode
        self.original_stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()


# Import forex detection from IBKR downloader (canonical source of truth)
try:
    from algos.common.ibkr_downloader import is_forex_ticker as _is_forex_ticker
except ImportError:
    try:
        _ibkr_dl_path = Path(os.path.dirname(os.path.abspath(__file__)))
        if str(_ibkr_dl_path) not in sys.path:
            sys.path.insert(0, str(_ibkr_dl_path))
        from ibkr_downloader import is_forex_ticker as _is_forex_ticker
    except ImportError:
        # Minimal fallback: detect 6-char all-alpha tickers ending in common currencies
        def _is_forex_ticker(ticker: str) -> bool:
            """Fallback forex detection when ibkr_downloader is unavailable."""
            clean = ticker.replace("=X", "").replace("/", "").replace(".", "")
            if len(clean) != 6 or not clean.isalpha() or not clean.isupper():
                return False
            _fx_ccys = {
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
            }
            return clean[:3] in _fx_ccys and clean[3:] in _fx_ccys


def _get_jpy_policy_rate(date: pd.Timestamp) -> float:
    """Return approximate BoJ policy rate for a given date.

    Uses a step function based on known BoJ policy rate decisions.
    Accurate to within ~10bps for backtesting carry-adjusted returns.

    Args:
        date: The date to look up the BoJ rate for.

    Returns:
        Annualized BoJ policy rate as a decimal (e.g., 0.005 = 0.5%).
    """
    # BoJ policy rate history (approximate effective dates)
    # Source: Bank of Japan monetary policy announcements
    if date >= pd.Timestamp("2025-01-24"):
        return 0.005  # 0.50% (Jan 2025 hike)
    elif date >= pd.Timestamp("2024-07-31"):
        return 0.0025  # 0.25% (Jul 2024 hike)
    elif date >= pd.Timestamp("2024-03-19"):
        return 0.001  # 0.10% (Mar 2024: end of negative rates)
    elif date >= pd.Timestamp("2016-02-01"):
        return -0.001  # -0.10% (negative rate era)
    else:
        return 0.001  # ~0.10% (pre-2016)


def compute_carry_differential(
    index: pd.DatetimeIndex,
    irx_series: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute daily carry differential (USD rate - JPY rate) for each date.

    This represents the annualized cost of holding USD debt instead of JPY debt.
    A positive value means USD debt is more expensive than JPY debt.

    Uses ^IRX (13-week T-bill yield) as the USD short rate proxy.
    Falls back to a static step function if ^IRX data is unavailable.

    Args:
        index: DatetimeIndex for the output series.
        irx_series: Optional pandas Series of ^IRX close values (in percentage points,
                     e.g., 4.5 = 4.5% yield). Index must be DatetimeIndex.

    Returns:
        Series of daily carry differentials (annualized, as decimal).
        Example: 0.03 means 3% annualized carry spread.
    """
    # USD short rate: use ^IRX if available, else step function
    if irx_series is not None and not irx_series.empty:
        # ^IRX is quoted in percentage points (e.g., 4.5 = 4.5%)
        usd_rate = irx_series.reindex(index, method="ffill") / 100.0
    else:
        # Fallback: approximate Fed Funds rate history
        usd_rate = pd.Series(index=index, dtype=float)
        for i, dt in enumerate(index):
            if dt >= pd.Timestamp("2024-09-18"):
                usd_rate.iloc[i] = 0.0450  # Post Sep 2024 cuts
            elif dt >= pd.Timestamp("2023-07-26"):
                usd_rate.iloc[i] = 0.0525  # Peak rate
            elif dt >= pd.Timestamp("2022-03-17"):
                usd_rate.iloc[i] = 0.0350  # Hiking cycle avg
            elif dt >= pd.Timestamp("2020-03-15"):
                usd_rate.iloc[i] = 0.0025  # COVID zero rates
            else:
                usd_rate.iloc[i] = 0.0150  # Pre-COVID

    # JPY rate: step function
    jpy_rate = pd.Series(
        [_get_jpy_policy_rate(dt) for dt in index],
        index=index,
        dtype=float,
    )

    # Daily carry differential (annualized)
    carry_diff = usd_rate - jpy_rate
    carry_diff.name = "carry_differential"
    return carry_diff


def compute_direction(
    returns: pd.Series,
    ticker: str,
    carry_series: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute direction target variable, with carry adjustment for forex tickers.

    For forex tickers: direction = sign(returns + daily_carry / 365)
    This accounts for the interest rate differential in the carry trade decision.
    A small FX loss that is offset by carry income should still be direction +1.

    For stocks/crypto: direction = sign(returns) -- unchanged behavior.

    Args:
        returns: Series of log returns.
        ticker: Ticker symbol (used to detect forex).
        carry_series: Optional Series of annualized carry differentials.
                      Only used for forex tickers.

    Returns:
        Series of direction values: 1 (up/favorable) or -1 (down/unfavorable).
    """
    if carry_series is not None and _is_forex_ticker(ticker):
        # Convert annualized carry to daily
        daily_carry = carry_series.reindex(returns.index, method="ffill") / 365.0
        adjusted_returns = returns + daily_carry
        return pd.Series(
            np.where(adjusted_returns > 0, 1, -1),
            index=returns.index,
            name="direction",
        )
    return pd.Series(
        np.where(returns > 0, 1, -1),
        index=returns.index,
        name="direction",
    )


# Helper function to determine if a ticker is likely a cryptocurrency
def _is_crypto_ticker(ticker: str) -> bool:
    crypto_suffixes = ["-USD", "-USDT", "-BTC", "-ETH", "-EUR"]
    known_crypto_symbols = {
        "BTC",
        "ETH",
        "XRP",
        "LTC",
        "ADA",
        "DOGE",
        "SOL",
        "BNB",
        "DOT",
        "TRX",
        "LINK",
        "UNI",
        "AVAX",
        "MATIC",
    }

    if any(ticker.upper().endswith(suffix) for suffix in crypto_suffixes):
        return True
    if ticker.upper() in known_crypto_symbols:
        return True
    return False


# Function to calculate annual trading periods based on interval and asset type
def _calculate_annual_trading_periods(
    interval: str, is_crypto: bool, is_forex: bool = False
) -> int:
    unit_match = re.match(r"(\d+)([a-zA-Z]+)", interval)
    if not unit_match:
        raise ValueError(
            f"Invalid interval format: {interval}. Expected format like '1m', '1h', '1d', etc."
        )

    value = int(unit_match.group(1))
    unit = unit_match.group(2).lower()

    if is_crypto:
        if unit == "d":
            return 365 // value
        if unit == "wk":
            return 52 // value
        if unit == "mo":
            return 12 // value
        if unit == "h":
            return (365 * 24) // value
        if unit == "m":
            return (365 * 24 * 60) // value
    elif is_forex:
        # Forex trades ~260 weekdays/year (52 weeks x 5 days)
        if unit == "d":
            return 260 // value
        if unit == "wk":
            return 52 // value
        if unit == "mo":
            return 12 // value
        if unit == "h":
            return (260 * 24) // value  # Forex is 24h Sun-Fri
        if unit == "m":
            return (260 * 24 * 60) // value
    else:  # Stocks/ETFs (US market specific)
        if unit == "d":
            return 252 // value
        if unit == "wk":
            return 52 // value
        if unit == "mo":
            return 12 // value
        if unit == "h":
            return int(252 * 6.5) // value  # Approximately 1638 trading hours per year
        if unit == "m":
            return (
                int(252 * 6.5 * 60) // value
            )  # Approximately 98280 trading minutes per year

    raise ValueError(
        f"Unsupported interval '{interval}' for asset type "
        f"(crypto: {is_crypto}, forex: {is_forex})."
    )


def load_and_preprocess_data(
    ticker: str = None,
    start: str = None,
    end: str = None,
    interval: str = "1d",
    symbol: str = "Adj Close",
    user_provided_file_path: str = None,
    log_filename: str = "data_loading_output.txt",
) -> pd.DataFrame:
    """
    Loads historical data from a user-provided CSV file, Yahoo Finance cache, or downloads from Yahoo Finance.
    Performs basic preprocessing and sets 'annual_trading_periods' attribute.

    Args:
        ticker (str, optional): The stock/ETF/crypto ticker symbol (e.g., 'QQQ', 'BTC-USD').
                                Required if user_provided_file_path is None.
        start (str, optional): Start date in 'YYYY-MM-DD' format.
                                Required if user_provided_file_path is None.
        end (str, optional): End date in 'YYYY-MM-DD' format.
                              Required if user_provided_file_path is None.
        interval (str, optional): Data interval (e.g., '1m', '1h', '1d', '1wk', '1mo'). Defaults to '1d'.
                                  Used for yfinance and to infer annual_trading_periods.
        symbol (str): The preferred column name for price data (e.g., 'Adj Close', 'Close', 'Open').
                      Will fall back to 'Adj Close', then 'Close', then 'Open' if preferred is not found.
        user_provided_file_path (str, optional): Path to a pre-processed CSV data file. If provided,
                                                 ticker, start, end, interval are ignored for data source.
        log_filename (str): Name of the log file to redirect stdout.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with OHLCV columns (open, high, low, close, volume)
                      plus 'price', 'returns', 'direction'.
                      Includes 'annual_trading_periods' in its .attrs.
                      Returns None if data loading/preprocessing fails.
    """

    with RedirectStdoutToFile(log_filename):
        print(f"Starting data loading and preprocessing. Log file: {log_filename}")

        raw_data_source = None  # Will store the DataFrame from file or yfinance
        chosen_ticker_for_metadata = (
            None  # The ticker that will be stored in data.attrs
        )
        chosen_interval_for_metadata = (
            None  # The interval that will be stored in data.attrs
        )

        # --- PRIORITY 1: Load from user-provided file_path ---
        if user_provided_file_path:
            print(
                f"Attempting to load data from user-provided file: {user_provided_file_path}"
            )
            try:
                # This assumes the format from portfolio_data_gen.py (2 custom header rows + DataFrame header)
                # Header looks like:
                # Price,Adj Close,Close,High,Low,Open
                # Ticker,portfolio_TIMESTAMP,...
                # Date,Adj Close,Close,High,Low,Open <-- Actual DataFrame header (index 2)
                df = pd.read_csv(
                    user_provided_file_path,
                    skiprows=2,
                    index_col="Date",
                    parse_dates=["Date"],
                )

                # Expected columns from portfolio_data_gen.py output
                expected_cols_from_portfolio = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                ]

                # Check if all expected columns are present
                if not all(col in df.columns for col in expected_cols_from_portfolio):
                    print(
                        f"Error: Missing expected columns in user-provided file {user_provided_file_path}. Expected: {expected_cols_from_portfolio}, Found: {df.columns.tolist()}"
                    )
                    return None

                # Ensure consistent column order and use .copy() to avoid SettingWithCopyWarning
                raw_data_source = df[expected_cols_from_portfolio].copy()
                raw_data_source.index.name = (
                    "Date"  # Explicitly set index name after reading
                )

                print(
                    f"Successfully loaded data from user-provided file. Shape: {raw_data_source.shape}"
                )

                # When loading from a user-provided file, we infer ticker from filename
                chosen_ticker_for_metadata = Path(user_provided_file_path).stem
                # And assume '1d' interval for annual_trading_periods unless a way to infer is added
                chosen_interval_for_metadata = "1d"

            except FileNotFoundError:
                print(
                    f"Error: User-provided data file not found at {user_provided_file_path}"
                )
                return None
            except Exception as e:
                print(
                    f"Error loading data from user-provided file ({user_provided_file_path}): {e}"
                )
                return None

        # --- PRIORITY 2: Check parquet store (local, fast, no network) ---
        # --- PRIORITY 3: Check for cached yfinance data (original logic) ---
        else:  # user_provided_file_path is None, proceed with data loading
            if not all([ticker, start, end, interval]):
                print(
                    "Error: 'ticker', 'start', 'end', and 'interval' are required if no 'user_provided_file_path' is given."
                )
                return None

            # Try parquet store first
            try:
                from algos.common.market_data_store import MarketDataStore

                _store = MarketDataStore()
                if _store.has_ticker(ticker):
                    parquet_df = _store.get_ohlcv(
                        ticker, start, end, use_adj_close=True
                    )
                    if parquet_df is not None and not parquet_df.empty:
                        print(
                            f"Loaded {ticker} from parquet store ({len(parquet_df)} rows)"
                        )
                        raw_data_source = parquet_df
                        chosen_ticker_for_metadata = ticker
                        chosen_interval_for_metadata = interval
            except ImportError:
                pass
            except Exception as e:
                print(f"Parquet store read failed for {ticker}, falling back: {e}")

            # Path for internally cached yfinance data
            internal_cache_file_path = (
                data_dir / f"yfinance_{ticker}_{start}_{end}_{interval}.csv"
            )

            # This will hold the downloaded/cached yfinance data (using original data_loader's `raw` variable logic)
            raw_yfinance_data_from_cache = None

            # Skip CSV cache + download if parquet store already provided data
            if raw_data_source is not None:
                raw_yfinance_data_from_cache = (
                    raw_data_source  # Satisfy downstream checks
                )

            elif internal_cache_file_path.exists():
                try:
                    print(
                        f"Loading cached yfinance data from local file: {internal_cache_file_path.resolve()}"
                    )

                    # This is the original logic for reading cached yfinance CSVs (skips 3 rows, no header)
                    temp_df = pd.read_csv(
                        internal_cache_file_path,
                        header=None,  # No header row provided directly by the cache file, we assign after skipping
                        skiprows=3,  # Skip yfinance's first 3 header-like rows (e.g., 'Date', 'Open', 'High', ...)
                        index_col=0,  # The first column (index 0) is the Date/Datetime column
                        parse_dates=True,  # Parse this index column as dates
                        float_precision="high",  # Ensure high precision for floats (float64)
                    )

                    # Reconstruct yfinance columns based on its usual output (Date, Open, High, Low, Close, Adj Close, Volume)
                    # The original data_loader.py had different column assignments based on num_data_cols
                    num_data_cols = temp_df.shape[1]

                    if num_data_cols == 5:  # Often if 'Adj Close' is same as 'Close'
                        temp_df.columns = ["Close", "High", "Low", "Open", "Volume"]
                        raw_yfinance_data_from_cache = temp_df[
                            ["Open", "High", "Low", "Close", "Volume"]
                        ].copy()
                        raw_yfinance_data_from_cache["Adj Close"] = (
                            raw_yfinance_data_from_cache["Close"]
                        )  # Create 'Adj Close' if missing
                    elif num_data_cols == 6:  # Standard case with 'Adj Close'
                        temp_df.columns = [
                            "Adj Close",
                            "Close",
                            "High",
                            "Low",
                            "Open",
                            "Volume",
                        ]
                        raw_yfinance_data_from_cache = temp_df[
                            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                        ].copy()
                    else:
                        print(
                            f"Error: Unexpected number of data columns ({num_data_cols}) in cached CSV. Expected 5 or 6 after index. "
                            f"Check CSV format for {ticker} at {internal_cache_file_path}."
                        )
                        return None  # Return None on error

                    raw_yfinance_data_from_cache.index.name = (
                        "Date" if interval in ["1d", "1wk", "1mo"] else "Datetime"
                    )
                    print("Successfully loaded cached yfinance data.")

                except Exception as e:
                    print(
                        f"Error loading cached yfinance data from CSV ({internal_cache_file_path}): {e}. Attempting to download."
                    )
                    raw_yfinance_data_from_cache = (
                        None  # Force download if cached load fails
                    )
            else:
                print(
                    f"Cached yfinance file not found: {internal_cache_file_path.resolve()}. Attempting to download."
                )
                raw_yfinance_data_from_cache = (
                    None  # Force download if file doesn't exist
                )

            # --- PRIORITY 3: Download from yfinance (with resilient retry) ---
            if (
                raw_yfinance_data_from_cache is None
                or raw_yfinance_data_from_cache.empty
            ):
                try:
                    print(f"Downloading data for {ticker} from Yahoo Finance...")
                    if _HAS_RESILIENT_DOWNLOADER:
                        raw_yfinance_data_from_download = resilient_download_single(
                            ticker,
                            start=start,
                            end=end,
                            interval=interval,
                            progress=False,
                        )
                    else:
                        raw_yfinance_data_from_download = yf.download(
                            ticker,
                            start=start,
                            end=end,
                            interval=interval,
                            progress=False,
                            ignore_tz=True,
                        )
                    if (
                        raw_yfinance_data_from_download is None
                        or raw_yfinance_data_from_download.empty
                    ):
                        print(
                            f"No data downloaded for {ticker} with interval {interval} from {start} to {end}."
                        )
                        return pd.DataFrame()  # Return empty DataFrame

                    # Save to CSV for future faster loading (yfinance saves in its own consistent format)
                    raw_yfinance_data_from_download.to_csv(internal_cache_file_path)
                    print(
                        f"Data downloaded and saved to: {internal_cache_file_path.resolve()}"
                    )

                    raw_data_source = raw_yfinance_data_from_download  # Assign downloaded data to raw_data_source
                    chosen_ticker_for_metadata = ticker
                    chosen_interval_for_metadata = interval

                except Exception as e:
                    print(f"Fatal Error: Could not download data for {ticker}: {e}")
                    return None  # Return None on fatal error
            else:
                raw_data_source = raw_yfinance_data_from_cache  # Assign cached data to raw_data_source
                chosen_ticker_for_metadata = ticker
                chosen_interval_for_metadata = interval

        if raw_data_source is None or raw_data_source.empty:
            print("Error: No data loaded from any source.")
            return None

        # Ensure index is datetime and sorted (common for both sources)
        raw_data_source.index = pd.to_datetime(raw_data_source.index)
        raw_data_source = raw_data_source.sort_index()

        # --- Flexible Symbol Selection Logic ---
        # Prioritize user-provided symbol, then fallbacks
        chosen_symbol = None
        # Add 'AdjClose' to potential symbols as yfinance might return it directly before renaming
        potential_symbols = [symbol, "Adj Close", "AdjClose", "Close", "Open"]

        for s in potential_symbols:
            if s in raw_data_source.columns and s is not None:
                chosen_symbol = s
                break

        if chosen_symbol is None:
            print(
                f"Fatal Error: None of the preferred price columns ('{symbol}', 'Adj Close', 'Close', 'Open') found in data."
            )
            return None  # Return None instead of sys.exit

        # If 'AdjClose' was selected from yfinance output, rename it to 'Adj Close' for consistency
        if chosen_symbol == "AdjClose":
            if "AdjClose" in raw_data_source.columns:  # Check existence before renaming
                raw_data_source.rename(columns={"AdjClose": "Adj Close"}, inplace=True)
            chosen_symbol = (
                "Adj Close"  # Update chosen_symbol to the new standardized name
            )

        # --- Preserve OHLCV columns for feature engineering ---
        # Map raw column names to standardized lowercase names
        ohlcv_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        data = pd.DataFrame(index=raw_data_source.index)

        # Copy all available OHLCV columns with lowercase names
        for raw_col, std_col in ohlcv_mapping.items():
            if raw_col in raw_data_source.columns:
                col_data = raw_data_source[raw_col]
                if pd.api.types.is_numeric_dtype(col_data) or col_data.dtype == object:
                    data[std_col] = pd.to_numeric(col_data, errors="coerce")
                else:
                    data[std_col] = col_data

        # Set price column (same logic as before, for backward compat)
        data["price"] = pd.to_numeric(raw_data_source[chosen_symbol], errors="coerce")
        data = data.dropna(subset=["price"])

        if data.empty:
            print(
                f"Fatal Error: 'data' DataFrame is empty after initial processing for {chosen_ticker_for_metadata}. Exiting."
            )
            return None  # Return None instead of sys.exit

        # Calculate log returns
        data["returns"] = np.log(data["price"] / data["price"].shift(1))
        data.dropna(subset=["returns"], inplace=True)

        if data.empty:
            print(
                f"Fatal Error: 'data' DataFrame is empty after calculating returns and dropping NaNs for {chosen_ticker_for_metadata}. Exiting."
            )
            return None  # Return None instead of sys.exit

        # Calculate direction: carry-adjusted for forex, raw for stocks/crypto
        is_forex = _is_forex_ticker(chosen_ticker_for_metadata)
        if is_forex:
            # Compute carry differential for forex pairs (USD vs JPY)
            # This adjusts the target so that small FX losses within the carry
            # spread are still classified as direction +1 (favorable hold)
            carry_diff = compute_carry_differential(data.index)
            data["direction"] = compute_direction(
                data["returns"], chosen_ticker_for_metadata, carry_diff
            )
            data["carry_differential"] = carry_diff.reindex(data.index, method="ffill")
            print(
                f"Forex pair detected: using carry-adjusted direction for {chosen_ticker_for_metadata}. "
                f"Mean carry spread: {carry_diff.mean():.4f} ({carry_diff.mean() * 100:.2f}% annualized)"
            )
        else:
            data["direction"] = compute_direction(
                data["returns"], chosen_ticker_for_metadata
            )

        # --- Set annual_trading_periods based on interval and asset type ---
        # Use chosen_interval_for_metadata for calculation
        is_crypto = _is_crypto_ticker(chosen_ticker_for_metadata)
        try:
            data.attrs["annual_trading_periods"] = _calculate_annual_trading_periods(
                chosen_interval_for_metadata, is_crypto, is_forex
            )
            print(
                f"Calculated annual_trading_periods: {data.attrs['annual_trading_periods']} "
                f"for interval '{chosen_interval_for_metadata}' and asset type "
                f"(crypto: {is_crypto}, forex: {is_forex})."
            )
        except ValueError as e:
            print(
                f"Fatal Error: {e}. Cannot determine annual trading periods. Exiting."
            )
            return None  # Return None instead of sys.exit

        # Store other useful metadata in attrs for later use in metrics/risk analysis
        data.attrs["ticker"] = chosen_ticker_for_metadata
        data.attrs["symbol"] = chosen_symbol  # Store the actual symbol used
        data.attrs["start"] = (
            start if not user_provided_file_path else None
        )  # Original start/end if yfinance used
        data.attrs["end"] = end if not user_provided_file_path else None
        data.attrs["interval"] = (
            chosen_interval_for_metadata  # Store the actual interval used
        )

        print("=" * 50)
        print(
            f"\nPreprocessed Data Info for {chosen_ticker_for_metadata} (Source: {'File' if user_provided_file_path else 'YFinance'}):\n"
        )
        print("=" * 50 + "\n")
        data.info()
        print("=" * 50 + "\n" * 3)
        print(f"Data Head:\n{data.head()}\n")
        print(f"Data Tail:\n{data.tail()}\n")

        # The RedirectStdoutToFile context manager will handle closing the file
        return data
