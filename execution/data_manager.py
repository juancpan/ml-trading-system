import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import threading

# Import config from current directory
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add project root for algos imports
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config

SYMBOLS = config.SYMBOLS

# Feature engine imports (optional — graceful fallback to legacy)
try:
    from algos.common.feature_engine import (
        FeatureEngine,
        FeatureConfig,
        load_feature_metadata,
    )
    from algos.common.external_data import ExternalDataCache

    _HAS_FEATURE_ENGINE = True
except ImportError:
    _HAS_FEATURE_ENGINE = False


class DataManager:
    """Manages historical OHLC data for strategy input, using yfinance."""

    def __init__(self, logger, exchange_manager=None):
        self.logger = logger
        self.exchange_manager = (
            exchange_manager  # For symbol conversion (config -> yfinance)
        )
        # Stores history as a DataFrame for each symbol, with common columns
        self.historical_data = {
            symbol: pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )  # yfinance columns
            for symbol in SYMBOLS
        }
        # Keep enough history for any reasonable 'lags' value (e.g., 20 days for lags up to 19)
        # IMPORTANT: Some models may have been trained with longer sequences and need more history
        self.max_history_length = 200  # Increased to match potential training context

        # Thread safety for parallel fetching
        self.data_lock = threading.Lock()

        # Primary IBKR data source (set via main.py after IBKR connection)
        self.ibkr_data_manager = None

    def fetch_and_store_historical_data(self, symbol: str, end_date: datetime.date):
        """
        Fetches historical daily data for a symbol using yfinance up to the end_date.
        Thread-safe for parallel fetching.

        Args:
            symbol (str): The ticker symbol.
            end_date (datetime.date): The date up to which to fetch data (exclusive for yfinance 'end' parameter).
                                      So, if you want data *for* July 25, set end_date to July 26.
        """
        # Calculate a start date far enough back to cover max_history_length
        # Adding extra buffer to account for weekends and holidays
        start_date = end_date - timedelta(
            days=self.max_history_length + 100
        )  # Increased buffer

        # Convert config symbol to yfinance symbol if exchange_manager is available
        # Handles special cases like RR. -> RR.L, BA. -> BA.L
        yf_symbol = symbol
        if self.exchange_manager is not None:
            yf_symbol = self.exchange_manager.get_yfinance_symbol(symbol)
            if yf_symbol != symbol:
                self.logger.info(
                    f"Symbol conversion: {symbol} -> {yf_symbol} for yfinance"
                )

        self.logger.info(
            f"Fetching historical data for {symbol} from {start_date} to {end_date}..."
        )

        # Try parquet store first (local, fast, no network needed for historical warmup)
        try:
            from algos.common.market_data_store import MarketDataStore

            _store = MarketDataStore()
            store_ticker = (
                yf_symbol.replace("=X", "") if "=X" in yf_symbol else yf_symbol
            )
            if _store.has_ticker(store_ticker) and _store.check_freshness(
                store_ticker, max_age_hours=48
            ):
                parquet_df = _store.get_ohlcv(
                    store_ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    use_adj_close=True,
                )
                if (
                    parquet_df is not None
                    and len(parquet_df) >= self.max_history_length // 2
                ):
                    self.logger.info(
                        f"Loaded {symbol} from parquet store ({len(parquet_df)} rows)"
                    )
                    # Process and store -- skip yfinance download entirely
                    with self.data_lock:
                        self.historical_data[symbol] = parquet_df.sort_index().tail(
                            self.max_history_length
                        )
                    self.logger.info(
                        f"Stored {len(self.historical_data[symbol])} bars for {symbol} from parquet"
                    )
                    return
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"Parquet store not available for {symbol}: {e}")

        # === Source 2: IBKR historical bars (primary, authoritative) ===
        if self.ibkr_data_manager is not None:
            ibkr_df = self.ibkr_data_manager.fetch_historical_bars(
                symbol, num_days=self.max_history_length + 50
            )
            if ibkr_df is not None and not ibkr_df.empty:
                # Ensure consistent column types
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in ibkr_df.columns:
                        ibkr_df[col] = pd.to_numeric(ibkr_df[col], errors="coerce")
                ibkr_df = ibkr_df.dropna(subset=["Close"])

                if not ibkr_df.empty:
                    final_df = ibkr_df.tail(self.max_history_length)
                    with self.data_lock:
                        self.historical_data[symbol] = final_df

                    self.logger.info(
                        f"Successfully fetched and stored {len(final_df)} historical bars "
                        f"for {symbol} from IBKR."
                    )

                    # Log last 3 bars (same format as existing yfinance logging)
                    last_bars = final_df.tail(3)
                    self.logger.info(f"\nLatest data for {symbol} (last 3 bars):")
                    self.logger.info(
                        f"{'Date':<12} {'Close':>12} {'Open':>12} {'High':>12} {'Low':>12} {'Volume':>12}"
                    )
                    self.logger.info("-" * 72)
                    for date_idx, row in last_bars.iterrows():
                        date_str = (
                            date_idx.strftime("%Y-%m-%d")
                            if hasattr(date_idx, "strftime")
                            else str(date_idx)
                        )
                        self.logger.info(
                            f"{date_str:<12} {row.get('Close', 0):>12.2f} "
                            f"{row.get('Open', 0):>12.2f} {row.get('High', 0):>12.2f} "
                            f"{row.get('Low', 0):>12.2f} {row.get('Volume', 0):>12.0f}"
                        )

                    latest_date = final_df.index[-1]
                    latest_close = final_df["Close"].iloc[-1]
                    day_name = (
                        latest_date.strftime("%A")
                        if hasattr(latest_date, "strftime")
                        else "Unknown"
                    )
                    self.logger.info(
                        f">>> USING: {latest_date.date()} ({day_name}) Close = {latest_close:.4f} <<< [source: IBKR]\n"
                    )
                    return

            self.logger.warning(
                f"{symbol}: IBKR historical bars unavailable. Falling back to yfinance."
            )

        # === Source 3: yfinance fallback (for symbols IBKR doesn't cover) ===
        try:
            import yfinance as yf

            # CRITICAL: Use auto_adjust=True to match backtesting (OptimizedDataLoader)
            # Models were trained using algos/common/data_cache.py which uses auto_adjust=True
            df = yf.download(
                yf_symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            # CRITICAL FIX: yfinance sometimes returns MultiIndex columns for single ticker
            # Flatten columns if MultiIndex is detected
            if isinstance(df.columns, pd.MultiIndex):
                # For single ticker, columns are like: ('Close', 'AVGO'), ('Open', 'AVGO'), etc.
                # Flatten to just: 'Close', 'Open', etc.
                df.columns = df.columns.get_level_values(0)
                self.logger.debug(f"{symbol}: Flattened MultiIndex columns")

            # If yfinance returns an empty DataFrame, handle it immediately
            if df.empty:
                self.logger.warning(
                    f"No historical data returned from yfinance for {symbol} between {start_date} and {end_date}. DataFrame is empty upon download."
                )
                with self.data_lock:
                    self.historical_data[symbol] = pd.DataFrame(
                        columns=["Open", "High", "Low", "Close", "Volume"]
                    )
                return

            # Drop rows where all values are NaN *after* initial download, before processing columns
            df = df.dropna(how="all")
            if df.empty:  # Check again if it became empty after dropping all NaNs
                self.logger.warning(
                    f"Historical data for {symbol} became empty after dropping all NaN rows (all-NaN rows)."
                )
                with self.data_lock:
                    self.historical_data[symbol] = pd.DataFrame(
                        columns=["Open", "High", "Low", "Close", "Volume"]
                    )
                return

            # Ensure consistent column names (yfinance uses 'Adj Close', we might prefer 'Close')
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]

            required_cols = ["Open", "High", "Low", "Close", "Volume"]

            # --- NEW: Explicitly select columns to ensure they are proper Series ---
            # Thread-safe column selection with proper Series extraction
            temp_df = pd.DataFrame(index=df.index)
            for col in required_cols:
                if col in df.columns:
                    # Use .copy() to ensure we get a proper Series, not a view
                    col_data = df[col].copy()
                    # Ensure it's actually a Series (handle multi-index edge cases)
                    if isinstance(col_data, pd.DataFrame):
                        # If somehow a DataFrame, take the first column
                        col_data = col_data.iloc[:, 0]
                    temp_df[col] = col_data
                else:
                    self.logger.warning(
                        f"Column '{col}' not found in yfinance data for {symbol} after initial download. It won't be processed."
                    )
            df = temp_df.copy()
            # --- END NEW ---

            processed_numeric_cols = []
            for col in required_cols:  # Iterate through required_cols, not df.columns
                if (
                    col in df.columns
                ):  # Check if the column actually exists in the (potentially new) df
                    # Attempt conversion, errors='coerce' will turn non-numeric into NaN
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                    # After coercing, if the column is not entirely NaN, add it to our list
                    if not df[col].isnull().all():
                        processed_numeric_cols.append(col)
                    else:
                        self.logger.warning(
                            f"Column '{col}' for {symbol} became all NaN after numeric conversion. It might have contained non-numeric or empty data."
                        )
                # else: warning already logged above if column not found during initial selection

            if not processed_numeric_cols:
                self.logger.error(
                    f"No valid numeric columns could be processed with actual data for {symbol}. Cannot proceed with data cleaning."
                )
                with self.data_lock:
                    self.historical_data[symbol] = pd.DataFrame(
                        columns=["Open", "High", "Low", "Close", "Volume"]
                    )
                return

            # Perform dropna on the columns that successfully became numeric and had data
            df = df.dropna(subset=processed_numeric_cols)

            # If the DataFrame becomes empty after dropping NaNs, reset and return.
            if df.empty:
                self.logger.warning(
                    f"Historical data for {symbol} became empty after dropping NaNs based on {processed_numeric_cols}. Cannot proceed."
                )
                with self.data_lock:
                    self.historical_data[symbol] = pd.DataFrame(
                        columns=["Open", "High", "Low", "Close", "Volume"]
                    )
                return

            # Keep only the relevant and successfully processed columns
            df = df[processed_numeric_cols]

            # Sort by index (date) and keep only the most recent 'max_history_length' bars
            final_df = df.sort_index().tail(self.max_history_length)

            # Thread-safe storage
            with self.data_lock:
                self.historical_data[symbol] = final_df

            self.logger.info(
                f"Successfully fetched and stored {len(final_df)} historical bars for {symbol} from yfinance."
            )

            # Show last 3 bars for verification (especially important for Monday - verify we have Friday data)
            last_bars = final_df.tail(3)
            self.logger.info(f"\nLatest data for {symbol} (last 3 bars):")
            self.logger.info(
                f"{'Date':<12} {'Close':>12} {'Open':>12} {'High':>12} {'Low':>12} {'Volume':>12}"
            )
            self.logger.info("-" * 72)
            for date_idx, row in last_bars.iterrows():
                date_str = (
                    date_idx.strftime("%Y-%m-%d")
                    if hasattr(date_idx, "strftime")
                    else str(date_idx)
                )
                close = row.get("Close", 0)
                open_price = row.get("Open", 0)
                high = row.get("High", 0)
                low = row.get("Low", 0)
                volume = row.get("Volume", 0)
                self.logger.info(
                    f"{date_str:<12} {close:>12.2f} {open_price:>12.2f} {high:>12.2f} {low:>12.2f} {volume:>12.0f}"
                )

            # Highlight the latest bar (what we'll use for signals)
            latest_date = final_df.index[-1]
            latest_close = final_df["Close"].iloc[-1]
            day_name = (
                latest_date.strftime("%A")
                if hasattr(latest_date, "strftime")
                else "Unknown"
            )
            self.logger.info(
                f">>> USING: {latest_date.date()} ({day_name}) Close = {latest_close:.4f} <<<\n"
            )

        except Exception as e:
            self.logger.error(
                f"Error fetching historical data for {symbol} from yfinance: {e}",
                exc_info=True,
            )
            with self.data_lock:
                self.historical_data[symbol] = pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"]
                )

    def get_data_for_strategy(
        self, symbol, lags=5
    ):  # Default lags to 5 for consistency
        """
        Retrieves and processes historical OHLC data to generate lagged features for the ML strategy.
        FIXED to match backtesting preprocessing exactly.
        Thread-safe for parallel access.
        """
        # Thread-safe read from historical data
        with self.data_lock:
            history_df = self.historical_data.get(symbol)
            if history_df is not None and not history_df.empty:
                # Make a copy to avoid holding lock during processing
                history_df = history_df.copy()
            else:
                history_df = None

        # We need at least (lags + 1) rows to calculate 'lags' features (one for current return, then 'lags' shifts)
        if history_df is None or history_df.empty or len(history_df) < (lags + 1):
            self.logger.warning(
                f"Not enough historical data ({len(history_df) if history_df is not None else 0} bars) for {symbol} to generate {lags} lagged features. Need at least {lags + 1} valid bars after all processing."
            )
            return None

        # Check if 'Close' column exists and is numeric
        if "Close" not in history_df.columns or not pd.api.types.is_numeric_dtype(
            history_df["Close"]
        ):
            self.logger.error(
                f"Missing or non-numeric 'Close' column in history for {symbol}. This should not happen if fetch_and_store_historical_data worked correctly."
            )
            return None

        temp_df = history_df.copy()

        # Use LOG returns as requested by user
        # np.log(price[t] / price[t-1]) is more mathematically sound for aggregation
        temp_df["returns"] = np.log(temp_df["Close"] / temp_df["Close"].shift(1))

        # Create lagged features exactly as in backtesting
        feature_cols = []
        for lag in range(1, lags + 1):
            col = f"lag_{lag}"
            temp_df[col] = temp_df["returns"].shift(lag)
            feature_cols.append(col)

        # Ensure all feature columns are present in temp_df before trying to use them
        if not all(col in temp_df.columns for col in feature_cols):
            self.logger.error(
                f"Not all required feature columns {feature_cols} could be created in the DataFrame for {symbol}."
            )
            return None

        # Drop rows with NaN values introduced by shifting for lagged features.
        features_row_df = temp_df.dropna(subset=feature_cols).tail(1)

        if features_row_df.empty:
            self.logger.warning(
                f"Could not generate a complete set of {lags} lagged features for {symbol} from available history after dropping NaNs. This usually means not enough valid data points after calculating returns and lags."
            )
            return None

        # Extract the features as a NumPy array
        features = features_row_df[feature_cols].values

        # Get model type to determine if scaler is needed
        model_config = config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = model_config.get("model_type", "standard")

        # ARIMA models don't use scalers (they work with raw returns)
        if model_type == "arima":
            self.logger.debug(f"ARIMA model for {symbol} - using unscaled features")
            return features.reshape(1, -1)

        # FIX: Apply StandardScaler if available (for non-ARIMA models like li_reg, lstm, svm, etc.)
        try:
            scaler = self.load_scaler(symbol)
            if scaler is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                self.logger.debug(
                    f"Applied StandardScaler to {symbol} features for {model_type} model"
                )
                return features_scaled
            else:
                self.logger.warning(
                    f"No scaler found for {symbol} ({model_type}), using unscaled features (may cause prediction issues)"
                )
                return features.reshape(1, -1)
        except Exception as e:
            self.logger.warning(f"Could not apply scaler for {symbol}: {e}")
            # Return unscaled features if scaler fails
            return features.reshape(1, -1)

    def get_latest_historical_bar(self, symbol):
        """Returns the latest historical bar DataFrame (single row) for a given symbol. Thread-safe."""
        with self.data_lock:
            history_df = self.historical_data.get(symbol)
            if history_df is not None and not history_df.empty:
                return history_df.iloc[-1:].copy()  # Return copy to avoid lock holding
        return None

    def create_sequence_data(
        self, symbol: str, sequence_length: int = 20, lags: int = None
    ) -> np.ndarray:
        """
        Create sequence data for LSTM/RNN models - FIXED to match backtesting.
        Thread-safe for parallel access.

        Args:
            symbol: Stock symbol
            sequence_length: Number of timesteps in each sequence (deprecated, use lags)
            lags: Number of lagged features (defaults to 5 for consistency with backtesting)

        Returns:
            Array of shape (1, lags, 1) for LSTM input with properly scaled features
        """
        # Use lags parameter for consistency with backtesting
        if lags is None:
            lags = 5  # Default to 5 lags as used in backtesting

        # Thread-safe read
        with self.data_lock:
            history_df = self.historical_data.get(symbol)
            if history_df is not None and not history_df.empty:
                history_df = history_df.copy()
            else:
                history_df = None

        if history_df is None or len(history_df) < (lags + 1):
            self.logger.warning(
                f"Not enough historical data for {symbol} LSTM sequence. Need {lags + 1}, have {len(history_df) if history_df is not None else 0}"
            )
            return np.zeros((1, lags, 1))

        temp_df = history_df.copy()

        # Use LOG returns as requested by user
        # np.log(price[t] / price[t-1]) is more mathematically sound for aggregation
        temp_df["returns"] = np.log(temp_df["Close"] / temp_df["Close"].shift(1))

        # FIX 2: Create lagged features exactly like backtesting
        feature_cols = []
        for lag in range(1, lags + 1):
            col = f"lag_{lag}"
            temp_df[col] = temp_df["returns"].shift(lag)
            feature_cols.append(col)

        # Get latest complete row with all lagged features
        features_row = temp_df.dropna(subset=feature_cols).tail(1)

        if features_row.empty:
            self.logger.warning(
                f"Could not generate complete lagged features for {symbol}"
            )
            return np.zeros((1, lags, 1))

        # Extract the lagged features
        features = features_row[feature_cols].values

        # FIX 3: Apply StandardScaler if available
        # lstm_optimized trains scaler on flattened data (n_features_in_=1)
        # So we must: flatten to column -> scale -> reshape back
        try:
            scaler = self.load_scaler(symbol)
            if scaler is not None:
                # Flatten to column (lags, 1), scale each value, reshape to (1, lags)
                features_scaled = scaler.transform(features.reshape(-1, 1)).reshape(
                    1, -1
                )
                self.logger.debug(f"Applied StandardScaler to {symbol} features")
            else:
                features_scaled = features.reshape(1, -1)
                self.logger.warning(
                    f"No scaler found for {symbol}, using unscaled features"
                )
        except Exception as e:
            self.logger.warning(f"Could not apply scaler for {symbol}: {e}")
            features_scaled = features.reshape(1, -1)

        # FIG 4: Reshape for LSTM: (1, lags, 1)
        return features_scaled.reshape(1, lags, 1)

    def load_scaler(self, symbol: str):
        """
        Load the StandardScaler used during model training.

        Args:
            symbol: Stock symbol

        Returns:
            StandardScaler object or None if not found
        """
        import pickle

        # Import config from current directory, not from algos.common
        import sys
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Now import from local config
        import config as local_config

        ASSET_SPECIFIC_CONFIGS = local_config.ASSET_SPECIFIC_CONFIGS

        config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = config.get("model_type", "standard")

        # Try multiple possible scaler paths
        # Use absolute paths based on current directory
        scaler_paths = [
            os.path.join(
                current_dir, f"strategy_models/{model_type}_scaler_{symbol}.pkl"
            ),
            os.path.join(current_dir, f"strategy_models/scaler_{symbol}.pkl"),
            os.path.join(current_dir, f"strategy_models/{symbol}_scaler.pkl"),
        ]

        for scaler_path in scaler_paths:
            try:
                import warnings

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                    for w in caught:
                        self.logger.warning(
                            f"Scaler load warning for {symbol}: {w.message}"
                        )
                    self.logger.info(f"Loaded scaler for {symbol} from {scaler_path}")
                    return scaler
            except FileNotFoundError:
                continue
            except Exception as e:
                self.logger.warning(f"Error loading scaler from {scaler_path}: {e}")

        self.logger.warning(f"No scaler found for {symbol} in any expected location")
        return None

    def get_engineered_features(
        self,
        symbol: str,
        feature_config=None,
        external_data: dict = None,
        lags: int = 5,
    ) -> tuple:
        """
        Compute features using the centralized FeatureEngine.
        Uses THE SAME CODE PATH as backtesting for consistency.
        Falls back to legacy get_data_for_strategy() if feature engine unavailable.

        Args:
            symbol: Stock symbol
            feature_config: FeatureConfig instance
            external_data: Dict mapping feature_name -> pd.Series
            lags: Fallback lag count if feature engine unavailable

        Returns:
            Tuple of (features_array shape (1, n), feature_names list)
            Returns (None, []) if insufficient data
        """
        if not _HAS_FEATURE_ENGINE or feature_config is None:
            # Legacy fallback
            features = self.get_data_for_strategy(symbol, lags=lags)
            if features is None:
                return None, []
            legacy_names = [f"lag_{i}" for i in range(1, lags + 1)]
            return features, legacy_names

        # Thread-safe read of historical data
        with self.data_lock:
            history_df = self.historical_data.get(symbol)
            if history_df is not None and not history_df.empty:
                history_df = history_df.copy()
            else:
                history_df = None

        max_warmup = feature_config.get_max_warmup()
        min_rows = max_warmup + 10  # Need warmup + some buffer

        if history_df is None or history_df.empty or len(history_df) < min_rows:
            self.logger.warning(
                f"Not enough historical data ({len(history_df) if history_df is not None else 0} bars) "
                f"for {symbol} feature engineering. Need at least {min_rows} bars "
                f"(warmup={max_warmup})."
            )
            return None, []

        # Prepare OHLCV DataFrame with lowercase columns (matching feature engine expectations)
        ohlcv = pd.DataFrame(index=history_df.index)
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        for raw_col, std_col in col_map.items():
            if raw_col in history_df.columns:
                ohlcv[std_col] = history_df[raw_col]

        # Add price and returns (same as backtest data_loader)
        ohlcv["price"] = ohlcv["close"]
        ohlcv["returns"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        ohlcv["direction"] = np.where(ohlcv["returns"] > 0, 1, -1)
        ohlcv = ohlcv.dropna(subset=["returns"])

        if ohlcv.empty:
            self.logger.warning(
                f"OHLCV data empty after computing returns for {symbol}"
            )
            return None, []

        # Use the EXACT SAME feature engine as backtesting
        engine = FeatureEngine()
        try:
            features_array, feature_names = engine.compute_live_features(
                ohlcv, feature_config, external_data
            )
        except Exception as e:
            self.logger.error(f"Feature engine error for {symbol}: {e}")
            return None, []

        if features_array.size == 0:
            self.logger.warning(f"Feature engine produced empty features for {symbol}")
            return None, []

        # Validate feature count against scaler
        model_config = config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = model_config.get("model_type", "standard")

        # ARIMA models don't use scalers
        if model_type == "arima":
            self.logger.debug(
                f"ARIMA model for {symbol} - using unscaled engineered features"
            )
            return features_array, feature_names

        # Apply scaler
        try:
            scaler = self.load_scaler(symbol)
            if scaler is not None:
                expected_n = getattr(scaler, "n_features_in_", None)
                actual_n = features_array.shape[1]
                if expected_n is not None and expected_n != actual_n:
                    self.logger.error(
                        f"FEATURE MISMATCH for {symbol}: scaler expects {expected_n} features "
                        f"but engine produced {actual_n}. Model was likely trained with a "
                        f"different feature config. Refusing to predict with mismatched features."
                    )
                    return None, []
                features_scaled = scaler.transform(features_array)
                self.logger.debug(
                    f"Applied StandardScaler to {symbol} engineered features "
                    f"({actual_n} features)"
                )
                return features_scaled, feature_names
            else:
                self.logger.warning(
                    f"No scaler found for {symbol} ({model_type}), "
                    f"using unscaled engineered features"
                )
                return features_array, feature_names
        except Exception as e:
            self.logger.warning(f"Could not apply scaler for {symbol}: {e}")
            return features_array, feature_names

    def load_feature_metadata(self, symbol: str) -> dict:
        """
        Load feature metadata for a deployed model.
        Used to validate feature alignment between training and inference.

        Returns:
            Metadata dict or empty dict if not found
        """
        if not _HAS_FEATURE_ENGINE:
            return {}

        model_config = config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = model_config.get("model_type", "standard")

        meta = load_feature_metadata(
            symbol=symbol,
            model_type=model_type,
            load_dir=os.path.join(current_dir, "strategy_models"),
        )
        return meta or {}
