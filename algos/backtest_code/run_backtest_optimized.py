"""
Optimized backtesting runner with improved efficiency and parallel processing.
"""

import os

# Prevent HDF5 file lock contention when multiple backtest processes run concurrently.
# Must be set BEFORE importing h5py/tensorflow/keras.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import pandas as pd
import numpy as np
import sys
import time
import importlib
import inspect
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from dateutil import parser as dateutil_parser

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import seed manager for reproducibility
from algos.common.seed_manager import set_seed, get_model_seed, save_seed_info

# Auto-save scalers when backtesting
try:
    from algos.backtest_code.apply_scaler_patch import ensure_scaler_saved

    SCALER_AUTOSAVE_ENABLED = True
except ImportError:
    SCALER_AUTOSAVE_ENABLED = False
    print(
        "Note: Scaler auto-save not available. Install apply_scaler_patch.py for automatic scaler saving."
    )

from algos.common.data_cache import OptimizedDataLoader
from algos.common.metrics import calculate_strategy_performance
from algos.common.risk_analysis import calculate_risk_metrics, plot_drawdowns
from algos.common.utils import RedirectStdoutToFile
import algos.common.persistence as save_model
from algos.common.persistence import save_arima_settings
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (used in lightweight WFOV mode)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Import crypto data fetcher for consistent data
try:
    from crypto_trading.utils.consistent_data_fetcher import ConsistentDataFetcher

    CONSISTENT_FETCHER_AVAILABLE = True
except ImportError:
    CONSISTENT_FETCHER_AVAILABLE = False

try:
    from crypto_trading.utils.hybrid_data_fetcher import HybridDataFetcher

    HYBRID_FETCHER_AVAILABLE = True
except ImportError:
    HYBRID_FETCHER_AVAILABLE = False

if not CONSISTENT_FETCHER_AVAILABLE and not HYBRID_FETCHER_AVAILABLE:
    print("Warning: No crypto data fetchers available, using yfinance for all data")


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""

    model_name: str
    ticker: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    interval: str
    data_path: Optional[str]
    train_split: float
    rf_rate: float
    ptc: float
    symbol: str
    model_params: Dict[str, Any]
    force_data_source: Optional[str] = None  # 'yfinance', 'ccxt', or None for auto
    max_leverage: float = 4.0  # Maximum leverage cap for position sizing
    embargo_pct: float = (
        0.02  # Embargo percentage for Lopez de Prado's purged CV (default 2%)
    )
    no_plots: bool = False  # Skip generating plots/images (for batch/summary modes)
    save_intermediates: bool = (
        True  # Save intermediate CSV files (raw, processed, features, predictions)
    )
    skip_model_save: bool = (
        False  # Skip saving model/scaler/seed_info files (for WFOV iterations)
    )
    train_end_date: Optional[str] = (
        None  # Explicit train end date (overrides train_split ratio)
    )
    test_start_date: Optional[str] = (
        None  # Explicit test start date (after embargo gap)
    )
    iteration_seed: int = None  # Per-iteration seed for WFOV randomization


class ModelRegistry:
    """Dynamic model registry with lazy loading."""

    def __init__(self):
        self._models = {}
        self._loaded_modules = {}
        self._register_models()

    def _register_models(self):
        """Register available models including optimized versions."""
        model_mappings = {
            # Original models
            "dqn": "algos.backtest_code.models.dqn_model:run_dqn_strategy",
            "cnn": "algos.backtest_code.models.cnn_model:run_cnn_strategy",
            "arima": "algos.backtest_code.models.arima_model:run_arima_strategy",
            "dnn": "algos.backtest_code.models.dnn_model:run_dnn_strategy",
            "sklearn_dnn": "algos.backtest_code.models.sklearn_dnn_model:run_sklearn_dnn_strategy",
            "gbm": "algos.backtest_code.models.gbm_model:run_xgboost_strategy",
            "gnb": "algos.backtest_code.models.gnb_model:run_gnb_strategy",
            "kmeans": "algos.backtest_code.models.kmeans_model:run_kmeans_strategy",
            "li_reg": "algos.backtest_code.models.linear_regression_model:run_linear_regression_strategy",
            "log_reg": "algos.backtest_code.models.logistic_regression_model:run_logistic_regression_strategy",
            "lstm": "algos.backtest_code.models.lstm_model:run_lstm_strategy",
            "rf": "algos.backtest_code.models.random_forest_model:run_random_forest_strategy",
            "sarimax": "algos.backtest_code.models.sarimax_model:run_sarimax_strategy",
            "svm": "algos.backtest_code.models.svm_model:run_svm_strategy",
            "tcn": "algos.backtest_code.models.tcn_model:run_tcn_strategy",
            "var": "algos.backtest_code.models.var_model:run_var_strategy",
            "arima_v2": "algos.backtest_code.models.arima_model_v2:run_arima_v2_strategy",
            "svm_tuned": "algos.backtest_code.models.svm_tuned_model:run_svm_tuned_strategy",
            "stacking": "algos.backtest_code.models.stacking_model:run_stacking_strategy",
            # Optimized models
            "svm_optimized": "algos.backtest_code.models.svm_model_optimized:run_svm_strategy",
            "rf_optimized": "algos.backtest_code.models.random_forest_optimized:run_random_forest_strategy",
            "random_forest_optimized": "algos.backtest_code.models.random_forest_optimized:run_random_forest_strategy",
            "lstm_optimized": "algos.backtest_code.models.lstm_optimized:run_lstm_strategy",
            "xgb_optimized": "algos.backtest_code.models.xgboost_optimized:run_xgboost_strategy",
            "xgboost_optimized": "algos.backtest_code.models.xgboost_optimized:run_xgboost_strategy",
            "linear_optimized": "algos.backtest_code.models.linear_models_optimized:run_linear_regression_strategy",
            "logistic_optimized": "algos.backtest_code.models.linear_models_optimized:run_logistic_regression_strategy",
            "sgd_optimized": "algos.backtest_code.models.linear_models_optimized:run_sgd_linear_strategy",
            "ensemble_optimized": "algos.backtest_code.models.ensemble_optimized:run_ensemble_strategy",
            "ensemble_voting": "algos.backtest_code.models.ensemble_optimized:run_ensemble_strategy",
            "ensemble_stacking": "algos.backtest_code.models.ensemble_optimized:run_ensemble_strategy",
            "ensemble_adaptive": "algos.backtest_code.models.ensemble_optimized:run_adaptive_ensemble_strategy",
            "ensemble_oof": "algos.backtest_code.models.ensemble_oof:run_ensemble_oof_strategy",
        }

        for name, import_path in model_mappings.items():
            self._models[name] = import_path

    def get_model(self, model_name: str):
        """Lazy load and return model function."""
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        if model_name not in self._loaded_modules:
            import_path = self._models[model_name]
            module_path, func_name = import_path.split(":")

            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                self._loaded_modules[model_name] = func
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to load model '{model_name}': {e}")

        return self._loaded_modules[model_name]

    def list_models(self):
        """Return list of available models."""
        return list(self._models.keys())


class OptimizedBacktester:
    """Optimized backtesting engine with caching and parallel processing."""

    def __init__(self, logs_dir: Path = None, models_dir: Path = None):
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[1]
        self.logs_dir = logs_dir or script_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir or repo_root / "algos" / "model_dumps"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_loader = OptimizedDataLoader(cache_ttl_hours=24)
        self.model_registry = ModelRegistry()
        # Use lazy initialization for crypto data fetchers to avoid unnecessary CCXT connections
        self.consistent_fetcher = (
            None  # Will be initialized only when fetching crypto data
        )
        self.hybrid_fetcher = None  # Will be initialized only when fetching crypto data
        self._consistent_fetcher_available = CONSISTENT_FETCHER_AVAILABLE
        self._hybrid_fetcher_available = HYBRID_FETCHER_AVAILABLE

    def load_and_preprocess_data(
        self, config: BacktestConfig, save_intermediates: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load and preprocess data with optimized caching."""

        # Load from file if provided
        if config.data_path:
            try:
                if str(config.data_path).endswith(".parquet"):
                    df = pd.read_parquet(config.data_path)
                    if df.index.name != "Date":
                        df.index.name = "Date"
                else:
                    df = pd.read_csv(
                        config.data_path,
                        skiprows=2,
                        index_col="Date",
                        parse_dates=["Date"],
                    )
                data = self._preprocess_dataframe(
                    df, config.symbol, ticker=config.ticker
                )
                data.attrs["ticker"] = Path(config.data_path).stem
                data.attrs["interval"] = config.interval
                data.attrs["annual_trading_periods"] = self._calculate_annual_periods(
                    config.interval, self._is_crypto(config.ticker or config.symbol)
                )
                return data
            except Exception as e:
                print(f"Error loading data from file: {e}")
                return None

        # Load from yfinance with caching (or hybrid for crypto)
        if not all([config.ticker, config.start_date, config.end_date]):
            print("Error: ticker, start, and end required if data_path not provided")
            return None

        # Check if we should use a specific data source or auto-detect
        use_crypto_fetcher = False

        # If force_data_source is specified, use that
        if config.force_data_source == "yfinance":
            use_crypto_fetcher = False
            print(f"Using yfinance data source for: {config.ticker}")
        elif config.force_data_source == "ccxt":
            if not self._is_crypto(config.ticker):
                print(f"Error: CCXT source only works with crypto tickers")
                return None
            use_crypto_fetcher = True
            print(f"Using CCXT/ByBit data source for: {config.ticker}")
        else:
            # Auto-detect based on ticker
            use_crypto_fetcher = self._is_crypto(config.ticker)

        # Check if this is a crypto ticker and use appropriate fetcher
        if use_crypto_fetcher:
            # Lazy initialization - only create CCXT connection when actually needed
            if self._consistent_fetcher_available and self.consistent_fetcher is None:
                try:
                    print(
                        f"Initializing crypto data fetcher (CCXT/ByBit) for {config.ticker}..."
                    )
                    self.consistent_fetcher = ConsistentDataFetcher(
                        exchange="bybit", testnet=False
                    )
                    print("✓ CCXT connection established")
                except Exception as e:
                    print(f"Warning: Failed to initialize CCXT: {e}")
                    print("Falling back to yfinance for crypto data...")
                    self._consistent_fetcher_available = False

            if self.consistent_fetcher:
                print(
                    f"Using consistent data fetcher (mainnet) for crypto ticker: {config.ticker}"
                )
                try:
                    # Convert ticker format (BTC-USD -> BTC/USDT)
                    symbol = config.ticker.replace("-USD", "/USDT").replace("-", "/")

                    # Calculate days based on date range
                    start_dt = pd.to_datetime(config.start_date)
                    end_dt = pd.to_datetime(config.end_date)
                    days = (end_dt - start_dt).days + 30  # Add buffer for indicators

                    # Fetch with UTC timezone consistency
                    df = self.consistent_fetcher.fetch_ohlcv_utc(
                        symbol=symbol,
                        timeframe="1d",
                        limit=min(days, 998),  # CCXT limit
                    )

                    if df.empty:
                        print(f"Warning: No data fetched from ByBit for {symbol}")
                        # Fallback to yfinance
                        print("Falling back to yfinance...")
                        df = self.data_loader.load_data(
                            config.ticker,
                            config.start_date,
                            config.end_date,
                            config.interval,
                        )
                    else:
                        # Filter to requested date range only if we have data
                        df_filtered = df.loc[config.start_date : config.end_date]
                        if df_filtered.empty:
                            print(
                                f"Warning: No data in requested range {config.start_date} to {config.end_date}"
                            )
                            print(f"Available range: {df.index[0]} to {df.index[-1]}")
                            # Use all available data
                            df_filtered = df
                        df = df_filtered

                        print(
                            f"Fetched {len(df)} daily candles from ByBit mainnet (UTC)"
                        )
                        if len(df) > 0:
                            print(f"Date range: {df.index[0]} to {df.index[-1]}")
                except Exception as e:
                    print(f"Error fetching data from CCXT: {e}")
                    print("Falling back to yfinance...")
                    df = self.data_loader.load_data(
                        config.ticker,
                        config.start_date,
                        config.end_date,
                        config.interval,
                    )

            elif self._hybrid_fetcher_available and self.hybrid_fetcher is None:
                # Lazy initialization for hybrid fetcher
                try:
                    print(f"Initializing hybrid data fetcher for {config.ticker}...")
                    self.hybrid_fetcher = HybridDataFetcher()
                except Exception as e:
                    print(f"Warning: Failed to initialize hybrid fetcher: {e}")
                    self._hybrid_fetcher_available = False

            if self.hybrid_fetcher:
                print(f"Using hybrid data fetcher for crypto ticker: {config.ticker}")
                df, metadata = self.hybrid_fetcher.fetch_hybrid_data(
                    config.ticker, config.start_date, config.end_date, overlap_days=30
                )

                # Log metadata
                if metadata.get("adjustment_factor", 1.0) != 1.0:
                    print(
                        f"Applied adjustment factor: {metadata['adjustment_factor']:.4f}"
                    )
                    print(
                        f"Data correlation: {metadata.get('overlap_correlation', 0):.4f}"
                    )
            else:
                # Fallback to yfinance for crypto if no specialized fetcher available
                print(
                    f"Warning: Using yfinance for crypto ticker {config.ticker} (may have price discrepancies)"
                )
                df = self.data_loader.load_data(
                    config.ticker, config.start_date, config.end_date, config.interval
                )
        else:
            # Use standard yfinance for stocks
            df = self.data_loader.load_data(
                config.ticker, config.start_date, config.end_date, config.interval
            )

        if df is None or df.empty:
            return None

        data = self._preprocess_dataframe(df, config.symbol, ticker=config.ticker)
        data.attrs["ticker"] = config.ticker
        data.attrs["interval"] = config.interval
        is_forex = self._is_forex(config.ticker or "")
        data.attrs["annual_trading_periods"] = self._calculate_annual_periods(
            config.interval, self._is_crypto(config.ticker), is_forex=is_forex
        )

        # Save fetched data for inspection with source labeling (skip in WFOV mode)
        if save_intermediates:
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)

            # Determine data source for filename
            if config.force_data_source:
                source_label = config.force_data_source
            elif use_crypto_fetcher and self.consistent_fetcher:
                source_label = "ccxt"
            elif use_crypto_fetcher and self.hybrid_fetcher:
                source_label = "hybrid"
            else:
                source_label = "yfinance"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.ticker}_{config.interval}_{timestamp}_{source_label}_backtest.csv"
            filepath = data_dir / filename

            # Save both raw and processed data
            # Standardize raw data column names to lowercase (matching crypto format)
            df_standardized = df.copy()

            # Handle MultiIndex columns (flatten if needed)
            if isinstance(df_standardized.columns, pd.MultiIndex):
                df_standardized.columns = df_standardized.columns.get_level_values(0)

            # Convert column names to lowercase
            df_standardized.columns = [
                col.lower() if isinstance(col, str) else str(col).lower()
                for col in df_standardized.columns
            ]

            raw_filepath = (
                data_dir
                / f"{config.ticker}_{config.interval}_{timestamp}_{source_label}_raw.csv"
            )
            df_standardized.to_csv(raw_filepath)
            data.to_csv(filepath)

            # Also save with feature engineering matching crypto trading format
            data_with_features = data.copy()
            # Add lag features to match crypto trading (using simple returns for lags)
            for i in range(1, 6):  # Default 5 lags
                data_with_features[f"returns_lag_{i}"] = data_with_features[
                    "returns"
                ].shift(i)
            # Rename price to close to match crypto format
            data_with_features["close"] = data_with_features["price"]
            data_with_features.dropna(inplace=True)

            features_filepath = (
                data_dir
                / f"{config.ticker}_{config.interval}_{timestamp}_{source_label}_features.csv"
            )
            data_with_features.to_csv(features_filepath)

            print(f"Saved raw data to: {raw_filepath}")
            print(f"Saved processed data to: {filepath}")
            print(f"Saved feature-engineered data to: {features_filepath}")

        return data

    def _preprocess_dataframe(
        self, df: pd.DataFrame, symbol: str, ticker: str = None
    ) -> pd.DataFrame:
        """Preprocess raw dataframe into required format."""
        # Handle MultiIndex columns (flatten if needed)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Handle adjusted close - use it if available, otherwise use Close
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        elif "adj close" in [
            col.lower() if isinstance(col, str) else str(col).lower()
            for col in df.columns
        ]:
            # Find the actual column name with case-insensitive search
            adj_close_col = [
                col for col in df.columns if str(col).lower() == "adj close"
            ][0]
            close_col = [col for col in df.columns if str(col).lower() == "close"][0]
            df[close_col] = df[adj_close_col]

        # Select price column - prefer Adj Close/Close over others
        price_col = None
        # Check for columns in order of preference
        for col in ["Adj Close", "Close", symbol, "Open"]:
            if col in df.columns:
                price_col = col
                break

        # If still not found, try case-insensitive search
        if price_col is None:
            for target in [
                "close",
                "adj close",
                symbol.lower() if symbol else "",
                "open",
            ]:
                for col in df.columns:
                    if str(col).lower() == target:
                        price_col = col
                        break
                if price_col:
                    break

        if price_col is None:
            raise ValueError(f"No valid price column found in data")

        # Create processed dataframe with OHLCV for feature engineering
        data = pd.DataFrame(index=df.index)

        # Preserve OHLCV columns with lowercase names for feature engine
        ohlcv_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        for raw_col, std_col in ohlcv_mapping.items():
            if raw_col in df.columns:
                data[std_col] = pd.to_numeric(df[raw_col], errors="coerce")

        data["price"] = pd.to_numeric(df[price_col], errors="coerce")
        data = data.dropna(subset=["price"])

        # Strip phantom non-trading rows (yfinance artifact for Sun-Thu exchanges)
        # Phantom rows: volume=0 AND open=high=low=close (forward-filled holidays)
        # Activated by STRIP_PHANTOM_ROWS=1 env var (set by model_selection_workflow)
        if os.environ.get("STRIP_PHANTOM_ROWS") == "1":
            _has_ohlcv = all(
                c in data.columns for c in ("open", "high", "low", "close", "volume")
            )
            if _has_ohlcv:
                _phantom = (
                    (data["volume"] == 0)
                    & (data["open"] == data["high"])
                    & (data["high"] == data["low"])
                    & (data["low"] == data["close"])
                )
                _n_phantom = int(_phantom.sum())
                if _n_phantom > 0:
                    print(
                        f"  Stripped {_n_phantom} phantom non-trading rows (volume=0, O=H=L=C)"
                    )
                    data = data[~_phantom].copy()

        # Calculate returns and direction (using log returns as requested)
        data["returns"] = np.log(
            data["price"] / data["price"].shift(1)
        )  # Log returns (primary)
        data["simple_returns"] = data[
            "price"
        ].pct_change()  # Simple returns (kept for reference)

        # Carry-adjusted direction for forex tickers, raw direction for stocks/crypto
        if self._is_forex(ticker or ""):
            from algos.common.data_loader import (
                compute_carry_differential,
                compute_direction,
            )

            carry_diff = compute_carry_differential(data.index)
            data["direction"] = compute_direction(data["returns"], ticker, carry_diff)
            data["carry_differential"] = carry_diff.reindex(data.index, method="ffill")
        else:
            data["direction"] = np.where(data["returns"] > 0, 1, -1)

        data = data.dropna(subset=["returns"])

        return data

    def _is_forex(self, ticker: str) -> bool:
        """Check if ticker is a forex pair."""
        try:
            from algos.common.data_loader import _is_forex_ticker

            return _is_forex_ticker(ticker)
        except ImportError:
            # Minimal fallback
            clean = ticker.replace("=X", "").replace("/", "")
            return len(clean) == 6 and clean.isalpha() and clean.isupper()

    def _is_crypto(self, ticker: str) -> bool:
        """Check if ticker is cryptocurrency."""
        crypto_suffixes = ["-USD", "-USDT", "-BTC", "-ETH"]
        return any(ticker.upper().endswith(s) for s in crypto_suffixes)

    def _calculate_annual_periods(
        self, interval: str, is_crypto: bool, is_forex: bool = False
    ) -> int:
        """Calculate annual trading periods based on interval."""
        import re

        match = re.match(r"(\d+)([a-zA-Z]+)", interval)
        if not match:
            return 252  # Default to daily

        value = int(match.group(1))
        unit = match.group(2).lower()

        if is_crypto:
            periods = {"d": 365, "wk": 52, "mo": 12, "h": 365 * 24, "m": 365 * 24 * 60}
        elif is_forex:
            periods = {"d": 260, "wk": 52, "mo": 12, "h": 260 * 24, "m": 260 * 24 * 60}
        else:
            periods = {
                "d": 252,
                "wk": 52,
                "mo": 12,
                "h": 252 * 6.5,
                "m": 252 * 6.5 * 60,
            }

        return periods.get(unit, 252) // value

    def run_single_backtest(
        self, config: BacktestConfig, pre_loaded_data: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[pd.DataFrame], Any]:
        """
        Run a single backtest with the given configuration.

        Args:
            config: Backtest configuration
            pre_loaded_data: Optional pre-loaded DataFrame (for WFOV optimization)
                            If provided, skips data loading and uses this data

        Returns:
            Tuple of (test_results, final_model)
        """

        # Generate log prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = (
            Path(config.data_path).stem
            if config.data_path
            else f"{config.ticker}_{config.interval}_{config.start_date}_{config.end_date}"
        )
        log_prefix = f"{config.model_name}_{data_id}_{timestamp}"

        # Configure global settings
        from algos.common import config as global_config

        global_config.PTC = config.ptc
        global_config.RF_RATE = config.rf_rate

        # In lightweight mode (WFOV iterations), suppress ALL file log writes
        # This flag is checked by RedirectStdoutToFile in utils.py and persistence.py,
        # preventing the 140+ log file writes across 22 model files.
        if config.skip_model_save:
            from algos.common.utils import set_suppress_file_logs

            set_suppress_file_logs(True)

        log_context = (
            suppress_stdout()
            if config.skip_model_save
            else RedirectStdoutToFile(
                f"{self.logs_dir}/{log_prefix}_overall_output.txt"
            )
        )

        try:
            with log_context:
                print(f"Starting backtest for {config.model_name} on {data_id}")
                print(
                    f"Parameters: train_split={config.train_split:.1%}, rf_rate={config.rf_rate:.1%}, ptc={config.ptc:.4%}"
                )

                # Set seed for reproducibility
                if config.iteration_seed is not None:
                    seed = set_seed(
                        model_name=config.model_name,
                        ticker=config.ticker,
                        seed=config.iteration_seed,
                    )
                else:
                    seed = set_seed(model_name=config.model_name, ticker=config.ticker)
                print(
                    f"Random seed set to {seed} for {config.model_name} on {config.ticker}"
                )

                # Save seed info for reproducibility (skip in WFOV mode)
                if not config.skip_model_save:
                    seed_info_path = (
                        self.models_dir
                        / f"seed_info_{config.model_name}_{config.ticker}_{timestamp}.json"
                    )
                    save_seed_info(
                        seed_info_path,
                        config.model_name,
                        config.ticker or "N/A",
                        backtest_config=vars(config),
                        timestamp=timestamp,
                    )

                # Load or use pre-loaded data
                start_time = time.time()
                if pre_loaded_data is not None:
                    # Use pre-loaded data (WFOV optimization - already preprocessed)
                    data = pre_loaded_data.copy()
                    # Preserve existing attrs or add if missing
                    if "ticker" not in data.attrs:
                        data.attrs["ticker"] = config.ticker
                    if "interval" not in data.attrs:
                        data.attrs["interval"] = config.interval
                    if "annual_trading_periods" not in data.attrs:
                        data.attrs["annual_trading_periods"] = (
                            self._calculate_annual_periods(
                                config.interval, self._is_crypto(config.ticker)
                            )
                        )
                    print(f"Using pre-loaded data ({len(data)} rows, pre-processed)")
                else:
                    # Load data normally (skip intermediate CSV saves in WFOV mode)
                    data = self.load_and_preprocess_data(
                        config, save_intermediates=config.save_intermediates
                    )
                    if data is None:
                        print("Failed to load data")
                        return None, None
                    print(f"Data loaded in {time.time() - start_time:.2f}s")

                # Get and run model
                start_time = time.time()
                model_func = self.model_registry.get_model(config.model_name)

                # Check if model function accepts 'interval' parameter
                func_signature = inspect.signature(model_func)
                func_params = func_signature.parameters
                accepts_interval = "interval" in func_params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in func_params.values()
                )

                # Build call arguments
                call_kwargs = {
                    "data": data.copy(),
                    "initial_train_split_ratio": config.train_split,
                    "log_prefix": log_prefix,
                    "embargo_pct": config.embargo_pct,  # Lopez de Prado embargo parameter
                    **config.model_params,
                }

                # Pass date-based split parameters if available (WFOV walk-forward modes)
                if config.train_end_date is not None:
                    call_kwargs["train_end_date"] = config.train_end_date
                if config.test_start_date is not None:
                    call_kwargs["test_start_date"] = config.test_start_date

                # Only pass interval if function accepts it
                if accepts_interval:
                    call_kwargs["interval"] = config.interval or "1d"

                # Load and pass feature engineering config if available
                # This flows through to BaseStrategyModel.run_strategy() via **kwargs
                accepts_feature_config = "feature_config" in func_params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in func_params.values()
                )
                if accepts_feature_config:
                    try:
                        from algos.common.feature_engine import FeatureConfig
                        from algos.common.external_data import fetch_multiple_external

                        fc = FeatureConfig(
                            model_name=config.model_name, ticker=config.ticker
                        )
                        if fc.indicators:
                            call_kwargs["feature_config"] = fc
                            print(
                                f"Feature engineering enabled: {len(fc.indicators)} indicator groups, "
                                f"config hash={fc.config_hash}"
                            )

                            # Pre-fetch external data if configured
                            if (
                                fc.external_configs
                                and data is not None
                                and not data.empty
                            ):
                                start_str = data.index.min().strftime("%Y-%m-%d")
                                end_str = data.index.max().strftime("%Y-%m-%d")
                                ext_data = fetch_multiple_external(
                                    fc.external_configs,
                                    start=start_str,
                                    end=end_str,
                                    primary_index=data.index,
                                )
                                if ext_data:
                                    call_kwargs["external_data"] = ext_data
                                    print(
                                        f"External data loaded: {list(ext_data.keys())}"
                                    )
                    except ImportError:
                        pass  # Feature engine not installed
                    except Exception as e:
                        print(
                            f"Warning: Feature engine init failed: {e}. Using legacy features."
                        )

                test_results, final_model = model_func(**call_kwargs)
                print(f"Model trained in {time.time() - start_time:.2f}s")

                if test_results is None:
                    return None, None

                # Save model predictions (position column) if available (skip in WFOV mode)
                if "position" in test_results.columns and config.save_intermediates:
                    # Use the algos/data directory for predictions
                    # Handle both module (-m) and direct script execution
                    current_file = Path(__file__).resolve()
                    project_root = current_file.parents[2]  # Go up to project root
                    pred_data_dir = project_root / "algos" / "data"
                    pred_data_dir.mkdir(parents=True, exist_ok=True)
                    # Determine source label based on data source
                    if config.data_path:
                        pred_source_label = "file"
                    elif config.force_data_source == "ccxt":
                        pred_source_label = "ccxt"
                    else:
                        pred_source_label = "yfinance"
                    predictions_filepath = (
                        pred_data_dir
                        / f"{config.ticker}_{config.interval}_{timestamp}_{pred_source_label}_predictions.csv"
                    )
                    # Create a DataFrame with dates and predictions
                    predictions_df = pd.DataFrame(
                        {
                            "Date": test_results.index,
                            "position": test_results["position"],
                            "returns": test_results["returns"],
                            "direction": test_results.get("direction", np.nan),
                            "strategy": test_results.get("strategy", np.nan),
                        }
                    )
                    predictions_df.to_csv(predictions_filepath, index=False)
                    print(f"✅ Saved model predictions to: {predictions_filepath}")
                    print(f"   - Total predictions: {len(predictions_df)}")
                    print(
                        f"   - Buy signals: {(predictions_df['position'] == 1).sum()}"
                    )
                    print(
                        f"   - Sell signals: {(predictions_df['position'] == -1).sum()}"
                    )

                # Transfer attributes
                test_results.attrs["annual_trading_periods"] = data.attrs[
                    "annual_trading_periods"
                ]

                # Calculate metrics (skip plots if no_plots=True)
                performance_metrics = calculate_strategy_performance(
                    test_results,
                    config.model_name,
                    log_prefix,
                    max_leverage=config.max_leverage,
                    no_plots=config.no_plots,
                )

                risk_metrics = calculate_risk_metrics(
                    test_results, config.model_name, log_prefix
                )

                if not config.no_plots:
                    plot_drawdowns(test_results, config.model_name, log_prefix)

                # Save model (skip in WFOV iteration mode to avoid 1000s of throwaway files)
                if not config.skip_model_save:
                    save_model.save_model(
                        model_obj=final_model,
                        model_name=config.model_name,
                        ticker=config.ticker or "N/A",
                        symbol=config.symbol,
                        start=config.start_date or "N/A",
                        end=config.end_date or "N/A",
                        interval=config.interval or "N/A",
                        timestamp=timestamp,
                    )

                    # Save ARIMA-specific settings if applicable
                    if config.model_name == "arima":
                        # Use the parameters passed from command line (they are required for ARIMA)
                        arima_settings = {
                            "signal_method": config.model_params.get(
                                "signal_method", "z_score"
                            ),
                            "threshold": config.model_params.get("threshold"),
                            "z_score_threshold": config.model_params.get(
                                "z_score_threshold"
                            ),
                            "lookback_window": config.model_params.get(
                                "lookback_window"
                            ),
                            "order": (1, 0, 1),  # Default ARIMA order
                            "ticker": config.ticker or "N/A",
                            "timestamp": timestamp,
                        }
                        save_arima_settings(
                            ticker=config.ticker or "N/A",
                            settings=arima_settings,
                            timestamp=timestamp,
                        )
                        print(f"✅ Saved ARIMA settings for {config.ticker}")

                    # Ensure scaler is saved if auto-save is enabled
                    if SCALER_AUTOSAVE_ENABLED:
                        ensure_scaler_saved(
                            config.model_name, config.ticker or "N/A", timestamp
                        )

                    # Save feature metadata if feature engineering was used
                    if (
                        "feature_config" in call_kwargs
                        and hasattr(final_model, "_feature_cols") is False
                    ):
                        # Try to get feature info from the model instance
                        try:
                            from algos.common.feature_engine import (
                                save_feature_metadata,
                            )

                            fc = call_kwargs["feature_config"]
                            # Get feature names from the engine (approximate)
                            from algos.common.feature_engine import FeatureEngine

                            engine = FeatureEngine()
                            feat_names = engine.get_feature_names(fc)
                            if feat_names:
                                scalers_dir = (
                                    Path(__file__).resolve().parents[1]
                                    / "algos"
                                    / "scalers"
                                )
                                save_feature_metadata(
                                    feature_names=feat_names,
                                    feature_config=fc,
                                    symbol=config.ticker or "N/A",
                                    model_type=config.model_name,
                                    save_dir=str(scalers_dir),
                                )
                        except Exception as e:
                            print(f"Warning: Could not save feature metadata: {e}")

                print("\nBacktest Complete!")
                return test_results, final_model
        finally:
            # Always reset the suppress flag so subsequent calls in the same process
            # (e.g., Step 1 training after WFOV) write logs normally.
            if config.skip_model_save:
                from algos.common.utils import set_suppress_file_logs

                set_suppress_file_logs(False)

    def run_multiple_backtests(self, configs: list, max_workers: int = 4) -> list:
        """Run multiple backtests in parallel."""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_single_backtest, config): config
                for config in configs
            }

            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                try:
                    test_results, model = future.result()
                    results.append(
                        {"config": config, "results": test_results, "model": model}
                    )
                except Exception as e:
                    print(f"Backtest failed for {config.model_name}: {e}")
                    results.append(
                        {
                            "config": config,
                            "results": None,
                            "model": None,
                            "error": str(e),
                        }
                    )

        return results


def main():
    """Main entry point for optimized backtesting."""
    parser = argparse.ArgumentParser(
        description="Run optimized backtests for trading strategies"
    )

    # Model selection
    registry = ModelRegistry()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=registry.list_models(),
        help="Model to run",
    )

    # Data source
    parser.add_argument("--data_path", type=str, help="Path to CSV data file")
    parser.add_argument("--ticker", type=str, help="Ticker symbol")
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Omit if using --lookback_days",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today if using --lookback_days",
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        help="Number of days to look back from end date (e.g., 1260 for ~5 years). If provided, --start is calculated automatically",
    )
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")

    # Parameters
    parser.add_argument(
        "--train_split", type=float, default=0.5, help="Training split ratio"
    )
    parser.add_argument("--rf_rate", type=float, default=0.04, help="Risk-free rate")
    parser.add_argument("--ptc", type=float, default=0.00035, help="Per-trade cost")
    parser.add_argument("--symbol", type=str, default="Adj Close", help="Price column")
    parser.add_argument(
        "--source",
        type=str,
        default="yfinance",
        choices=["yfinance", "ccxt"],
        help="Data source: yfinance (default) or ccxt (crypto only, ~1000 days limit)",
    )
    parser.add_argument(
        "--max_leverage",
        type=float,
        default=4.0,
        help="Maximum leverage cap for position sizing (default: 4x)",
    )
    parser.add_argument(
        "--embargo_pct",
        type=float,
        default=0.02,
        help="Embargo percentage for train/test split (default: 0.02 = 2%%). "
        "Set to 0 to disable embargo. Lopez de Prado recommends 0.01-0.05 "
        "depending on asset class and holding period.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots/images (useful for batch runs)",
    )
    parser.add_argument(
        "--no-save-intermediates",
        action="store_true",
        help="Skip saving intermediate CSV files (raw, processed, features, predictions)",
    )
    parser.add_argument(
        "--skip-model-save",
        action="store_true",
        help="Skip saving model, scaler, and seed_info files (for WFOV iterations)",
    )

    # Model-specific parameters
    parser.add_argument("--dqn_epochs", type=int, default=50)
    parser.add_argument("--dqn_batch_size", type=int, default=32)
    parser.add_argument("--dqn_hidden_units", type=int, default=64)

    # ARIMA-specific parameters (required when model is ARIMA)
    parser.add_argument(
        "--arima_threshold",
        type=float,
        help="ARIMA threshold for threshold method (required for ARIMA)",
    )
    parser.add_argument(
        "--arima_zscore",
        type=float,
        help="ARIMA z-score threshold (required for ARIMA)",
    )
    parser.add_argument(
        "--arima_lookback",
        type=int,
        help="ARIMA lookback window in days (required for ARIMA)",
    )
    parser.add_argument(
        "--arima_signal_method",
        type=str,
        default="z_score",
        choices=["z_score", "threshold", "percentile", "simple"],
        help="ARIMA signal generation method (default: z_score)",
    )

    args = parser.parse_args()

    # Calculate dates from lookback_days if provided
    if args.lookback_days is not None:
        if args.start is not None:
            print("Warning: --start is ignored when --lookback_days is provided")

        # Get end date (use provided end_date or default to today)
        if args.end is None:
            end_dt = datetime.now()
            args.end = end_dt.strftime("%Y-%m-%d")
            print(f"Using current date as end date: {args.end}")
        else:
            # Parse and regularize the end_date using dateutil
            try:
                end_dt = dateutil_parser.parse(args.end)
                args.end = end_dt.strftime("%Y-%m-%d")
                print(f"Parsed and regularized end date: {args.end}")
            except Exception as e:
                parser.error(f"Failed to parse end date '{args.end}': {e}")

        # Calculate start_date
        start_dt = end_dt - timedelta(days=args.lookback_days)
        args.start = start_dt.strftime("%Y-%m-%d")
        print(
            f"Calculated start date from lookback_days={args.lookback_days}: {args.start}"
        )

    # Regularize dates using dateutil if they were provided directly (and not calculated from lookback)
    elif args.start is not None and args.end is not None:
        try:
            start_dt = dateutil_parser.parse(args.start)
            args.start = start_dt.strftime("%Y-%m-%d")
            end_dt = dateutil_parser.parse(args.end)
            args.end = end_dt.strftime("%Y-%m-%d")
            print(f"Regularized dates - start: {args.start}, end: {args.end}")
        except Exception as e:
            parser.error(f"Failed to parse dates: {e}")

    # Validate arguments
    if not args.data_path:
        if not args.ticker:
            parser.error("--ticker is required when not using --data_path")
        if args.lookback_days is None and not all([args.start, args.end]):
            parser.error(
                "Either --lookback_days or (--start and --end) required when not using --data_path"
            )

    # Validate ARIMA-specific parameters
    if args.model_name == "arima":
        missing_params = []
        if args.arima_threshold is None:
            missing_params.append("--arima_threshold")
        if args.arima_zscore is None:
            missing_params.append("--arima_zscore")
        if args.arima_lookback is None:
            missing_params.append("--arima_lookback")

        if missing_params:
            print("\n❌ ERROR: ARIMA model requires the following parameters:")
            print(f"   Missing: {', '.join(missing_params)}")
            print("\nExample usage:")
            print(
                "  python run_backtest_optimized.py --model_name arima --ticker BTC-USD \\"
            )
            print("    --arima_threshold 0.0002 --arima_zscore 1.0 --arima_lookback 5")
            print("\nParameters explanation:")
            print(
                "  --arima_threshold: Threshold for simple threshold method (e.g., 0.0002)"
            )
            print("  --arima_zscore: Z-score threshold for z_score method (e.g., 1.0)")
            print(
                "  --arima_lookback: Lookback window in days for rolling statistics (e.g., 5)"
            )
            print(
                "  --arima_signal_method: Signal generation method (default: z_score)"
            )
            sys.exit(1)

    # Build model parameters
    model_params = {}
    if args.model_name == "dqn":
        model_params = {
            "dqn_epochs": args.dqn_epochs,
            "dqn_batch_size": args.dqn_batch_size,
            "dqn_hidden_units": args.dqn_hidden_units,
        }
    elif args.model_name == "arima":
        model_params = {
            "signal_method": args.arima_signal_method,
            "threshold": args.arima_threshold,
            "z_score_threshold": args.arima_zscore,
            "lookback_window": args.arima_lookback,
        }
    elif "xgb" in args.model_name or "xgboost" in args.model_name:
        # XGBoost parameters - auto-detect GPU (Apple Silicon Metal or NVIDIA CUDA)
        model_params = {
            "use_gpu": "auto",  # Auto-detect GPU availability (Apple Silicon / NVIDIA)
            "auto_tune": False,  # Set to False by default for speed
            "n_jobs": -1,
        }

    # Validate source option for crypto
    if (
        args.source == "ccxt"
        and args.ticker
        and not any(
            crypto in args.ticker.upper()
            for crypto in ["BTC", "ETH", "SOL", "DOGE", "COIN", "CRYPTO"]
        )
    ):
        print(f"Error: --source ccxt only works with crypto tickers, not {args.ticker}")
        sys.exit(1)

    # Create configuration
    config = build_backtest_config(args, model_params=model_params)

    # Run backtest
    backtester = OptimizedBacktester()
    results, model = backtester.run_single_backtest(config)

    if results is not None:
        print("\nBacktest completed successfully!")
        # Check if predictions were saved (check the log file for the message)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = (
            f"{config.ticker}_{config.interval}_{config.start_date}_{config.end_date}"
        )
        log_file = Path(
            f"algos/backtest_code/logs/{config.model_name}_{data_id}_{timestamp}_overall_output.txt"
        )
        # Try to find the most recent log file if exact match doesn't exist
        import glob

        log_pattern = f"algos/backtest_code/logs/{config.model_name}_{config.ticker}_*_overall_output.txt"
        log_files = sorted(glob.glob(log_pattern))
        if log_files:
            with open(log_files[-1], "r") as f:
                content = f.read()
                if "Saved model predictions to:" in content:
                    # Extract the predictions file path
                    for line in content.split("\n"):
                        if "Saved model predictions to:" in line:
                            print(f"\n✅ {line.strip()}")
                        elif (
                            "Total predictions:" in line
                            or "Buy signals:" in line
                            or "Sell signals:" in line
                        ):
                            print(f"   {line.strip()}")
    else:
        print("\nBacktest failed!")
        sys.exit(1)


def build_backtest_config(args, *, model_params: dict) -> BacktestConfig:
    """Build BacktestConfig from CLI args without dropping data_path metadata.

    File-backed runs still need ticker/interval for seed naming, feature config,
    forex/carry direction handling, annual period calculation, model dumps, and
    retrain/deploy observability. Dropping them made every --data_path retrain
    fail before training.
    """
    return BacktestConfig(
        model_name=args.model_name,
        ticker=args.ticker,
        start_date=args.start if not args.data_path else None,
        end_date=args.end if not args.data_path else None,
        interval=args.interval,
        data_path=args.data_path,
        train_split=args.train_split,
        rf_rate=args.rf_rate,
        ptc=args.ptc,
        symbol=args.symbol,
        model_params=model_params,
        force_data_source=args.source,  # Use the source argument
        max_leverage=args.max_leverage,  # Pass the max leverage cap
        embargo_pct=args.embargo_pct,  # Lopez de Prado embargo percentage
        no_plots=args.no_plots,  # Skip plot generation if True
        save_intermediates=not args.no_save_intermediates,  # Skip CSV saves when flagged
        skip_model_save=args.skip_model_save,  # Skip model/scaler saves when flagged
    )


if __name__ == "__main__":
    main()
