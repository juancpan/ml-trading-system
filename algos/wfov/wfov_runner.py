"""
Walk-Forward Out-of-Sample Validation (WFOV) Runner.

Main orchestrator for running multiple randomized backtests and aggregating results.
Validates model robustness across diverse market conditions.

Author: jcp
Date: 2025-12-02
"""

import numpy as np
import pandas as pd
import sys
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print(
        "Note: tqdm not available. Install with 'pip install tqdm' for progress bars."
    )

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from algos.backtest_code.run_backtest_optimized import (
    OptimizedBacktester,
    BacktestConfig,
)
from algos.wfov.window_generator import (
    generate_random_windows,
    generate_iteration_seed,
    generate_walk_forward_expanding_windows,
    generate_walk_forward_rolling_windows,
    validate_walk_forward_windows,
)
from algos.wfov.metrics_aggregator import extract_all_metrics
from algos.wfov.results_formatter import (
    save_iteration_to_csv,
    save_iterations_batch_to_csv,
    generate_summary_statistics,
    save_summary_json,
    format_console_summary,
    generate_wfov_filename_base,
)
from algos.common.embargo_utils import calculate_embargo_size
from algos.wfov.regime_analyzer import detect_market_regimes, assign_window_regime
from algos.wfov.statistical_tests import compute_all_statistical_tests


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout during backtest execution."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def suggest_max_workers(model_name: str, default: int = 4) -> int:
    """
    Suggest optimal max_workers based on model type to avoid CPU oversubscription.

    sklearn models with n_jobs=-1 already use all cores internally, so running
    multiple in parallel causes thrashing. TensorFlow/Keras models are more
    single-threaded and benefit from more parallel workers.

    Args:
        model_name: Model name string
        default: Default if model type is unknown

    Returns:
        Recommended max_workers count
    """
    import os

    cpu_count = os.cpu_count() or 4

    # Models that use n_jobs=-1 internally (saturate all cores per instance)
    sklearn_heavy = {
        "xgb_optimized",
        "xgboost_optimized",
        "rf_optimized",
        "random_forest_optimized",
        "ensemble_optimized",
        "ensemble_oof",
        "ensemble_voting",
        "ensemble_stacking",
        "ensemble_adaptive",
        "svm_optimized",
        "svm_tuned",
        "stacking",
    }
    # Lightweight models (single-threaded or limited parallelism)
    lightweight = {
        "lstm",
        "lstm_optimized",
        "cnn",
        "dnn",
        "dqn",
        "tcn",
        "li_reg",
        "linear_optimized",
        "logistic_optimized",
        "sgd_optimized",
        "gnb",
        "arima",
        "sarimax",
        "var",
        "kmeans",
        "log_reg",
    }

    if model_name in sklearn_heavy:
        # These use all cores internally; 2 concurrent is usually optimal
        return min(2, cpu_count)
    elif model_name in lightweight:
        # These are mostly single-threaded; use more workers
        return min(max(cpu_count - 1, 2), 8)
    else:
        return min(default, cpu_count)


class WFOVRunner:
    """
    Walk-Forward Out-of-Sample Validation orchestrator.

    Runs multiple backtests with randomized parameters to validate
    model robustness and minimize overfitting bias.
    """

    def __init__(self, master_seed: Optional[int] = None):
        """
        Initialize WFOV runner.

        Args:
            master_seed: Master seed for reproducibility. If None, generates random seed.
        """
        self.master_seed = (
            master_seed if master_seed is not None else np.random.randint(0, 2**31)
        )
        self.backtester = OptimizedBacktester()
        logger.info(f"WFOVRunner initialized with master_seed={self.master_seed}")

    def _calculate_spanning_window(self, windows: List[Dict]) -> Tuple[str, str]:
        """
        Calculate the minimum spanning window across all iterations.

        Args:
            windows: List of window dicts with 'start_date' and 'end_date'

        Returns:
            Tuple of (earliest_start_date, latest_end_date) as strings
        """
        start_dates = [pd.to_datetime(w["start_date"]) for w in windows]
        end_dates = [pd.to_datetime(w.get("test_end", w["end_date"])) for w in windows]

        earliest_start = min(start_dates)
        latest_end = max(end_dates)

        return earliest_start.strftime("%Y-%m-%d"), latest_end.strftime("%Y-%m-%d")

    def _preload_data(
        self,
        ticker: str,
        start: str,
        end: str,
        symbol: str = "Adj Close",
        interval: str = "1d",
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Pre-load and preprocess historical data for the full spanning window.
        Also detects market regimes for regime-aware analysis.

        Args:
            ticker: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Price column symbol (default: 'Adj Close')
            interval: Data interval (default: '1d')

        Returns:
            Tuple of (preprocessed_data, regimes)
            - preprocessed_data: DataFrame with full historical data, or None if failed
            - regimes: Series with regime classifications, or None if failed

        Example:
            >>> data, regimes = runner._preload_data('SPY', '2020-01-01', '2025-01-01')
            >>> regimes.value_counts()
            normal      150
            high_vol     80
            low_vol      70
        """
        print(f"\nPre-loading data for {ticker} ({start} to {end})...")

        try:
            # Load raw data
            raw_data = self.backtester.data_loader.load_data(
                ticker=ticker, start=start, end=end, interval=interval, use_cache=True
            )

            if raw_data is None or raw_data.empty:
                logger.error(f"Failed to pre-load data for {ticker}")
                return None, None

            # Preprocess the data (same as load_and_preprocess_data does)
            preprocessed_data = self.backtester._preprocess_dataframe(raw_data, symbol)

            # Calculate annual trading periods based on interval
            annual_periods_map = {
                "1d": 252,  # Daily (stocks)
                "1wk": 52,  # Weekly
                "1mo": 12,  # Monthly
                "1h": 252 * 7,  # Hourly (approximate)
            }
            annual_trading_periods = annual_periods_map.get(interval, 252)

            # Add required metadata attributes
            preprocessed_data.attrs["ticker"] = ticker
            preprocessed_data.attrs["interval"] = interval
            preprocessed_data.attrs["annual_trading_periods"] = annual_trading_periods

            print(f"✓ Pre-loaded and preprocessed {len(preprocessed_data)} rows")

            # Detect market regimes
            try:
                print(f"Detecting market regimes...")
                regimes = detect_market_regimes(
                    preprocessed_data, method="volatility_quantile"
                )
                regime_counts = regimes.value_counts()
                print(f"✓ Regimes detected: {dict(regime_counts)}")
            except Exception as e:
                logger.warning(f"Failed to detect regimes: {e}")
                regimes = None

            return preprocessed_data, regimes

        except Exception as e:
            logger.error(f"Error pre-loading data: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None, None

    def run_single_iteration(
        self,
        iteration: int,
        window: Dict,
        model_name: str,
        ticker: str,
        train_split: float,
        embargo_pct: float,
        rf_rate: float,
        ptc: float,
        max_leverage: float,
        model_params: Dict,
        pre_loaded_data: Optional[pd.DataFrame] = None,
        regimes: Optional[pd.Series] = None,
        interval: str = "1d",
        no_plots: bool = False,
    ) -> Dict:
        """
        Run a single WFOV iteration.

        Args:
            iteration: Iteration number
            window: Window dict with start_date, end_date, lookback_days, iteration_seed
            model_name: Model to run
            ticker: Ticker symbol
            train_split: Train/test split ratio
            embargo_pct: Embargo percentage
            rf_rate: Risk-free rate
            ptc: Per-trade cost
            max_leverage: Maximum leverage cap
            model_params: Model-specific parameters
            pre_loaded_data: Optional pre-loaded DataFrame (avoids redundant downloads)
            regimes: Optional regime classifications (for regime-aware analysis)

        Returns:
            Dict with:
            {
                'status': 'success' or 'failed',
                'iteration': int,
                'window': Dict,
                'config': Dict,
                'metrics': Dict (if successful),
                'regime': str (if regimes provided),
                'error': str (if failed)
            }
        """
        try:
            # Determine window regime if regimes provided
            window_regime = "unknown"
            if regimes is not None:
                try:
                    window_regime = assign_window_regime(
                        window["start_date"], window["end_date"], regimes, threshold=0.6
                    )
                except Exception as e:
                    logger.warning(
                        f"Iteration {iteration}: Failed to assign regime: {e}"
                    )
                    window_regime = "unknown"

            # Slice pre-loaded data to window if provided
            sliced_data = None
            if pre_loaded_data is not None:
                try:
                    # Include test period data if available (walk-forward modes)
                    slice_end = window.get("test_end", window["end_date"])
                    sliced_data = pre_loaded_data.loc[
                        window["start_date"] : slice_end
                    ].copy()
                    if sliced_data.empty:
                        logger.warning(
                            f"Iteration {iteration}: Sliced data is empty for window {window['start_date']} to {window['end_date']}"
                        )
                        sliced_data = None
                    else:
                        # Preserve metadata attributes from pre-loaded data
                        sliced_data.attrs = pre_loaded_data.attrs.copy()
                except Exception as e:
                    logger.warning(
                        f"Iteration {iteration}: Failed to slice pre-loaded data: {e}. Will download fresh data."
                    )
                    sliced_data = None

            # Compute canonical B&H returns (model-independent test window)
            # This ensures all models in the same iteration are benchmarked
            # against the identical buy-and-hold period.
            canonical_bh_returns = None
            if sliced_data is not None and "returns" in sliced_data.columns:
                try:
                    if window.get("test_start"):
                        # Walk-forward mode: use explicit date boundaries
                        canonical_bh_returns = sliced_data["returns"].loc[
                            window["test_start"] :
                        ]
                    else:
                        # Ratio-based mode: replicate split + embargo on raw data
                        raw_len = len(sliced_data)
                        initial_split = int(raw_len * train_split)
                        embargo_size = calculate_embargo_size(
                            raw_len, embargo_pct=embargo_pct, interval=interval
                        )
                        canonical_bh_returns = sliced_data["returns"].iloc[
                            initial_split + embargo_size :
                        ]
                    # Drop NaN from first row (log returns shift)
                    if canonical_bh_returns is not None:
                        canonical_bh_returns = canonical_bh_returns.dropna()
                except Exception:
                    canonical_bh_returns = None  # Fall back to model's test window

            # Create config for this iteration
            # WFOV iterations skip all file I/O (no model dumps, no intermediate CSVs)
            config = BacktestConfig(
                model_name=model_name,
                ticker=ticker,
                start_date=window["start_date"],
                end_date=window.get("test_end", window["end_date"]),
                interval=interval,
                data_path=None,
                train_split=train_split,
                rf_rate=rf_rate,
                ptc=ptc,
                symbol="Adj Close",
                model_params=model_params,
                force_data_source="yfinance",
                max_leverage=max_leverage,
                embargo_pct=embargo_pct,
                no_plots=no_plots,
                save_intermediates=False,  # Skip CSV writes (raw, processed, features, predictions)
                skip_model_save=True,  # Skip model/scaler/seed_info saves
                train_end_date=window.get("end_date"),
                test_start_date=window.get("test_start"),
                iteration_seed=window["iteration_seed"],
            )

            # Suppress verbose backtest output to keep progress bar clean
            with suppress_stdout():
                test_results, final_model = self.backtester.run_single_backtest(
                    config, pre_loaded_data=sliced_data
                )

            if test_results is None or test_results.empty:
                return {
                    "status": "failed",
                    "iteration": iteration,
                    "window": window,
                    "config": vars(config),
                    "regime": window_regime,
                    "error": "Backtest returned None or empty DataFrame",
                }

            # Extract metrics
            log_prefix = f"wfov_iter_{iteration}"
            metrics = extract_all_metrics(
                test_results,
                model_name,
                log_prefix,
                max_leverage,
                no_plots=no_plots,
                bh_returns=canonical_bh_returns,
            )

            return {
                "status": "success",
                "iteration": iteration,
                "window": window,
                "config": vars(config),
                "metrics": metrics,
                "regime": window_regime,
            }

        except Exception as e:
            import traceback

            return {
                "status": "failed",
                "iteration": iteration,
                "window": window,
                "config": vars(config) if "config" in locals() else {},
                "regime": window_regime if "window_regime" in locals() else "unknown",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def run_wfov_session(
        self,
        model_name: str,
        ticker: str,
        start_date: str,
        end_date: str,
        validation_mode: str = "monte_carlo",
        iterations: Optional[int] = None,
        min_lookback_days: Optional[int] = None,
        max_lookback_days: Optional[int] = None,
        initial_train_days: Optional[int] = None,
        window_size: Optional[int] = None,
        test_days: Optional[int] = None,
        step_days: Optional[int] = None,
        min_train_split: float = 0.5,
        max_train_split: float = 0.8,
        min_embargo_pct: float = 0.01,
        max_embargo_pct: float = 0.02,
        rf_rate: float = 0.04,
        ptc: float = 0.00035,
        max_leverage: float = 4.0,
        max_workers: int = 4,
        output_dir: str = "algos/wfov/results",
        model_params: Optional[Dict] = None,
        interval: str = "1d",
        no_plots: bool = False,
        no_save_iterations: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute complete validation session (Monte Carlo or Walk-Forward).

        Supports 3 validation modes:
        1. Monte Carlo: Random window sampling (default, backward compatible)
        2. Walk-Forward Expanding: Growing training set
        3. Walk-Forward Rolling: Fixed-size sliding window

        Args:
            model_name: Model to validate (e.g., 'svm_optimized')
            ticker: Ticker symbol (e.g., 'SPY')
            start_date: Full date range start (YYYY-MM-DD)
            end_date: Full date range end (YYYY-MM-DD)
            validation_mode: 'monte_carlo' | 'walk_forward_expanding' | 'walk_forward_rolling'

            --- Monte Carlo parameters (required if mode='monte_carlo') ---
            iterations: Number of random backtests to run
            min_lookback_days: Minimum lookback period (e.g., 365)
            max_lookback_days: Maximum lookback period (e.g., 1825)

            --- Walk-Forward parameters (required if mode='walk_forward_*') ---
            initial_train_days: Initial training period (expanding mode only)
            window_size: Training window size (rolling mode only)
            test_days: Test period size (both walk-forward modes)
            step_days: Step between windows (both walk-forward modes)

            --- Common parameters ---
            min_train_split: Minimum train ratio (default: 0.5)
            max_train_split: Maximum train ratio (default: 0.8)
            min_embargo_pct: Minimum embargo % (default: 0.0)
            max_embargo_pct: Maximum embargo % (default: 0.02)
            rf_rate: Risk-free rate (default: 0.04)
            ptc: Per-trade cost (default: 0.00035)
            max_leverage: Maximum leverage cap (default: 4.0)
            max_workers: Number of parallel workers (default: 4)
            output_dir: Output directory for results
            model_params: Model-specific parameters
            interval: Data interval (default: '1d'). Supported: '1d' (daily), '1wk' (weekly), '1mo' (monthly)

        Returns:
            Tuple of (iterations_df, summary_dict)
            - iterations_df: DataFrame with all iteration results
            - summary_dict: Aggregated statistics with statistical rigor

        Examples:
            # Monte Carlo (backward compatible)
            >>> runner = WFOVRunner(master_seed=42)
            >>> df, summary = runner.run_wfov_session(
            ...     validation_mode='monte_carlo',
            ...     model_name='svm_optimized',
            ...     ticker='SPY',
            ...     iterations=100,
            ...     start_date='2020-01-01',
            ...     end_date='2025-01-01',
            ...     min_lookback_days=365,
            ...     max_lookback_days=1825
            ... )

            # Walk-Forward Expanding
            >>> df, summary = runner.run_wfov_session(
            ...     validation_mode='walk_forward_expanding',
            ...     model_name='lstm',
            ...     ticker='SPY',
            ...     start_date='2020-01-01',
            ...     end_date='2027-01-01',
            ...     initial_train_days=1260,
            ...     test_days=252,
            ...     step_days=126
            ... )
        """
        start_time = time.time()

        # Validate mode and parameters
        valid_modes = ["monte_carlo", "walk_forward_expanding", "walk_forward_rolling"]
        if validation_mode not in valid_modes:
            raise ValueError(
                f"Invalid validation_mode: {validation_mode}. Must be one of {valid_modes}"
            )

        # Validate min_lookback_days against feature engineering warmup
        try:
            from algos.common.feature_engine import FeatureConfig

            fc = FeatureConfig(model_name=model_name)
            if fc.indicators:  # Feature engineering is configured
                max_warmup = fc.get_max_warmup()
                if min_lookback_days is not None and min_lookback_days < max_warmup * 2:
                    logger.warning(
                        f"min_lookback_days ({min_lookback_days}) may be too short for "
                        f"feature engineering warmup ({max_warmup} bars). "
                        f"Recommend at least {max_warmup * 2} days."
                    )
        except (ImportError, Exception):
            pass  # Feature engine not available or no config — skip validation

        # Mode-specific parameter validation
        if validation_mode == "monte_carlo":
            if (
                iterations is None
                or min_lookback_days is None
                or max_lookback_days is None
            ):
                raise ValueError(
                    "Monte Carlo mode requires: iterations, min_lookback_days, max_lookback_days"
                )
        elif validation_mode == "walk_forward_expanding":
            if initial_train_days is None or test_days is None or step_days is None:
                raise ValueError(
                    "Walk-Forward Expanding mode requires: initial_train_days, test_days, step_days"
                )
        elif validation_mode == "walk_forward_rolling":
            if window_size is None or test_days is None or step_days is None:
                raise ValueError(
                    "Walk-Forward Rolling mode requires: window_size, test_days, step_days"
                )

        print("\nValidation Session Starting...")
        print("━" * 80)
        print(f"Mode: {validation_mode.upper().replace('_', ' ')}")
        print(f"Model: {model_name} | Ticker: {ticker} | Interval: {interval}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Master Seed: {self.master_seed}")
        print()

        # Set model_params to empty dict if not provided
        if model_params is None:
            model_params = {}

        # Generate windows based on validation mode
        if validation_mode == "monte_carlo":
            logger.info(f"Generating {iterations} random windows (stratified)...")
            windows = generate_random_windows(
                full_start_date=start_date,
                full_end_date=end_date,
                min_lookback_days=min_lookback_days,
                max_lookback_days=max_lookback_days,
                num_iterations=iterations,
                master_seed=self.master_seed,
                stratified=True,
            )
            if len(windows) < iterations:
                logger.warning(
                    f"Only generated {len(windows)} windows (requested {iterations})"
                )
            print(
                f"Generating random windows... ✓ ({len(windows)} windows, stratified)"
            )

        elif validation_mode == "walk_forward_expanding":
            logger.info(f"Generating walk-forward expanding windows...")
            windows = generate_walk_forward_expanding_windows(
                full_start_date=start_date,
                full_end_date=end_date,
                initial_train_days=initial_train_days,
                test_days=test_days,
                step_days=step_days,
                embargo_pct=(min_embargo_pct + max_embargo_pct)
                / 2,  # Use average embargo
                master_seed=self.master_seed,
            )
            validate_walk_forward_windows(windows)
            print(
                f"Generating walk-forward windows... ✓ ({len(windows)} windows, expanding train)"
            )

        elif validation_mode == "walk_forward_rolling":
            logger.info(f"Generating walk-forward rolling windows...")
            windows = generate_walk_forward_rolling_windows(
                full_start_date=start_date,
                full_end_date=end_date,
                window_size=window_size,
                test_days=test_days,
                step_days=step_days,
                embargo_pct=(min_embargo_pct + max_embargo_pct)
                / 2,  # Use average embargo
                master_seed=self.master_seed,
            )
            validate_walk_forward_windows(windows)
            print(
                f"Generating walk-forward windows... ✓ ({len(windows)} windows, rolling)"
            )

        # Calculate spanning window and pre-load data with regime detection
        spanning_start, spanning_end = self._calculate_spanning_window(windows)
        print(f"Spanning window: {spanning_start} to {spanning_end}")

        pre_loaded_data, regimes = self._preload_data(
            ticker, spanning_start, spanning_end, interval=interval
        )
        if pre_loaded_data is None:
            logger.warning(
                "Failed to pre-load data. Iterations will download data individually (slower)."
            )

        # Prepare output files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate filename base (include mode and iteration count)
        n_windows = len(windows)
        mode_prefix = validation_mode.replace("_", "")[:6]  # Shorten for filename
        filename_base = f"{mode_prefix}_{model_name}_{ticker}_{n_windows}iter_{start_date}_{end_date}_{timestamp}"

        csv_filepath = output_path / "iterations" / f"{filename_base}_iterations.csv"
        json_filepath = output_path / "summaries" / f"{filename_base}_summary.json"
        log_filepath = output_path / "logs" / f"{filename_base}_execution.log"

        # Always create summaries dir (Step 3 reads these)
        json_filepath.parent.mkdir(parents=True, exist_ok=True)

        if not no_save_iterations:
            csv_filepath.parent.mkdir(parents=True, exist_ok=True)
            log_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging (suppress console during iterations)
        if no_save_iterations:
            logger.propagate = False
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.WARNING)
        else:
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            file_handler.setLevel(logging.INFO)
            logger.propagate = False
            logger.handlers.clear()
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

        # Initialize results tracking
        results = []
        failed_iterations = []
        csv_rows = []  # Collect all rows in memory, batch-write at end

        # Pre-generate random parameters for all iterations (for reproducibility)
        rng = np.random.default_rng(self.master_seed)
        iteration_params = []
        for i, window in enumerate(windows):
            train_split = rng.uniform(min_train_split, max_train_split)
            embargo_pct = rng.uniform(min_embargo_pct, max_embargo_pct)
            iteration_params.append((i, window, train_split, embargo_pct))

        def _build_csv_row(i, window, train_split, embargo_pct, result):
            """Build a CSV row dict from iteration result."""
            metrics = result["metrics"]
            return {
                "iteration": i,
                "model": model_name,
                "ticker": ticker,
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "lookback_days": window["lookback_days"],
                "train_split": train_split,
                "embargo_pct": embargo_pct,
                "hit_ratio": metrics.get("hit_ratio", np.nan),
                "annual_return": metrics.get("annual_return", np.nan),
                "annual_volatility": metrics.get("annual_volatility", np.nan),
                "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                "kelly_leverage": metrics.get("kelly_leverage", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "longest_drawdown_days": metrics.get("longest_drawdown_days", np.nan),
                "daily_var_95": metrics.get("daily_var_95", np.nan),
                "daily_cvar_95": metrics.get("daily_cvar_95", np.nan),
                "skewness": metrics.get("skewness", np.nan),
                "kurtosis": metrics.get("kurtosis", np.nan),
                "bh_annual_return": metrics.get("bh_annual_return", np.nan),
                "bh_sharpe_ratio": metrics.get("bh_sharpe_ratio", np.nan),
                "excess_return": metrics.get("excess_return", np.nan),
                "excess_sharpe": metrics.get("excess_sharpe", np.nan),
                "information_ratio": metrics.get("information_ratio", np.nan),
                "random_seed": window["iteration_seed"],
                "validation_mode": validation_mode,
                "regime": result.get("regime", "unknown"),
            }

        # Run iterations with parallel execution
        if max_workers > 1:
            # Parallel execution with ProcessPoolExecutor
            print(f"Running backtests (parallel, {max_workers} workers)...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all iterations
                futures = {
                    executor.submit(
                        self.run_single_iteration,
                        iteration=i,
                        window=window,
                        model_name=model_name,
                        ticker=ticker,
                        train_split=train_split,
                        embargo_pct=embargo_pct,
                        rf_rate=rf_rate,
                        ptc=ptc,
                        max_leverage=max_leverage,
                        model_params=model_params,
                        pre_loaded_data=pre_loaded_data,
                        regimes=regimes,
                        interval=interval,
                        no_plots=no_plots,
                    ): (i, window, train_split, embargo_pct)
                    for i, window, train_split, embargo_pct in iteration_params
                }

                # Process results as they complete
                completed_count = 0
                if TQDM_AVAILABLE:
                    iterator = tqdm(
                        as_completed(futures), total=len(futures), desc="WFOV Progress"
                    )
                else:
                    iterator = as_completed(futures)

                for future in iterator:
                    i, window, train_split, embargo_pct = futures[future]
                    result = future.result()

                    # Handle result
                    if result["status"] == "success":
                        results.append(result)
                        csv_rows.append(
                            _build_csv_row(i, window, train_split, embargo_pct, result)
                        )
                        logger.info(f"Iteration {i} completed successfully")
                    else:
                        failed_iterations.append(result)
                        logger.error(f"Iteration {i} failed: {result['error']}")

                    completed_count += 1

                    # Progress update (if no tqdm) - pad to 80 chars to clear line
                    if not TQDM_AVAILABLE:
                        progress_str = f"Running backtests: {completed_count}/{len(windows)} [{completed_count / len(windows) * 100:.0f}%]"
                        print(f"\r{progress_str:<80}", end="", flush=True)

                if not TQDM_AVAILABLE:
                    print()  # New line after progress

        else:
            # Serial execution (max_workers=1)
            print(f"Running backtests (serial)...")

            for idx, (i, window, train_split, embargo_pct) in enumerate(
                iteration_params
            ):
                result = self.run_single_iteration(
                    iteration=i,
                    window=window,
                    model_name=model_name,
                    ticker=ticker,
                    train_split=train_split,
                    embargo_pct=embargo_pct,
                    rf_rate=rf_rate,
                    ptc=ptc,
                    max_leverage=max_leverage,
                    model_params=model_params,
                    pre_loaded_data=pre_loaded_data,
                    regimes=regimes,
                    interval=interval,
                    no_plots=no_plots,
                )

                # Handle result
                if result["status"] == "success":
                    results.append(result)
                    csv_rows.append(
                        _build_csv_row(i, window, train_split, embargo_pct, result)
                    )
                    logger.info(f"Iteration {i} completed successfully")
                else:
                    failed_iterations.append(result)
                    logger.error(f"Iteration {i} failed: {result['error']}")

                # Progress update - pad to 80 chars to clear line
                completed = idx + 1
                progress_str = f"Running backtests: {completed}/{len(windows)} [{completed / len(windows) * 100:.0f}%]"
                print(f"\r{progress_str:<80}", end="", flush=True)

            print()  # New line after progress

        # Impute failed iterations to prevent survivorship bias
        if failed_iterations and csv_rows:
            successful_sharpes = [
                r.get("sharpe_ratio", np.nan)
                for r in csv_rows
                if not np.isnan(r.get("sharpe_ratio", np.nan))
            ]
            if successful_sharpes:
                penalty_sharpe = max(np.percentile(successful_sharpes, 5), -1.0)
            else:
                penalty_sharpe = -1.0

            for fi in failed_iterations:
                imputed_row = {col: np.nan for col in csv_rows[0].keys()}
                imputed_row["iteration"] = fi.get("iteration", -1)
                imputed_row["sharpe_ratio"] = penalty_sharpe
                imputed_row["status"] = "imputed_failure"
                csv_rows.append(imputed_row)

        # Batch-write all iteration results to CSV at once (instead of N file open/close ops)
        if not no_save_iterations:
            save_iterations_batch_to_csv(csv_rows, csv_filepath)

        # Check if any windows were successfully generated
        if len(windows) == 0:
            error_msg = "WFOV aborted: No valid windows could be generated. Check date range and lookback parameters."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Check failure rate
        failure_rate = len(failed_iterations) / len(windows) if len(windows) > 0 else 0
        if failure_rate > 0.20:
            error_msg = f"WFOV aborted: {len(failed_iterations)}/{len(windows)} iterations failed (>{failure_rate * 100:.1f}% failure rate)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Build iterations DataFrame (from memory when skipping CSV, else from disk)
        if no_save_iterations:
            iterations_df = pd.DataFrame(csv_rows)
        else:
            iterations_df = pd.read_csv(csv_filepath)

        # Generate summary statistics with enhanced statistical tests
        execution_time = time.time() - start_time

        # Detect feature engineering config for provenance tracking
        _fc_hash = None
        _n_features = None
        _feature_names = None
        try:
            from algos.common.feature_engine import FeatureConfig, FeatureEngine

            fc = FeatureConfig(model_name=model_name, ticker=ticker)
            if fc.indicators:
                _fc_hash = fc.config_hash
                engine = FeatureEngine()
                _feature_names = engine.get_feature_names(fc)
                _n_features = len(_feature_names)
        except (ImportError, Exception):
            pass  # Feature engine not available

        summary = generate_summary_statistics(
            iterations_df=iterations_df,
            model_name=model_name,
            ticker=ticker,
            iterations_requested=iterations if iterations is not None else len(windows),
            execution_time_seconds=execution_time,
            master_seed=self.master_seed,
            start_date=start_date,
            end_date=end_date,
            validation_mode=validation_mode,
            feature_config_hash=_fc_hash,
            n_features=_n_features,
            feature_names=_feature_names,
        )

        # Save summary JSON
        save_summary_json(summary, json_filepath)

        # Save failed iterations if any
        if failed_iterations and not no_save_iterations:
            failed_filepath = (
                output_path / "logs" / f"{filename_base}_failed_iterations.json"
            )
            import json

            with open(failed_filepath, "w") as f:
                # Remove traceback from JSON (too verbose)
                failed_clean = [
                    {k: v for k, v in f.items() if k != "traceback"}
                    for f in failed_iterations
                ]
                json.dump(failed_clean, f, indent=2)
            logger.info(f"Failed iterations saved to: {failed_filepath}")

        # Print console summary
        print(format_console_summary(summary))

        # Print output file paths
        print("\nOutput Files:")
        if not no_save_iterations:
            print(f"  ✓ {csv_filepath}")
        print(f"  ✓ {json_filepath}")
        if not no_save_iterations:
            print(f"  ✓ {log_filepath}")
            if failed_iterations:
                print(
                    f"  ⚠  {failed_filepath} ({len(failed_iterations)} failed iterations)"
                )

        return iterations_df, summary


def main():
    """CLI interface for enhanced validation framework (v2)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Validation Framework: Monte Carlo + Walk-Forward (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monte Carlo mode (enhanced v1 behavior)
  python -m algos.wfov.wfov_runner \\
      --mode monte_carlo \\
      --model_name svm_optimized --ticker SPY --iterations 100 \\
      --start_date 2020-01-01 --end_date 2025-01-01 \\
      --min_lookback_days 365 --max_lookback_days 1825 \\
      --seed 42

  # Walk-Forward Expanding (growing training set)
  python -m algos.wfov.wfov_runner \\
      --mode walk_forward_expanding \\
      --model_name lstm --ticker SPY \\
      --start_date 2020-01-01 --end_date 2027-01-01 \\
      --initial_train_days 1260 --test_days 252 --step_days 126 \\
      --seed 42

  # Walk-Forward Rolling (fixed-size sliding window)
  python -m algos.wfov.wfov_runner \\
      --mode walk_forward_rolling \\
      --model_name xgb_optimized --ticker NVDA \\
      --start_date 2020-01-01 --end_date 2027-01-01 \\
      --window_size 1260 --test_days 252 --step_days 126 \\
      --seed 42
        """,
    )

    # Validation mode selector
    parser.add_argument(
        "--mode",
        type=str,
        choices=["monte_carlo", "walk_forward_expanding", "walk_forward_rolling"],
        default="monte_carlo",
        help="Validation mode (default: monte_carlo)",
    )

    # Required common parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model to validate (e.g., svm_optimized, li_reg, lstm)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g., SPY, NVDA, BTC-USD)",
    )

    # Date range (common)
    parser.add_argument(
        "--start_date",
        type=str,
        default="2020-01-01",
        help="Full date range start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-01-01",
        help="Full date range end (YYYY-MM-DD)",
    )

    # Monte Carlo parameters (conditional)
    parser.add_argument(
        "--iterations", type=int, help="Number of random backtests (Monte Carlo only)"
    )
    parser.add_argument(
        "--min_lookback_days",
        type=int,
        default=365,
        help="Min lookback period in days (Monte Carlo only, default: 365)",
    )
    parser.add_argument(
        "--max_lookback_days",
        type=int,
        default=1825,
        help="Max lookback period in days (Monte Carlo only, default: 1825)",
    )

    # Walk-Forward parameters (conditional)
    parser.add_argument(
        "--initial_train_days",
        type=int,
        help="Initial training period in days (Walk-forward expanding only, e.g., 1260)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Training window size in days (Walk-forward rolling only, e.g., 1260)",
    )
    parser.add_argument(
        "--test_days",
        type=int,
        help="Test period size in days (Walk-forward modes, e.g., 252)",
    )
    parser.add_argument(
        "--step_days",
        type=int,
        help="Step size between windows in days (Walk-forward modes, e.g., 126)",
    )

    # Train split range (common)
    parser.add_argument(
        "--min_train_split",
        type=float,
        default=0.5,
        help="Min train/test split ratio (default: 0.5)",
    )
    parser.add_argument(
        "--max_train_split",
        type=float,
        default=0.8,
        help="Max train/test split ratio (default: 0.8)",
    )

    # Embargo range (common)
    parser.add_argument(
        "--min_embargo_pct",
        type=float,
        default=0.0,
        help="Min embargo percentage (default: 0.0)",
    )
    parser.add_argument(
        "--max_embargo_pct",
        type=float,
        default=0.02,
        help="Max embargo percentage (default: 0.02 = 2%%)",
    )

    # Fixed parameters (common)
    parser.add_argument(
        "--rf_rate",
        type=float,
        default=0.04,
        help="Risk-free rate (default: 0.04 = 4%%)",
    )
    parser.add_argument(
        "--ptc",
        type=float,
        default=0.00035,
        help="Per-trade cost (default: 0.00035 = 0.035%%)",
    )
    parser.add_argument(
        "--max_leverage",
        type=float,
        default=4.0,
        help="Maximum leverage cap (default: 4.0)",
    )

    # Execution control
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, set to 1 for serial)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="algos/wfov/results",
        help="Output directory for results",
    )

    # ARIMA-specific parameters (fixed per session)
    parser.add_argument(
        "--arima_threshold", type=float, help="ARIMA threshold for threshold method"
    )
    parser.add_argument("--arima_zscore", type=float, help="ARIMA z-score threshold")
    parser.add_argument(
        "--arima_lookback", type=int, help="ARIMA lookback window in days"
    )
    parser.add_argument(
        "--arima_signal_method",
        type=str,
        default="z_score",
        help="ARIMA signal generation method",
    )

    # DQN-specific parameters (fixed per session)
    parser.add_argument(
        "--dqn_epochs", type=int, default=50, help="DQN training epochs"
    )
    parser.add_argument("--dqn_batch_size", type=int, default=32, help="DQN batch size")
    parser.add_argument(
        "--dqn_hidden_units", type=int, default=64, help="DQN hidden units"
    )

    # Data interval
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly). Default: 1d",
    )

    # Suppress plots (for batch/summary modes)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots/images (for batch or summary modes)",
    )
    parser.add_argument(
        "--auto-workers",
        action="store_true",
        help="Auto-tune max_workers based on model type (overrides --max_workers)",
    )
    parser.add_argument(
        "--no-save-iterations",
        action="store_true",
        help="Skip saving iteration CSVs and execution logs (summary JSON still saved)",
    )

    args = parser.parse_args()

    # Mode-specific parameter validation
    if args.mode == "monte_carlo":
        if args.iterations is None:
            parser.error("--iterations required for Monte Carlo mode")
        if args.min_lookback_days >= args.max_lookback_days:
            parser.error("--min_lookback_days must be less than --max_lookback_days")

    elif args.mode == "walk_forward_expanding":
        if (
            args.initial_train_days is None
            or args.test_days is None
            or args.step_days is None
        ):
            parser.error(
                "Walk-Forward Expanding requires: --initial_train_days, --test_days, --step_days"
            )

    elif args.mode == "walk_forward_rolling":
        if args.window_size is None or args.test_days is None or args.step_days is None:
            parser.error(
                "Walk-Forward Rolling requires: --window_size, --test_days, --step_days"
            )

    # Common validations
    if args.min_train_split >= args.max_train_split:
        parser.error("--min_train_split must be less than --max_train_split")
    if not (0 <= args.min_embargo_pct <= args.max_embargo_pct <= 0.05):
        parser.error(
            "--min_embargo_pct and --max_embargo_pct must be in range [0, 0.05]"
        )

    # Build model-specific parameters
    model_params = {}
    if args.model_name == "arima":
        if (
            args.arima_threshold is None
            or args.arima_zscore is None
            or args.arima_lookback is None
        ):
            parser.error(
                "ARIMA model requires --arima_threshold, --arima_zscore, and --arima_lookback"
            )
        model_params = {
            "signal_method": args.arima_signal_method,
            "threshold": args.arima_threshold,
            "z_score_threshold": args.arima_zscore,
            "lookback_window": args.arima_lookback,
        }
    elif args.model_name == "dqn":
        model_params = {
            "dqn_epochs": args.dqn_epochs,
            "dqn_batch_size": args.dqn_batch_size,
            "dqn_hidden_units": args.dqn_hidden_units,
        }

    # Auto-tune workers if requested
    max_workers = args.max_workers
    if args.auto_workers:
        max_workers = suggest_max_workers(args.model_name, default=args.max_workers)
        print(f"Auto-tuned max_workers={max_workers} for model '{args.model_name}'")

    # Initialize runner
    runner = WFOVRunner(master_seed=args.seed)

    # Run validation session (mode-aware)
    iterations_df, summary = runner.run_wfov_session(
        validation_mode=args.mode,
        model_name=args.model_name,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        # Monte Carlo params
        iterations=args.iterations,
        min_lookback_days=args.min_lookback_days,
        max_lookback_days=args.max_lookback_days,
        # Walk-Forward params
        initial_train_days=args.initial_train_days,
        window_size=args.window_size,
        test_days=args.test_days,
        step_days=args.step_days,
        # Common params
        min_train_split=args.min_train_split,
        max_train_split=args.max_train_split,
        min_embargo_pct=args.min_embargo_pct,
        max_embargo_pct=args.max_embargo_pct,
        rf_rate=args.rf_rate,
        ptc=args.ptc,
        max_leverage=args.max_leverage,
        max_workers=max_workers,
        output_dir=args.output_dir,
        model_params=model_params,
        interval=args.interval,
        no_plots=args.no_plots,
        no_save_iterations=args.no_save_iterations,
    )

    print(f"\n✅ Validation session completed successfully ({args.mode} mode)!")


if __name__ == "__main__":
    main()
