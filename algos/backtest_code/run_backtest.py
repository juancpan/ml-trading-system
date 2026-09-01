# algos/backtest_code/run_backtest.py

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import sys
import algos.common.persistence as save_model

# Ensure these paths are correct relative to your project root
# Add the project root to sys.path to allow imports from algos.common
project_root = Path(__file__).resolve().parents[2] # Go up two levels from run_backtest.py
sys.path.insert(0, str(project_root))


from algos.common.data_loader import load_and_preprocess_data
from algos.common.metrics import calculate_strategy_performance
from algos.common.risk_analysis import calculate_risk_metrics, plot_drawdowns
from algos.common.utils import RedirectStdoutToFile # Assuming this is in algos.common.utils

# Import the model functions
from algos.backtest_code.models.dqn_model import run_dqn_strategy
from algos.backtest_code.models.cnn_model import run_cnn_strategy
from algos.backtest_code.models.arima_model import run_arima_strategy   
from algos.backtest_code.models.dnn_model import run_dnn_strategy
from algos.backtest_code.models.sklearn_dnn_model import run_sklearn_dnn_strategy
from algos.backtest_code.models.gbm_model import run_xgboost_strategy
from algos.backtest_code.models.gnb_model import run_gnb_strategy
from algos.backtest_code.models.kmeans_model import run_kmeans_strategy
from algos.backtest_code.models.linear_regression_model import run_linear_regression_strategy
from algos.backtest_code.models.logistic_regression_model import run_logistic_regression_strategy
# from algos.backtest_code.models.ols_model import run_ols_strategy # Assuming you have this
from algos.backtest_code.models.lstm_model import run_lstm_strategy
from algos.backtest_code.models.random_forest_model import run_random_forest_strategy
from algos.backtest_code.models.svm_model import run_svm_strategy
from algos.backtest_code.models.sarimax_model import run_sarimax_strategy
from algos.backtest_code.models.tcn_model import run_tcn_strategy
from algos.backtest_code.models.var_model import run_var_strategy
from algos.deprecated.arima_model_v2 import run_arima_v2_strategy
from algos.backtest_code.models.svm_tuned_model import run_svm_tuned_strategy
from algos.backtest_code.models.stacking_model import run_stacking_strategy

# Set up logging directory
logs_dir = Path(os.getcwd()) / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Import optimized models
try:
    from algos.backtest_code.models.svm_model_optimized import run_svm_strategy as run_svm_optimized
    from algos.backtest_code.models.random_forest_optimized import run_random_forest_strategy as run_rf_optimized
    from algos.backtest_code.models.lstm_optimized import run_lstm_strategy as run_lstm_optimized
    from algos.backtest_code.models.xgboost_optimized import run_xgboost_strategy as run_xgb_optimized
    from algos.backtest_code.models.linear_models_optimized import (
        run_linear_regression_strategy as run_linear_optimized,
        run_logistic_regression_strategy as run_logistic_optimized,
        run_sgd_linear_strategy as run_sgd_optimized
    )
    from algos.backtest_code.models.ensemble_optimized import (
        run_ensemble_strategy as run_ensemble_optimized,
        run_adaptive_ensemble_strategy as run_adaptive_ensemble
    )
    OPTIMIZED_MODELS_AVAILABLE = True
except ImportError:
    OPTIMIZED_MODELS_AVAILABLE = False
    print("Warning: Optimized models not available. Using original models only.")

# MODEL_REGISTRY maps model names to their respective run functions
MODEL_REGISTRY = {
    # Original models
    'dqn': run_dqn_strategy,
    'cnn': run_cnn_strategy,
    'arima': run_arima_strategy,
    'dnn': run_dnn_strategy,
    'sklearn_dnn': run_sklearn_dnn_strategy,
    'gbm': run_xgboost_strategy,
    'gnb': run_gnb_strategy,
    'kmeans': run_kmeans_strategy,
    'li_reg': run_linear_regression_strategy,
    'log_reg': run_logistic_regression_strategy,
    'lstm': run_lstm_strategy,
    'rf': run_random_forest_strategy,
    'sarimax': run_sarimax_strategy,
    'svm': run_svm_strategy,
    'tcn': run_tcn_strategy,
    'var': run_var_strategy,
    'arima_v2': run_arima_v2_strategy,  # New ARIMA model with different parameters
    'svm_tuned': run_svm_tuned_strategy,  # New SVM model with tuned parameters
    'stacking': run_stacking_strategy,  # New stacking model
}

# Add optimized models if available
if OPTIMIZED_MODELS_AVAILABLE:
    MODEL_REGISTRY.update({
        # Optimized versions
        'svm_optimized': run_svm_optimized,
        'rf_optimized': run_rf_optimized,
        'random_forest_optimized': run_rf_optimized,
        'lstm_optimized': run_lstm_optimized,
        'xgb_optimized': run_xgb_optimized,
        'xgboost_optimized': run_xgb_optimized,
        'linear_optimized': run_linear_optimized,
        'logistic_optimized': run_logistic_optimized,
        'sgd_optimized': run_sgd_optimized,
        'ensemble_optimized': run_ensemble_optimized,
        'ensemble_voting': run_ensemble_optimized,
        'ensemble_stacking': run_ensemble_optimized,
        'ensemble_adaptive': run_adaptive_ensemble,
    })

def run_single_backtest(model_name: str, symbol: str, train_split: float, rf_rate: float, ptc: float,
                        ticker: str = None, start_date: str = None, end_date: str = None,
                        interval: str = None, data_path: str = None, **kwargs):
    """
    Executes a complete backtest for a single model and set of parameters.

    Args:
        model_name (str): The name of the strategy model to run (e.g., 'dqn', 'ols').
        symbol (str): The price column to use ('Adj Close' or 'Close').
        train_split (float): Ratio for initial training data split.
        rf_rate (float): Risk-free rate for Sharpe Ratio calculation.
        ptc (float): Per-trade transaction cost.
        ticker (str, optional): The financial instrument ticker. Required if data_path is None.
        start_date (str, optional): Start date for data download (YYYY-MM-DD). Required if data_path is None.
        end_date (str, optional): End date for data download (YYYY-MM-DD). Required if data_path is None.
        interval (str, optional): Data interval (e.g., '1d', '1h'). Required if data_path is None.
        data_path (str, optional): Path to a pre-processed CSV data file. If provided,
                                     ticker, start_date, end_date, interval are ignored.
        **kwargs: Additional model-specific parameters (e.g., dqn_epochs, dqn_batch_size).
    """
    # Adjust log filename prefix based on data source
    if data_path:
        data_source_id = Path(data_path).stem # Use filename as identifier
        print(f"Using data from file: {data_path}")
    else:
        data_source_id = f"{ticker}_{interval}_{start_date}_{end_date}"
        print(f"Downloading data for {ticker} from {start_date} to {end_date} ({interval}).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_prefix = f"{model_name}_{data_source_id}_{timestamp}"
    
    # Configure global settings for consistency
    from algos.common import config
    config.PTC = ptc
    config.RF_RATE = rf_rate

    with RedirectStdoutToFile(f"{logs_dir}/{log_filename_prefix}_overall_output.txt"):
        print(f"Starting backtest for {model_name} on {data_source_id}.")
        print(f"Train split: {train_split*100}%, Risk-free rate: {rf_rate*100}%, PTC: {ptc*100:.4f}%")
        print(f"Model-specific parameters: {kwargs}")

        # --- 1. Load and Preprocess Data ---
        print("\n" + "=" * 50)
        print("1. Data Loading and Preprocessing")
        print("=" * 50)
        
        # Pass data_path if provided, otherwise pass yfinance parameters
        # The data_loader.py will now handle the logic of which source to use
        data = load_and_preprocess_data(
            ticker=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            symbol=symbol,
            user_provided_file_path=data_path, # New argument name to avoid conflict with internal 'file_path'
            log_filename=f"{logs_dir}/{log_filename_prefix}_data_loading.txt"
        )

        if data is None or data.empty:
            print("Failed to load or preprocess data. Exiting.")
            return

        # Ensure annual_trading_periods is set for metrics calculations
        if 'annual_trading_periods' not in data.attrs:
            print("Error: 'annual_trading_periods' not set in data attributes. Cannot proceed.")
            return

        # --- 2. Run Strategy Model ---
        print("\n" + "=" * 50)
        print(f"2. Running {model_name} Strategy")
        print("=" * 50)
        strategy_run_function = MODEL_REGISTRY.get(model_name)
        if not strategy_run_function:
            print(f"Error: Model '{model_name}' not found in registry.")
            return

        test_df_results, final_model = strategy_run_function(
            data=data.copy(), # Pass a copy to avoid unintended modifications
            initial_train_split_ratio=train_split,
            log_prefix=log_filename_prefix,
            **kwargs # Pass model-specific args
        )

        if test_df_results is None or test_df_results.empty:
            print(f"Strategy {model_name} did not return valid results. Exiting.")
            return

        # Attach annual_trading_periods to test_df_results for metrics calculation
        test_df_results.attrs['annual_trading_periods'] = data.attrs['annual_trading_periods']


        # --- 3. Calculate Performance Metrics ---
        print("\n" + "=" * 50)
        print("3. Performance Metrics Calculation")
        print("=" * 50)
        performance_metrics = calculate_strategy_performance(test_df_results, model_name, log_filename_prefix)
        
        # --- 4. Risk Analysis ---
        print("\n" + "=" * 50)
        print("4. Risk Analysis")
        print("=" * 50)
        risk_metrics = calculate_risk_metrics(test_df_results, model_name, log_filename_prefix)
        plot_drawdowns(test_df_results, model_name, log_filename_prefix)


        print("\n" + "=" * 50)
        print("Backtest Complete!")
        print("=" * 50)
        
        save_model.save_model(
            model_obj=final_model,
            model_name=model_name,
            ticker=ticker if ticker else "N/A",  # Use 'N/A' if ticker is None
            symbol=symbol,
            start=start_date if start_date else "N/A",  # Use 'N/A'
            end=end_date if end_date else "N/A",  # Use 'N/A'
            interval=interval if interval else "N/A",  # Use 'N/A'
            timestamp=timestamp,)

def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a trading strategy.")
    parser.add_argument('--model_name', type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Name of the model to run (e.g., 'dqn', 'ols').")
    
    # Optional data path argument - if provided, other data args are ignored
    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to a pre-processed CSV data file. If provided, "
                             "ticker, start, end, and interval will be discarded for data loading.")

    # Data download arguments (required if data_path is NOT provided)
    parser.add_argument('--ticker', type=str, default=None, help="Ticker symbol (e.g., 'QQQ', 'BTC-USD').")
    parser.add_argument('--start', type=str, default="2019-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument('--end', type=str, default="2025-06-01", help="End date (YYYY-MM-DD).")
    parser.add_argument('--interval', type=str, default="1d",
                        help="Data interval (e.g., '1m', '5m', '15m', '1h', '1d', '1wk', '1mo').")
    
    parser.add_argument('--train_split', type=float, default=0.5,
                        help="Ratio of data for initial training split (0.0 to 1.0).")
    parser.add_argument('--rf_rate', type=float, default=0.04,
                        help="Annual risk-free rate (e.g., 0.04 for 4%).")
    parser.add_argument('--ptc', type=float, default=0.00035,
                        help="Per-trade transaction cost (e.g., 0.00035 for 0.035%).")
    parser.add_argument('--symbol', type=str, default="Adj Close",
                        help="Column name for price data ('Adj Close' or 'Close').")
    
    # Model-specific arguments for DQN
    parser.add_argument('--dqn_epochs', type=int, default=50, help="Number of epochs for DQN training.")
    parser.add_argument('--dqn_batch_size', type=int, default=32, help="Batch size for DQN training.")
    parser.add_argument('--dqn_hidden_units', type=int, default=64, help="Number of hidden units for DQN layers.")
    # parser.add_argument('--lags', type=int, default=5, help="Number of lag features for models (e.g., DQN, OLS).")


    args = parser.parse_args()

    # Validate data source arguments
    if args.data_path:
        if not os.path.exists(args.data_path):
            parser.error(f"Error: Data file not found at '{args.data_path}'")
        # If data_path is provided, discard yfinance-specific args by setting them to None
        ticker_val = None
        start_val = None
        end_val = None
        interval_val = None
    else:
        # If data_path is not provided, ticker, start, end, interval are required
        if not all([args.ticker, args.start, args.end, args.interval]):
            parser.error("Error: 'ticker', 'start', 'end', and 'interval' are required if '--data_path' is not provided.")
        ticker_val = args.ticker
        start_val = args.start
        end_val = args.end
        interval_val = args.interval


    # Collect kwargs for the specific model
    model_specific_kwargs = {}
    if args.model_name == 'dqn':
        model_specific_kwargs['dqn_epochs'] = args.dqn_epochs
        model_specific_kwargs['dqn_batch_size'] = args.dqn_batch_size
        model_specific_kwargs['dqn_hidden_units'] = args.dqn_hidden_units
    #     model_specific_kwargs['lags'] = args.lags # Lags is common to several models

    # # Add kwargs for other models as needed
    # elif args.model_name in ['ols', 'svm', 'rf', 'xgb', 'sarimax', 'arima_v2', 'svm_tuned', 'stacking', 'cnn', 'dnn', 'sklearn_dnn', 'kmeans', 'li_reg', 'log_reg', 'lstm', 'tcn', 'var']: # These models also use lags
    #      model_specific_kwargs['lags'] = args.lags # Ensure 'lags' is passed if needed

    run_single_backtest(
        model_name=args.model_name,
        ticker=ticker_val,          # Now can be None
        start_date=start_val,       # Now can be None
        end_date=end_val,           # Now can be None
        interval=interval_val,      # Now can be None
        data_path=args.data_path, # New argument
        train_split=args.train_split,
        rf_rate=args.rf_rate,
        ptc=args.ptc,
        symbol=args.symbol,
        **model_specific_kwargs
    )

if __name__ == "__main__":
    main()