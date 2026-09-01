# algos/backtest_code/models/arima_model.py

import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score
import sys  # For sys.exit
from algos.common.utils import (
    RedirectStdoutToFile,
)  # For logging within the model function
from algos.common.embargo_utils import (
    apply_embargo_to_walk_forward,
    get_embargo_message,
)
from collections import deque


def run_arima_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "ARIMA_model",
    signal_method: str = "z_score",  # "z_score", "threshold", "percentile", or "simple"
    threshold: float = 0.0002,  # For threshold method
    z_score_threshold: float = 1,  # For z-score method
    lookback_window: int = 5,
    embargo_pct: float = 0.02,  # Embargo percentage
    interval: str = "1d",  # Data interval for interval-aware validation
    **kwargs,
):  # Accept additional arguments for compatibility
    """
    Implements the ARIMA-based trading strategy with walk-forward validation and embargo.

    Signal methods:
    - 'z_score': Use z-score of predictions to generate balanced signals (default, recommended)
    - 'threshold': Buy if prediction > threshold, Sell if < -threshold
    - 'percentile': Buy if in top percentile, Sell if in bottom percentile
    - 'simple': Original method using np.sign (tends to generate mostly buy signals)

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns,
                             and 'annual_trading_periods' as an attribute.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        signal_method (str): Method to generate trading signals from predictions.
        threshold (float): Threshold for 'threshold' method.
        z_score_threshold (float): Z-score threshold for 'z_score' method.
        lookback_window (int): Window size for rolling statistics.
        embargo_pct (float): Embargo percentage (default 2%). Set to 0 to disable.
        interval (str): Data interval for auto-adjusting minimums ('1d', '1wk', '1mo').
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted ARIMA model (statsmodels.tsa.arima.model.ARIMAResultsWrapper).
    """
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    # Initial split
    split = int(len(data) * initial_train_split_ratio)
    train_initial = data.iloc[:split].copy()
    test_initial = data.iloc[split:].copy()

    # Apply embargo (interval-aware)
    train, test = apply_embargo_to_walk_forward(
        train_initial, test_initial, embargo_pct=embargo_pct, interval=interval
    )

    # Log embargo if enabled
    if embargo_pct > 0:
        embargo_size = len(test_initial) - len(test)
        print(get_embargo_message(embargo_size, len(data), len(train), len(test)))

    if train.empty or test.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Train or test DataFrame is empty after splitting for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Step 1: Find the optimal ARIMA order using auto_arima ONCE on the training data ---
    # optimal_order = (1, 0, 0) # Default fallback
    optimal_order = (1, 0, 1)
    # with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
    #     print('=' * 50)
    #     print(f"\nFinding Optimal ARIMA Order for {log_prefix} using pmdarima.auto_arima...\n")
    #     try:
    #         optimal_model_search = pm.auto_arima(train['returns'],
    #                                              start_p=1, start_q=1,
    #                                              test='adf',        # Use ADF test to determine 'd'
    #                                              max_p=5, max_q=5,  # Max order for p and q
    #                                              m=1,               # Non-seasonal data
    #                                              d=None,            # Let auto_arima determine 'd'
    #                                              seasonal=False,
    #                                              trace=False,       # Set to True for verbose output during search
    #                                              error_action='ignore',
    #                                              suppress_warnings=True,
    #                                              stepwise=True,
    #                                              information_criterion='aic',
    #                                              n_jobs=-1
    #                                              )
    #         optimal_order = optimal_model_search.order
    #         print(f"\nOptimal ARIMA Order (p,d,q): {optimal_order}")
    #         # print(f"Optimal Model Summary:\n{optimal_model_search.summary()}") # Too verbose for logs
    #     except Exception as e:
    #         print(f"Error during auto_arima search: {e}")
    #         print("Defaulting to ARIMA(1,0,0) order due to error.")
    #         optimal_order = (1, 0, 0)
    #     print('=' * 50 + '\n')

    # --- Step 2: Implement Walk-Forward Validation using the OPTIMAL ORDER ---
    predictions_list = []
    history = [x for x in train["returns"]]
    last_fitted_model = None  # To store the final model object

    # Store recent predictions for dynamic thresholding (used by some methods)
    recent_predictions = deque(maxlen=lookback_window)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(
            f"\n{log_prefix} Model Training and Forecasting (Walk-Forward with Optimal Order):\n"
        )
        print(f"Signal Method: {signal_method}")
        if signal_method == "threshold":
            print(f"Threshold: ±{threshold}")
        elif signal_method == "z_score":
            print(f"Z-Score Threshold: ±{z_score_threshold}")
            print(f"Lookback Window: {lookback_window} days")
        elif signal_method == "percentile":
            print(f"Lookback Window: {lookback_window} days")

        for t in range(len(test)):
            try:
                model = ARIMA(history, order=optimal_order)
                model_fit = model.fit()
                forecast_result = model_fit.forecast(steps=1)
                yhat = forecast_result[0]
                predictions_list.append(yhat)
                recent_predictions.append(yhat)
                last_fitted_model = model_fit  # Update with the last fitted model

                obs = test["returns"].iloc[t]
                history.append(obs)

                if (t + 1) % 100 == 0 or t == len(test) - 1:
                    print(f"Processed {t + 1}/{len(test)} test points.")
            except Exception as e:
                predictions_list.append(np.nan)
                if t < len(test):
                    obs = test["returns"].iloc[t]
                    history.append(obs)
                print(
                    f"Warning: {log_prefix} fit/forecast failed at step {t} with error: {e}"
                )

        test["predicted_returns"] = pd.Series(
            predictions_list, index=test.index[: len(predictions_list)]
        )
        test.dropna(subset=["predicted_returns"], inplace=True)
        print(f"\n{log_prefix} Walk-Forward Forecasting Complete.")
        print("=" * 50 + "\n")

    # --- Step 3: Generate trading signals based on selected method ---
    if signal_method == "z_score":
        # Z-score normalization method (recommended for balanced signals)
        # Calculate rolling mean and std of predictions
        test["pred_mean"] = (
            test["predicted_returns"]
            .rolling(window=lookback_window, min_periods=5)
            .mean()
        )
        test["pred_std"] = (
            test["predicted_returns"]
            .rolling(window=lookback_window, min_periods=5)
            .std()
        )

        # Fill initial NaN values with expanding window stats
        for i in range(len(test)):
            if pd.isna(test["pred_mean"].iloc[i]):
                test["pred_mean"].iloc[i] = (
                    test["predicted_returns"].iloc[: i + 1].mean()
                )
            if pd.isna(test["pred_std"].iloc[i]):
                test["pred_std"].iloc[i] = test["predicted_returns"].iloc[: i + 1].std()

        # Fill any remaining NaN std with a small value
        test["pred_std"] = test["pred_std"].fillna(test["predicted_returns"].std())

        # Calculate z-score
        test["z_score"] = (test["predicted_returns"] - test["pred_mean"]) / (
            test["pred_std"] + 1e-8
        )

        # Generate signals based on z-score
        test["position"] = np.where(
            test["z_score"] > z_score_threshold,
            1,
            np.where(test["z_score"] < -z_score_threshold, -1, 0),
        )

        # If position is 0 (neutral), use simple sign as fallback
        neutral_mask = test["position"] == 0
        test.loc[neutral_mask, "position"] = np.sign(
            test.loc[neutral_mask, "predicted_returns"]
        )

    elif signal_method == "threshold":
        # Simple threshold method
        test["position"] = np.where(
            test["predicted_returns"] > threshold,
            1,
            np.where(test["predicted_returns"] < -threshold, -1, 0),
        )
        # If position is 0 (hold), carry forward previous position or default to 1
        test["position"] = (
            test["position"].replace(0, np.nan).fillna(method="ffill").fillna(1)
        )

    elif signal_method == "percentile":
        # Percentile-based method
        # Calculate rolling percentile rank
        test["pred_rank"] = (
            test["predicted_returns"]
            .rolling(window=lookback_window, min_periods=5)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        )

        # Generate signals: Buy if top 30%, Sell if bottom 30%
        test["position"] = np.where(
            test["pred_rank"] > 0.7, 1, np.where(test["pred_rank"] < 0.3, -1, 0)
        )
        test["position"] = (
            test["position"].replace(0, np.nan).fillna(method="ffill").fillna(1)
        )

    else:  # signal_method == "simple" or default
        # Original simple method (tends to produce mostly buy signals)
        test["position"] = np.sign(test["predicted_returns"].values).flatten()
        # Handle edge case where sign might return 0 (for exactly 0 prediction)
        test["position"] = np.where(test["position"] == 0, 1, test["position"])

    # Calculate strategy returns
    test["strategy"] = test["position"].shift(1) * test["returns"]

    # Apply transaction costs
    from algos.common.config import PTC

    transaction_cost_log_impact = np.log(1 - PTC)
    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff() != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # Calculate hit ratio
    # test['direction'] is -1/1, test['position'] is -1/1
    # Execution-aligned: position[t] predicts direction[t+1]
    test_hit_ratio = accuracy_score(
        test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
    )

    # Calculate signal distribution
    buy_signals = (test["position"] == 1).sum()
    sell_signals = (test["position"] == -1).sum()
    total_signals = len(test)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: N/A (for walk-forward {log_prefix})")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")

        print(f"\nSignal Distribution ({signal_method} method):")
        print(
            f"Buy Signals: {buy_signals}/{total_signals} ({100 * buy_signals / total_signals:.1f}%)"
        )
        print(
            f"Sell Signals: {sell_signals}/{total_signals} ({100 * sell_signals / total_signals:.1f}%)"
        )

        print(f"\nPrediction Statistics:")
        print(f"Mean Predicted Return: {test['predicted_returns'].mean():.6f}")
        print(f"Std Predicted Return: {test['predicted_returns'].std():.6f}")
        print(f"Min Predicted Return: {test['predicted_returns'].min():.6f}")
        print(f"Max Predicted Return: {test['predicted_returns'].max():.6f}")

        if signal_method == "z_score" and "z_score" in test.columns:
            print(f"\nZ-Score Statistics:")
            print(f"Mean Z-Score: {test['z_score'].mean():.4f}")
            print(f"Std Z-Score: {test['z_score'].std():.4f}")
            print(f"Min Z-Score: {test['z_score'].min():.4f}")
            print(f"Max Z-Score: {test['z_score'].max():.4f}")

        print("=" * 50 + "\n" * 3)

        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    return test, last_fitted_model
