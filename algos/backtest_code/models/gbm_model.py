import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys  # For sys.exit
import pickle  # Used by XGBoost for model serialization
import re  # Still needed for interval parsing if it was uncommented and needed somewhere
import os  # For os.getcwd, though Pathlib is preferred
from pathlib import Path

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage


def run_xgboost_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "XGBoost_model",
    **kwargs,
):
    """
    Implements the XGBoost-based trading strategy with a fixed train/test split.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also implicitly contain 'annual_trading_periods' if needed
                             for downstream calculations (though not directly used by this model's logic).
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted XGBoost model (xgboost.core.Booster or xgb.XGBRegressor).
    """
    # --- Input Data Validation ---
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    if "returns" not in data.columns or "direction" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data for {log_prefix} is missing 'returns' or 'direction' columns. Exiting."
            )
        sys.exit(1)

    # --- Feature Engineering: Create Lagged Returns ---
    lags = 5  # Number of lagged returns to use as features
    cols = []  # List to store column names of lagged features
    for lag in range(1, lags + 1):
        col = "lag_{}".format(lag)
        data[col] = data["returns"].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)  # Drop rows with NaN values introduced by shifting

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating lags and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split ---
    split = int(len(data) * initial_train_split_ratio)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Insufficient data for training or testing {log_prefix}. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    # --- XGBoost Model Definition and Training ---
    model = xgb.XGBRegressor(
        objective="reg:squarederror",  # For regression, predicting a continuous score
        n_estimators=100,  # Number of boosting rounds (trees)
        random_state=1000,  # For reproducibility
        learning_rate=0.1,  # Step size shrinkage
        max_depth=3,  # Maximum depth of a tree
        n_jobs=-1,
    )  # Use all available cores

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        try:
            model.fit(
                X_train_scaled, train["direction"]
            )  # Train to predict the 'direction' (-1 or 1)
            print("Training finished.")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)  # Exit on critical training error
        print("=" * 50 + "\n")

    # --- Predictions ---
    # Predict continuous values first
    train_predictions_continuous = model.predict(X_train_scaled)
    test_predictions_continuous = model.predict(X_test_scaled)

    # Convert continuous predictions to binary (-1 or 1)
    # np.sign returns 0 if input is 0. We'll default to 1 (long position) in that case.
    train_predictions_binary = np.sign(train_predictions_continuous).flatten()
    train_predictions_binary[train_predictions_binary == 0] = 1

    test_predictions_binary = np.sign(test_predictions_continuous).flatten()
    test_predictions_binary[test_predictions_binary == 0] = 1

    # --- Calculate Hit Ratios ---
    train_hit_ratio = accuracy_score(train["direction"], train_predictions_binary)

    # Assign predictions to 'test' DataFrame
    test["predicted_returns"] = pd.Series(
        test_predictions_continuous, index=test.index
    )  # Store continuous prediction if desired
    test["position"] = test_predictions_binary

    # Ensure test['direction'] and test['position'] are not empty before calculating accuracy
    if (
        not test["direction"].empty
        and not test["position"].empty
        and len(test["direction"]) == len(test["position"])
    ):
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = (
            np.nan
        )  # Cannot calculate if test set or predictions are problematic

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Strategy Returns Calculation ---
    test.dropna(subset=["position", "returns"], inplace=True)
    if test.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'test' DataFrame is empty after dropping NaNs for strategy calculation for {log_prefix}. Exiting."
            )
        sys.exit(1)

    test["strategy"] = test["position"].shift(1) * test["returns"]

    # --- Apply Transaction Costs ---
    # PTC is imported from algos.common.config
    transaction_cost_log_impact = np.log(1 - PTC)

    # Apply transaction cost when position changes. fillna(0) handles the first row.
    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis
    return test, model
