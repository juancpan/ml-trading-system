import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys  # For sys.exit
import pickle  # For saving/loading the model

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)


def run_linear_regression_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "LinearRegression_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements a Linear Regression-based trading strategy with embargo. The model predicts future returns
    (or direction, as it's regressing on 'direction' here), and these predictions are then
    binarized to generate trading signals (-1 for sell, 1 for buy).

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also implicitly contain 'annual_trading_periods' as an attribute
                             if needed for downstream calculations (though not directly used by this model's core logic).
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        embargo_pct (float): Embargo percentage (default 2%). Set to 0 to disable.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'predicted_returns' (continuous predictions),
                                         'position' (binarized trading signals), 'strategy', 'strategy_tc'.
               final_model_object: The last fitted LinearRegression model.
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

    # --- Train/Test Split with Embargo ---
    split = int(len(data) * initial_train_split_ratio)

    if embargo_pct > 0:
        # Calculate embargo size
        embargo_size = calculate_embargo_size(len(data), embargo_pct, min_samples=5)

        # Apply embargo
        train = data.iloc[:split].copy()
        test = data.iloc[split + embargo_size :].copy()  # Skip embargo buffer

        # Validate
        validate_embargo_split(len(train), embargo_size, len(test), min_test_samples=30)

        # Log embargo configuration
        print(get_embargo_message(embargo_size, len(data), len(train), len(test)))
    else:
        # No embargo (backward compatibility)
        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()

        print(
            "\nNote: Embargo disabled (embargo_pct=0). Using simple train/test split."
        )

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

    # --- Linear Regression Model Definition and Training ---
    model = LinearRegression()

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        try:
            # Linear Regression is trained to predict the 'direction' (-1 or 1)
            model.fit(X_train_scaled, train["direction"])
            print(f"Model fitted successfully: {model}")
            print(f"Coefficients: {model.coef_}")
            print(f"Intercept: {model.intercept_}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)  # Exit on critical training error
        print("=" * 50 + "\n")

    # --- Predictions ---
    # Linear Regression predicts continuous values.
    train_predictions_continuous = model.predict(X_train_scaled)
    test_predictions_continuous = model.predict(X_test_scaled)

    # Convert continuous predictions to binary trading signals (-1 or 1)
    train_predictions_binary = np.sign(train_predictions_continuous).flatten()
    test_predictions_binary = np.sign(test_predictions_continuous).flatten()

    # Store continuous predictions in the 'test' DataFrame for potential later analysis
    test["predicted_returns"] = pd.Series(test_predictions_continuous, index=test.index)
    # The 'position' column holds the binarized trading signal
    test["position"] = pd.Series(test_predictions_binary, index=test.index)

    # --- Calculate Hit Ratios ---
    train_hit_ratio = accuracy_score(train["direction"], train_predictions_binary)

    # Ensure test['direction'] and test['position'] are valid before calculating accuracy
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

    # Apply transaction cost only when position changes. fillna(0) handles the first row's diff().
    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
    return test, model
