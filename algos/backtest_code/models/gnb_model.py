import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    StandardScaler,
)  # Retain if feature scaling is ever desired for GNB
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


def run_gnb_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "GNB_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements the Gaussian Naive Bayes (GNB)-based trading strategy with a fixed train/test split.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also implicitly contain 'annual_trading_periods' as an attribute
                             if needed for downstream calculations (though not directly used by this model's core logic).
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
                                         Note: 'predicted_returns' for GNB will effectively be the binary prediction,
                                         as GNB is a classifier.
               final_model_object: The last fitted GaussianNB model.
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
        embargo_size = calculate_embargo_size(len(data), embargo_pct, min_samples=5)
        train = data.iloc[:split].copy()
        test = data.iloc[split + embargo_size :].copy()
        validate_embargo_split(len(train), embargo_size, len(test), min_test_samples=30)
        print(get_embargo_message(embargo_size, len(data), len(train), len(test)))
    else:
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

    # --- Feature Scaling (Optional for GNB, but kept for consistency with other ML models) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    # --- Gaussian Naive Bayes Model Definition and Training ---
    model = GaussianNB()

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        try:
            # GNB takes numerical features directly, not binarized for training 'direction'
            model.fit(X_train_scaled, train["direction"])
            print(f"Model fitted successfully: {model}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)  # Exit on critical training error
        print("=" * 50 + "\n")

    # --- Predictions ---
    # GNB predicts classes directly (-1 or 1 in this case)
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    # --- Calculate Hit Ratios ---
    train_hit_ratio = accuracy_score(train["direction"], train_predictions)

    # Store predictions in the 'test' DataFrame
    # For a classification model like GNB, 'predicted_returns' is effectively the predicted direction.
    test["predicted_returns"] = pd.Series(test_predictions, index=test.index)
    test["position"] = test_predictions

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
    # The run_backtest.py script will handle plotting and full risk analysis
    return test, model
