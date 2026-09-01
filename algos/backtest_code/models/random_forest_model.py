import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle  # For saving/loading the scikit-learn model
import sys  # For sys.exit

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)


def run_random_forest_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
    log_prefix: str = "RandomForest_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements a Random Forest Classifier-based trading strategy. The model predicts
    the 'direction' of returns (1 for up, -1 for down) and uses these predictions
    as trading signals.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        lags (int): Number of previous time steps (lagged returns) to use as features.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Controls the randomness of the bootstrapping of the samples
                            and the features used when building trees.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last fitted RandomForestClassifier model.
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
    cols = []  # List to store column names of lagged features
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
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

    # --- Feature Scaling ---
    # StandardScaler is generally not strictly necessary for tree-based models like Random Forests,
    # as they are not sensitive to the scale of features. However, it doesn't hurt performance
    # and ensures consistency if you're switching between models. I'll keep it as it was in the original.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    # --- RandomForestClassifier Model Definition and Training ---
    # `class_weight='balanced'` can be useful if one class (e.g., up moves) is much rarer than another.
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
    )

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        try:
            model.fit(X_train_scaled, train["direction"])
            print(f"Model fitted successfully: {model}")
            # RandomForestClassifier doesn't have simple coefficients like linear models,
            # but you can inspect feature importances:
            print(f"Feature Importances: {model.feature_importances_}")
            print(f"Features (Lags): {cols}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)  # Exit on critical training error
        print("=" * 50 + "\n")

    # --- Predictions ---
    train_predictions_binary = model.predict(X_train_scaled)
    test_predictions_binary = model.predict(X_test_scaled)

    # The 'position' column holds the trading signal
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
