import numpy as np
import pandas as pd
import sys  # For sys.exit
import pickle  # For saving/loading the scikit-learn model

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
)  # For hyperparameter tuning and time-series CV

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage


def run_svm_tuned_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    cv_splits: int = 5,
    log_prefix: str = "SVM_tuned",
    **kwargs,
):
    """
    Implements an SVM (Support Vector Classifier) based trading strategy.
    It includes feature engineering with lagged returns and technical indicators,
    hyperparameter tuning using GridSearchCV with TimeSeriesSplit, and
    predicts the direction of returns.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also contain 'price' for indicator calculation.
        initial_train_split_ratio (float): Ratio for the initial training data size.
        lags (int): Number of lagged returns to use as features.
        cv_splits (int): Number of splits for TimeSeriesSplit cross-validation during GridSearchCV.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The best fitted sklearn.svm.SVC model object from GridSearchCV.
    """
    # --- Input Data Validation ---
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    if (
        "returns" not in data.columns
        or "direction" not in data.columns
        or "price" not in data.columns
    ):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data for {log_prefix} is missing 'returns', 'direction', or 'price' columns. Exiting."
            )
        sys.exit(1)

    # --- Feature Engineering for SVM ---
    features = []

    # Lagged returns
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        data[col] = data["returns"].shift(lag)
        features.append(col)

    # Add more technical indicators
    # Ensure window sizes are less than or equal to the total data length
    # and adjust them as needed to avoid introducing too many NaNs early.
    data["SMA_10"] = (
        data["price"].rolling(window=min(10, len(data)), min_periods=1).mean()
    )
    data["SMA_30"] = (
        data["price"].rolling(window=min(30, len(data)), min_periods=1).mean()
    )
    data["EMA_10"] = (
        data["price"].ewm(span=min(10, len(data)), adjust=False, min_periods=1).mean()
    )

    # Volatility (Standard Deviation of returns)
    data["volatility_10"] = (
        data["returns"].rolling(window=min(10, len(data)), min_periods=1).std()
    )
    data["volatility_30"] = (
        data["returns"].rolling(window=min(30, len(data)), min_periods=1).std()
    )

    # Adding new features to `features` list
    features.extend(["SMA_10", "SMA_30", "EMA_10", "volatility_10", "volatility_30"])

    # Drop any rows that now have NaN values due to feature creation (e.g., first few rows for rolling means)
    data.dropna(inplace=True)

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating features and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split ---
    split = int(len(data) * initial_train_split_ratio)
    if split == 0 or split >= len(data):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Invalid split ratio or insufficient data for {log_prefix}. Split results in empty train/test. Exiting."
            )
        sys.exit(1)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    # Ensure feature columns exist in train and test sets
    if (
        not features
        or not all(col in train.columns for col in features)
        or not all(col in test.columns for col in features)
    ):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Missing one or more feature columns in train or test data for {log_prefix}. Exiting."
            )
        sys.exit(1)

    X_train, y_train = train[features], train["direction"]
    X_test, y_test = test[features], test["direction"]

    # --- Scaling features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define SVM model for GridSearchCV
    svm_model = SVC(
        random_state=42, probability=True
    )  # probability=True for consistent behavior, though not strictly needed for just predict

    # Hyperparameter Grid for tuning
    param_grid = {
        "C": [0.1, 1, 10],  # Regularization parameter
        "kernel": ["linear", "rbf"],  # Kernel type
        "gamma": [
            "scale",
            0.1,
            1,
        ],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var())
    }

    # TimeSeriesSplit for cross-validation during tuning
    # n_splits determines the number of splits for the training set.
    # The last split will be the validation set.
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )  # n_jobs=-1 for parallel processing

    best_svm_estimator = None
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nStarting {log_prefix} Hyperparameter Tuning (GridSearchCV)...\n")
        try:
            grid_search.fit(X_train_scaled, y_train)
            print(f"\n{log_prefix} Hyperparameter Tuning Finished.")
            print(f"Best parameters found: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            best_svm_estimator = grid_search.best_estimator_
        except Exception as e:
            print(f"Error during GridSearchCV fit for {log_prefix}: {e}")
            print(f"Using default SVM parameters due to tuning failure.")
            best_svm_estimator = SVC(
                random_state=42, probability=True
            )  # Fallback to default model
            best_svm_estimator.fit(X_train_scaled, y_train)  # Fit the default model
        print("=" * 50 + "\n")

    # --- Model Prediction and Performance Metrics ---
    if best_svm_estimator is None:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: No SVM estimator available after tuning or fallback for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # Predictions on scaled data
    train_predictions = best_svm_estimator.predict(X_train_scaled)
    test_predictions = best_svm_estimator.predict(X_test_scaled)

    # Calculate hit ratios
    train_hit_ratio = accuracy_score(y_train, train_predictions)
    # Execution-aligned: prediction[t] predicts direction[t+1]
    test_hit_ratio = accuracy_score(y_test[1:], test_predictions[:-1])

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    # Add 'position' to the test DataFrame, aligning with its original index
    test["position"] = pd.Series(test_predictions, index=test.index)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Strategy Returns Calculation (shift position by 1 bar to avoid look-ahead bias) ---
    test["strategy"] = test["position"].shift(1) * test["returns"]

    # --- Apply Transaction Costs ---
    # PTC is imported from algos.common.config
    transaction_cost_log_impact = np.log(1 - PTC)

    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
    return test, best_svm_estimator
