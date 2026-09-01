# algos/backtest_code/models/sklearn_dnn_model.py

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys

# Import common utilities and configuration
from algos.common import config  # For PTC
from algos.common.utils import RedirectStdoutToFile

# Set seeds for reproducibility
np.random.seed(1000)


def run_sklearn_dnn_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "SKLEARN_DNN_model",
    lags: int = 5,
    hidden_layer_sizes: tuple = (100, 50, 25),
    mlp_activation: str = "relu",
    mlp_solver: str = "adam",
    mlp_alpha: float = 0.0005,
    mlp_max_iter: int = 500,
    mlp_tol: float = 1e-4,
    mlp_early_stopping: bool = True,
    mlp_validation_fraction: float = 0.1,
    mlp_n_iter_no_change: int = 10,
    **kwargs,
) -> tuple:
    """
    Implements the Scikit-learn Deep Neural Network (MLPClassifier) based trading strategy.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame containing 'returns', 'direction', 'price', etc.
                             It should also have 'annual_trading_periods' in its .attrs.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        lags (int): Number of previous time steps to consider for features.
        hidden_layer_sizes (tuple): Tuple defining the architecture of the hidden layers.
        mlp_activation (str): Activation function for the hidden layers.
        mlp_solver (str): Solver for weight optimization.
        mlp_alpha (float): L2 penalty (regularization term) parameter.
        mlp_max_iter (int): Maximum number of iterations (epochs).
        mlp_tol (float): Tolerance for the optimization.
        mlp_early_stopping (bool): Whether to use early stopping.
        mlp_validation_fraction (float): Fraction of training data to set aside for validation.
        mlp_n_iter_no_change (int): Number of iterations with no improvement to wait before stopping.

    Returns:
        tuple: (test_df_results, final_model_object)
               test_df_results: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted scikit-learn MLPClassifier model.
    """
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    # Ensure 'returns' column is present
    if "returns" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'returns' column missing in data for {log_prefix}. Calculating now."
            )
        data["returns"] = np.log(data["price"] / data["price"].shift(1))
        data.dropna(inplace=True)

    # Ensure 'direction' is present (target for MLPClassifier: -1 or 1)
    if "direction" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'direction' column missing in data for {log_prefix}. Calculating now."
            )
        data["direction"] = np.where(data["returns"] > 0, 1, -1)
    else:
        # Preserve upstream direction {-1, 1} encoding.
        # Preserves carry-adjusted direction from data_loader for forex tickers.
        pass

    # --- Feature Engineering (Lags) ---
    cols = []
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        data[col] = data["returns"].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)

    # Ensure there's still data after dropping NaNs due to lags
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Data is empty after creating lags for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train-Test Split ---
    split = int(len(data) * initial_train_split_ratio)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    # Ensure there's enough data in train and test for MLP input
    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                "Fatal Error: Insufficient data for training or testing MLP. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Scaling features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    y_train = train["direction"]
    y_test = test["direction"]

    # --- MLPClassifier Model Definition and Training ---
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} MLPClassifier Model Training:\n")
        mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=mlp_activation,
            solver=mlp_solver,
            alpha=mlp_alpha,
            batch_size="auto",  # scikit-learn handles this, can be fixed if needed
            learning_rate_init=0.001,
            max_iter=mlp_max_iter,
            tol=mlp_tol,
            random_state=1000,  # Use fixed seed for reproducibility
            verbose=True,  # Keep verbose for log
            early_stopping=mlp_early_stopping,
            validation_fraction=mlp_validation_fraction,
            n_iter_no_change=mlp_n_iter_no_change,
        )
        try:
            mlp_model.fit(X_train_scaled, y_train)
            print(f"\n{log_prefix} MLPClassifier Training Complete.")
            print(f"Number of iterations: {mlp_model.n_iter_}")
            print(f"Best validation score: {mlp_model.best_validation_score_:.4f}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} MLPClassifier training: {e}")
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predict and Evaluate ---
    train_predictions = mlp_model.predict(X_train_scaled)
    test_predictions = mlp_model.predict(X_test_scaled)

    train_hit_ratio = accuracy_score(y_train, train_predictions)

    if not y_test.empty and len(test_predictions) > 0:
        # Execution-aligned: prediction[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(y_test[1:], test_predictions[:-1])
    else:
        test_hit_ratio = np.nan

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    # Populate the 'position' column in the test DataFrame ({-1, 1} for strategy)
    test["position"] = np.where(test_predictions == 1, 1, -1)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Calculate Strategy Returns with Transaction Costs ---
    test.dropna(subset=["position", "returns"], inplace=True)
    if test.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'test' DataFrame is empty after dropping NaNs for strategy calculation for {log_prefix}. Exiting."
            )
        sys.exit(1)

    test["strategy"] = test["position"].shift(1) * test["returns"]

    ptc = config.PTC
    transaction_cost_log_impact = np.log(1 - ptc)

    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff() != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    return (
        test,
        mlp_model,
    )  # Return the test results DataFrame and the trained scikit-learn model
