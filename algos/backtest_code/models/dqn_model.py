# algos/backtest_code/models/dqn_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys

# Import common utilities and configuration
from algos.common import config  # For PTC
from algos.common.utils import RedirectStdoutToFile
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)

# Set seeds for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)


def run_dqn_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "DQN_model",
    lags: int = 5,
    dqn_epochs: int = 50,
    dqn_batch_size: int = 32,
    dqn_hidden_units: int = 64,
    embargo_pct: float = 0.02,
    **kwargs,
) -> tuple:
    """
    Implements the DQN-like (Keras MLP) trading strategy.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame containing 'returns', 'direction', 'price', etc.
                             It should also have 'annual_trading_periods' in its .attrs.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        lags (int): Number of previous time steps to consider for features.
        dqn_epochs (int): Number of training epochs for the Keras DQN-like model.
        dqn_batch_size (int): Batch size for Keras DQN-like model training.
        dqn_hidden_units (int): Number of units in the first hidden layer (subsequent layers will scale).

    Returns:
        tuple: (test_df_results, final_model_object)
               test_df_results: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted Keras model.
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

    # Ensure 'direction' is present (target for DQN-like: -1 or 1)
    # The original DQN example used -1 or 1 for the target `direction`
    if (
        "direction" not in data.columns
        or not ((data["direction"] == -1) | (data["direction"] == 1)).all()
    ):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'direction' column missing or not in -1/1 format for {log_prefix}. Calculating/adjusting now."
            )
        data["direction"] = np.where(data["returns"] > 0, 1, -1)

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

    # --- Train-Test Split with Embargo ---
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
                "Fatal Error: Insufficient data for training or testing DQN-like MLP. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Scaling features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    # --- DQN-like (Keras MLP) Model Definition and Training ---
    state_size = len(cols)
    model = Sequential(
        [
            Dense(dqn_hidden_units, activation="relu", input_shape=(state_size,)),
            Dense(
                dqn_hidden_units, activation="relu"
            ),  # <--- CHANGED THIS LINE BACK TO dqn_hidden_units
            Dense(
                1, activation="tanh"
            ),  # Output layer: predicts a value between -1 and 1
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Summary:\n")
        model.summary()
        print("=" * 50 + "\n")

        print(f"\n{log_prefix} Model Training:\n")
        try:
            # Train on the full training set
            model.fit(
                X_train_scaled,
                train["direction"],
                epochs=dqn_epochs,
                batch_size=dqn_batch_size,
                verbose=0,
            )
            print("Training complete.")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predict and Evaluate ---
    train_predictions_continuous = model.predict(X_train_scaled, verbose=0)
    test_predictions_continuous = model.predict(X_test_scaled, verbose=0)

    # Convert continuous predictions from tanh output (-1 to 1) to binary positions (-1 or 1)
    train_predictions_binary = np.sign(train_predictions_continuous).flatten()
    test_predictions_binary = np.sign(test_predictions_continuous).flatten()

    # Calculate Hit Ratios. Ensure actual `direction` matches the -1/1 output for accuracy_score.
    train_hit_ratio = accuracy_score(train["direction"], train_predictions_binary)
    if not test["direction"].empty and len(test_predictions_binary) > 0:
        # Execution-aligned: prediction[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test_predictions_binary[:-1]
        )
    else:
        test_hit_ratio = np.nan

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    # Populate the 'position' column in the test DataFrame
    test["position"] = test_predictions_binary

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

    return test, model  # Return the test results DataFrame and the trained Keras model
