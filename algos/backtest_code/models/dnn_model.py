# algos/backtest_code/models/dnn_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
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


def run_dnn_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "DNN_model",
    lags: int = 5,
    dnn_epochs: int = 50,
    dnn_batch_size: int = 32,
    dnn_hidden_units: int = 16,
    embargo_pct: float = 0.02,
    **kwargs,
) -> tuple:
    """
    Implements the Deep Neural Network (DNN) based trading strategy.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame containing 'returns', 'direction', 'price', etc.
                             It should also have 'annual_trading_periods' in its .attrs.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        lags (int): Number of previous time steps to consider for DNN features.
        dnn_epochs (int): Number of training epochs for the DNN model.
        dnn_batch_size (int): Batch size for DNN model training.
        dnn_hidden_units (int): Number of units in the hidden layers of the DNN.

    Returns:
        tuple: (test_df_results, final_model_object)
               test_df_results: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted Keras DNN model.
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

    # Ensure 'direction' is present (target for DNN)
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

    # Ensure there's enough data in train and test for scaling and DNN input
    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                "Fatal Error: Insufficient data for training or testing DNN. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Scaling features ---
    scaler = StandardScaler()
    train_cols_scaled = scaler.fit_transform(train[cols])
    test_cols_scaled = scaler.transform(test[cols])

    # Remap direction {-1, 1} -> {0, 1} internally for sigmoid + binary_crossentropy.
    train_labels_01 = np.where(train["direction"] == 1, 1, 0)
    test_labels_01 = np.where(test["direction"] == 1, 1, 0)

    # --- DNN Model Definition ---
    # np.random.seed(100)
    # tf.random.set_seed(100)
    model = Sequential()
    model.add(Dense(dnn_hidden_units, activation="relu", input_dim=lags))
    model.add(Dense(dnn_hidden_units, activation="relu"))
    model.add(
        Dense(1, activation="sigmoid")
    )  # Output layer for binary classification (0/1)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} DNN Model Summary:\n")
        model.summary()
        print("=" * 50 + "\n")

    # --- Train the DNN model ---
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training (DNN):\n")
        try:
            history = model.fit(
                train_cols_scaled,
                train_labels_01,
                epochs=dnn_epochs,
                batch_size=dnn_batch_size,
                verbose=0,
                validation_split=0.2,
            )
            print("Training finished.")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            if "val_loss" in history.history:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

            # Evaluate on scaled test data
            loss, accuracy = model.evaluate(test_cols_scaled, test_labels_01, verbose=0)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(
                f"An error occurred during {log_prefix} DNN model training or evaluation: {e}"
            )
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predict and Evaluate ---
    # Convert model predictions (sigmoid 0/1) to {-1, 1} positions for strategy
    train_preds_binary_01 = (
        (model.predict(train_cols_scaled, verbose=0) > 0.5).astype(int).flatten()
    )
    test_preds_binary_01 = (
        (model.predict(test_cols_scaled, verbose=0) > 0.5).astype(int).flatten()
    )

    train_hit_ratio = accuracy_score(train_labels_01, train_preds_binary_01)

    # Convert 0/1 predictions to -1/1 positions for strategy calculation
    test["position"] = np.where(test_preds_binary_01 == 1, 1, -1)

    # Calculate test hit ratio using {-1, 1} direction vs {-1, 1} position
    if not test["position"].isnull().all() and not test.empty:
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = np.nan

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

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

    # Get transaction cost from centralized config
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
