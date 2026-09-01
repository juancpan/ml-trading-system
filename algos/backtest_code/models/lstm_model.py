import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys  # For sys.exit
# import os
# import random
# import torch # For PyTorch seed setting

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)

# Set TensorFlow seed for reproducibility


def run_lstm_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    lstm_units: int = 50,
    log_prefix: str = "LSTM_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements an LSTM-based trading strategy. The model predicts the 'direction' of returns
    (-1 or 1) based on lagged returns, and these predictions are used as trading signals.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        lags (int): Number of previous time steps (lagged returns) to use as features for the LSTM.
        epochs (int): Number of training epochs for the LSTM model.
        batch_size (int): Batch size for LSTM training.
        lstm_units (int): Number of units (memory cells) in the LSTM layer.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last fitted Keras LSTM model.
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
    scaler = StandardScaler()
    train_cols_scaled = scaler.fit_transform(train[cols])
    test_cols_scaled = scaler.transform(test[cols])

    # --- Reshape data for LSTM input: (samples, timesteps, features) ---
    # Each lag becomes a timestep, and there's 1 feature per timestep (the lagged return value)
    if train_cols_scaled.size == 0 or test_cols_scaled.size == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Scaled training or testing data is empty, cannot reshape for {log_prefix}. Exiting."
            )
        sys.exit(1)

    X_train_lstm = train_cols_scaled.reshape(
        train_cols_scaled.shape[0], train_cols_scaled.shape[1], 1
    )
    y_train_lstm = train["direction"].values.reshape(-1, 1)  # Ensure y is 2D for Keras

    X_test_lstm = test_cols_scaled.reshape(
        test_cols_scaled.shape[0], test_cols_scaled.shape[1], 1
    )
    y_test_lstm = test["direction"].values.reshape(-1, 1)  # Ensure y is 2D for Keras

    # --- LSTM Model Definition ---
    model = models.Sequential()
    model.add(layers.LSTM(units=lstm_units, activation="relu", input_shape=(lags, 1)))
    # Output layer: 1 unit with 'linear' activation if predicting a continuous score
    # which is then binarized by np.sign. If targets were 0/1, 'sigmoid' would be used.
    model.add(layers.Dense(units=1, activation="linear"))

    # Compile the model
    # 'mse' loss for regression-like output, 'adam' optimizer is a good default.
    model.compile(optimizer="adam", loss="mse")

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Summary:\n")
        model.summary()
        print("=" * 50 + "\n")

    # --- Train the LSTM model ---
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        try:
            history = model.fit(
                X_train_lstm,
                y_train_lstm,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.2,
            )
            print("Training finished.")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            if "val_loss" in history.history:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predictions ---
    # Predict continuous values, then binarize for trading signals
    train_predictions_continuous = model.predict(X_train_lstm, verbose=0)
    test_predictions_continuous = model.predict(X_test_lstm, verbose=0)

    # Convert continuous predictions to binary (-1 or 1)
    train_predictions_binary = np.sign(train_predictions_continuous).flatten()
    test_predictions_binary = np.sign(test_predictions_continuous).flatten()

    # Handle cases where np.sign might return 0 for exactly 0 prediction (unlikely but good practice)
    train_predictions_binary[train_predictions_binary == 0] = (
        1  # Default to 1 if prediction is exactly 0
    )
    test_predictions_binary[test_predictions_binary == 0] = 1

    # --- Calculate Hit Ratios ---
    train_hit_ratio = accuracy_score(train["direction"], train_predictions_binary)

    if not test["direction"].empty and len(test_predictions_binary) == len(
        test["direction"]
    ):
        test["position"] = test_predictions_binary
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = (
            np.nan
        )  # Cannot calculate if test set is empty or predictions mismatch

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
