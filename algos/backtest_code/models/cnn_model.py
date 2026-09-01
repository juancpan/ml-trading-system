# algos/backtest_code/models/cnn_model.py

# Normal imports
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys  # For sys.exit

# Import common utilities and configuration
from algos.common import config  # For PTC and potentially RF_RATE if needed directly
from algos.common.utils import (
    RedirectStdoutToFile,
)  # For internal logging within this function
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)

# Set seeds for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)


def run_cnn_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "CNN_model",
    lags: int = 5,
    cnn_epochs: int = 50,
    cnn_batch_size: int = 32,
    embargo_pct: float = 0.02,
    **kwargs,
) -> tuple:
    """
    Implements the CNN-based trading strategy.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame containing 'returns', 'direction', 'price', etc.
                             It should also have 'annual_trading_periods' in its .attrs.
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.
        lags (int): Number of previous time steps to consider for CNN features.
        cnn_epochs (int): Number of epochs for CNN training.
        cnn_batch_size (int): Batch size for CNN training.

    Returns:
        tuple: (test_df_results, final_model_object)
               test_df_results: DataFrame with 'predicted_returns', 'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted Keras CNN model.
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

    # Ensure 'direction' is present
    if "direction" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'direction' column missing in data for {log_prefix}. Calculating now."
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

    # Ensure there's enough data in train and test for scaling and CNN input
    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                "Fatal Error: Insufficient data for training or testing CNN. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Scaling features ---
    scaler = StandardScaler()
    train_cols_scaled = scaler.fit_transform(train[cols])
    test_cols_scaled = scaler.transform(test[cols])

    # --- Reshape data for 1D CNN input: (samples, timesteps, features) ---
    X_train_cnn = train_cols_scaled.reshape(
        train_cols_scaled.shape[0], train_cols_scaled.shape[1], 1
    )
    y_train_cnn = train["direction"].values.reshape(-1, 1)  # Use direction for target

    X_test_cnn = test_cols_scaled.reshape(
        test_cols_scaled.shape[0], test_cols_scaled.shape[1], 1
    )
    y_test_cnn = test["direction"].values.reshape(-1, 1)  # Use direction for target

    # --- 1D CNN Model Definition ---
    model = Sequential()
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]),
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=10, activation="relu"))
    model.add(
        Dense(units=1, activation="linear")
    )  # Linear activation for regression-like output

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} CNN Model Summary:\n")
        model.summary()
        print("=" * 50 + "\n")

    # --- Train the CNN model ---
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training (CNN):\n")
        try:
            history = model.fit(
                X_train_cnn,
                y_train_cnn,
                epochs=cnn_epochs,
                batch_size=cnn_batch_size,
                verbose=0,
                validation_split=0.2,
            )
            print("Training finished.")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            if "val_loss" in history.history:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        except Exception as e:
            print(f"An error occurred during {log_prefix} CNN model training: {e}")
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predict and Evaluate ---
    # Predict continuous values
    train_predictions_continuous = model.predict(X_train_cnn, verbose=0)
    test_predictions_continuous = model.predict(X_test_cnn, verbose=0)

    # Convert continuous predictions to binary (-1 or 1)
    train_predictions_binary = np.sign(train_predictions_continuous).flatten()
    test_predictions_binary = np.sign(test_predictions_continuous).flatten()

    # Handle cases where np.sign might return 0 for exactly 0 prediction.
    train_predictions_binary[train_predictions_binary == 0] = 1
    test_predictions_binary[test_predictions_binary == 0] = 1

    train_hit_ratio = accuracy_score(train["direction"], train_predictions_binary)
    test["position"] = test_predictions_binary

    # Ensure test['position'] is not all NaN/empty before calculating accuracy
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
    # Filter out rows where 'position' or 'returns' might be NaN after operations
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

    # The rest of the plotting, market benchmark, strategy performance, and risk analysis
    # will be handled by the run_backtest.py script and common functions.

    return test, model  # Return the test results DataFrame and the trained Keras model
