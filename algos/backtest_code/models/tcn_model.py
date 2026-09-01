import numpy as np
import pandas as pd
import sys  # For sys.exit

# --- Conditional imports for TensorFlow and TCN
# These are essential for the TCN model, so we'll check for their presence.
_HAS_TCN_LIBS = False
try:
    import tensorflow as tf
    from keras.models import Model  # Model is in models for TensorFlow 2.x
    from keras.layers import (
        Input,
        Dense,
    )  # Input and Dense are in layers for TensorFlow 2.x
    from tcn import TCN  # You'll need to pip install keras-tcn

    _HAS_TCN_LIBS = True
    # Set TF seed here for reproducibility in the TCN model
    tf.random.set_seed(1000)
except ImportError as e:
    print(
        f"Warning: Missing required deep learning libraries (TensorFlow/Keras/TCN). TCN model functionality will be skipped. Error: {e}",
        file=sys.stderr,
    )

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)


def run_tcn_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    tcn_nb_filters: int = 64,
    tcn_kernel_size: int = 2,
    tcn_dilations: list = None,
    tcn_dropout_rate: float = 0.2,
    epochs_per_step: int = 5,
    batch_size: int = 32,
    refit_interval: int = 20,
    log_prefix: str = "TCN_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements a Temporal Convolutional Network (TCN) based trading strategy.
    The TCN model predicts the next period's return, and these continuous predictions
    are converted into binary trading signals (long/short).
    It uses a walk-forward validation approach with periodic retraining.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
        initial_train_split_ratio (float): Ratio for the initial training data size before walk-forward.
        lags (int): Number of previous time steps (lagged returns) to use as input features for the TCN.
        tcn_nb_filters (int): Number of filters in the TCN layers.
        tcn_kernel_size (int): Kernel size for the convolutional layers in TCN.
        tcn_dilations (list): List of dilation rates for the TCN layers. Defaults to [1, 2, 4, 8].
        tcn_dropout_rate (float): Dropout rate for regularization in TCN layers.
        epochs_per_step (int): Number of training epochs for each TCN fit/refit step.
        batch_size (int): Batch size for TCN training.
        refit_interval (int): Number of test steps after which the TCN model is re-trained.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last fitted Keras TCN model object. Returns None if TCN libs are missing.
    """
    # --- Check for TCN library availability early ---
    if not _HAS_TCN_LIBS:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: TensorFlow/Keras/TCN libraries not found. Cannot run {log_prefix} strategy. Exiting."
            )
        return pd.DataFrame(), None  # Return empty DataFrame and None model

    if tcn_dilations is None:
        tcn_dilations = [1, 2, 4, 8]

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
    cols = []
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

    # --- Prepare data for TCN input format ---
    # TCN expects input shape (samples, timesteps, features)
    timesteps = lags  # Each 'lag' is a timestep in the sequence
    features = 1  # Each timestep has one feature: the lagged return value

    X_data_raw = data[cols].values  # Features are the lagged returns
    y_data_raw = data["returns"].values  # Target is the actual return

    # --- Train/Test Split with Embargo ---
    split = int(len(data) * initial_train_split_ratio)
    if split == 0 or split >= len(data):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Invalid split ratio or insufficient data for {log_prefix}. Split results in empty train/test. Exiting."
            )
        sys.exit(1)

    if embargo_pct > 0:
        embargo_size = calculate_embargo_size(len(data), embargo_pct, min_samples=5)

        X_train_raw = X_data_raw[:split]
        y_train_raw = y_data_raw[:split]
        X_test_raw = X_data_raw[split + embargo_size :]
        y_test_actual = y_data_raw[split + embargo_size :]

        # Validate
        validate_embargo_split(
            len(X_train_raw), embargo_size, len(X_test_raw), min_test_samples=30
        )

        # Log embargo
        print(
            get_embargo_message(
                embargo_size, len(data), len(X_train_raw), len(X_test_raw)
            )
        )

        # Align test DataFrame with embargoged test data
        test_df_original_index = data.iloc[split + embargo_size :].index
        test = pd.DataFrame(index=test_df_original_index)
        test["returns"] = data["returns"].iloc[split + embargo_size :]
        test["direction"] = data["direction"].iloc[split + embargo_size :]
    else:
        X_train_raw = X_data_raw[:split]
        y_train_raw = y_data_raw[:split]
        X_test_raw = X_data_raw[split:]
        y_test_actual = y_data_raw[split:]

        print(
            "\nNote: Embargo disabled (embargo_pct=0). Using simple train/test split."
        )

        # Align test DataFrame with actual test data (after drops from lags)
        test_df_original_index = data.iloc[split:].index
        test = pd.DataFrame(index=test_df_original_index)
        test["returns"] = data["returns"].iloc[split:]
        test["direction"] = data["direction"].iloc[split:]

    if X_train_raw.shape[0] == 0 or X_test_raw.shape[0] == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Train or test set is empty after splitting for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # --- Reshape data for TCN input (samples, timesteps, features) ---
    X_train_reshaped = X_train_scaled.reshape(
        (X_train_scaled.shape[0], timesteps, features)
    )
    X_test_reshaped = X_test_scaled.reshape(
        (X_test_scaled.shape[0], timesteps, features)
    )

    # --- Build the TCN Model ---
    inputs = Input(shape=(timesteps, features))
    tcn_layer = TCN(
        nb_filters=tcn_nb_filters,
        kernel_size=tcn_kernel_size,
        dilations=tcn_dilations,
        return_sequences=False,  # Output a single vector per sequence
        activation="relu",
        use_batch_norm=False,
        dropout_rate=tcn_dropout_rate,
    )(inputs)

    outputs = Dense(1, activation="linear")(
        tcn_layer
    )  # Output is a single continuous return prediction
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="mse"
    )  # Adam optimizer, Mean Squared Error loss for regression

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Summary:\n")
        model.summary()
        print("=" * 50 + "\n")

    # --- Walk-Forward Validation for TCN ---
    predictions_list = []
    final_model = None  # To store the last fitted model for returning

    # Initial training data for the first iteration of walk-forward
    history_X = X_train_reshaped.tolist()
    history_y = y_train_raw.tolist()

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training and Forecasting (Walk-Forward):\n")

        # Initial training of the model before the walk-forward loop begins
        try:
            print(
                f"Initial training of {log_prefix} model on {len(history_X)} samples..."
            )
            model.fit(
                np.array(history_X),
                np.array(history_y),
                epochs=epochs_per_step,
                batch_size=batch_size,
                verbose=0,
            )
            print("Initial training complete.")
        except Exception as e:
            print(
                f"Fatal Error: Initial {log_prefix} model training failed: {e}. Exiting."
            )
            sys.exit(1)

        for t in range(len(X_test_reshaped)):
            # Get the current sample for prediction (needs to be 3D: 1 sample, timesteps, features)
            current_X_sample = X_test_reshaped[[t]]

            try:
                # Predict the next return (continuous value)
                yhat_continuous = model.predict(current_X_sample, verbose=0)[0, 0]
                predictions_list.append(yhat_continuous)
            except Exception as e:
                print(
                    f"Warning: {log_prefix} prediction failed at step {t} with error: {e}. Appending NaN."
                )
                predictions_list.append(np.nan)

            # Add the actual observation from the test set to the history for the next iteration
            history_X.append(X_test_reshaped[t].tolist())
            history_y.append(y_test_actual[t])

            # Periodically refit the model
            if (t + 1) % refit_interval == 0:
                try:
                    print(
                        f"Refitting {log_prefix} model at step {t + 1} with {len(history_X)} samples..."
                    )
                    # Re-compile the model to reset optimizer states if needed (good practice for refitting)
                    model.compile(optimizer="adam", loss="mse")
                    model.fit(
                        np.array(history_X),
                        np.array(history_y),
                        epochs=epochs_per_step,
                        batch_size=batch_size,
                        verbose=0,
                    )
                    print("Refitting complete.")
                    final_model = model  # Store the last successfully fitted model
                except Exception as e:
                    print(
                        f"Warning: {log_prefix} model refitting failed at step {t + 1}: {e}. Continuing without refit."
                    )
                    final_model = None  # Clear model if refit fails

        # Store the model after the last iteration, if it was successfully fitted
        if model is not None:
            final_model = model

        # Align predictions with the test DataFrame. It's crucial that predictions_list length matches the test index size.
        # This slice ensures correct alignment even if some predictions were NaN and dropped later.
        test["predicted_returns"] = pd.Series(
            predictions_list, index=test.index[: len(predictions_list)]
        )
        test.dropna(subset=["predicted_returns"], inplace=True)

        print("=" * 50 + "\n")

    # --- Convert Predicted Returns to Trading Signals (Position) ---
    # Using np.sign for consistency across all models
    test["position"] = np.sign(test["predicted_returns"].values).flatten()
    # Handle edge case where sign might return 0 (for exactly 0 prediction)
    test["position"] = np.where(test["position"] == 0, 1, test["position"])

    # Ensure the 'direction' column is correctly aligned and present for accuracy_score
    # This was already done during initial test df creation, but re-checking after drops.
    test.dropna(subset=["direction", "position"], inplace=True)

    # --- Calculate Hit Ratios ---
    # For walk-forward, "training hit ratio" isn't directly applicable in the same way,
    # as the model is continually retrained on expanding window. We only assess performance on the 'test' set.
    train_hit_ratio = np.nan  # N/A for this approach

    if not test.empty and "direction" in test.columns and "position" in test.columns:
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = (
            np.nan
        )  # Cannot calculate if test set is empty or columns are missing

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio} (N/A for walk-forward TCN)")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

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

    # Apply transaction cost only when the position changes.
    # `fillna(0)` handles the very first `.diff()` value, which is NaN.
    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
    return test, final_model
