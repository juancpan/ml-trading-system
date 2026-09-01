"""
Optimized LSTM trading strategy model with GPU support and batch processing.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)
import sys
from typing import Tuple, Optional


class OptimizedLSTMModel:
    """
    Optimized LSTM model with improved architecture and training efficiency.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        Initialize LSTM model with optimized architecture.

        Args:
            input_shape: (sequence_length, n_features)
            units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.model = self._build_model(input_shape, units, dropout_rate, learning_rate)
        self.scaler = StandardScaler()

    def _build_model(
        self,
        input_shape: Tuple[int, int],
        units: int,
        dropout_rate: float,
        learning_rate: float,
    ) -> models.Model:
        """Build optimized LSTM architecture."""
        # Use functional API for flexibility
        inputs = layers.Input(shape=input_shape)

        # First LSTM layer with return sequences
        x = layers.LSTM(
            units,
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            recurrent_regularizer=l1_l2(l1=0.01, l2=0.01),
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        # Second LSTM layer
        x = layers.LSTM(units // 2, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        # Dense layers
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

        # Output layer for classification
        outputs = layers.Dense(3, activation="softmax")(x)  # 3 classes: buy, hold, sell

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Use optimized optimizer
        optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 20,
        train_split_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training with efficient vectorization.

        Buy/sell thresholds are adaptive: we use half the standard deviation
        of returns, floored at 1e-8 to avoid degeneracy.  This prevents
        ultra-low-volatility instruments (e.g. T-bill ETFs like BIL, where
        daily returns are ~0.0002) from collapsing into 100% "hold" labels,
        which makes the LSTM learn nothing and ultimately crashes the metrics
        pipeline.  For typical equities (std ~0.01-0.03), 0.5*std gives
        thresholds of 0.005-0.015 -- comparable to the old hardcoded 0.001
        but properly scaled.
        """
        # Calculate returns
        returns = data["returns"].values

        # Adaptive threshold: 0.5 * std(train returns only)
        # Use only training portion to avoid leaking test-period volatility
        # into the label construction.
        train_end = int(len(returns) * train_split_ratio)
        returns_std = np.nanstd(returns[:train_end])
        threshold = max(1e-8, 0.5 * returns_std)

        # Create sequences efficiently using stride tricks
        n_samples = len(returns) - sequence_length
        sequences = np.zeros((n_samples, sequence_length))
        labels = np.zeros(n_samples)

        for i in range(n_samples):
            sequences[i] = returns[i : i + sequence_length]
            # Convert direction to 0, 1, 2 (sell, hold, buy)
            next_return = returns[i + sequence_length]
            if next_return > threshold:  # Buy threshold
                labels[i] = 2
            elif next_return < -threshold:  # Sell threshold
                labels[i] = 0
            else:  # Hold
                labels[i] = 1

        # Reshape for LSTM (samples, timesteps, features)
        sequences = sequences.reshape((n_samples, sequence_length, 1))

        return sequences, labels

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> None:
        """
        Train LSTM with early stopping and learning rate reduction.

        Uses inverse-frequency class weighting to prevent majority-class
        collapse on imbalanced label distributions (e.g. high-kurtosis
        assets like GLD where the "hold" class dominates — see the H2
        2026 GLD training incident: with the symmetric ±0.5σ threshold,
        62% of GLD labels were "hold", and without class weighting the
        softmax heavily biased toward hold, making argmax(pred) == hold
        on every day and producing position=0 in backtest output).

        Args:
            X_train: Training features.
            y_train: Training labels.
            validation_data: Explicit (X_val, y_val) tuple for temporal validation.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            verbose: Verbosity level.
        """
        # Define callbacks for optimization
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        # Model checkpoint -- use a unique temp file per process to avoid
        # HDF5 file lock contention when multiple LSTM models train concurrently.
        import tempfile

        checkpoint_file = tempfile.NamedTemporaryFile(
            suffix=".weights.h5", prefix="lstm_ckpt_", delete=False
        )
        checkpoint_path = checkpoint_file.name
        checkpoint_file.close()

        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        # Compute inverse-frequency class weights on the training set so
        # imbalanced 3-class label distributions (sell/hold/buy) do not
        # collapse the model into predicting the majority class.
        # Equivalent to sklearn.utils.class_weight.compute_class_weight
        # with class_weight='balanced' but computed inline to avoid the
        # extra import and match the exact keras fit() signature.
        unique_classes, class_counts = np.unique(
            y_train.astype(int), return_counts=True
        )
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        class_weight_dict = {
            int(cls): float(n_samples / (n_classes * cnt))
            for cls, cnt in zip(unique_classes, class_counts)
        }
        # Ensure all 3 classes have a weight even if absent from training
        # (rare but possible on very skewed tickers). Assign the mean
        # weight so the softmax head stays well-defined.
        mean_weight = float(np.mean(list(class_weight_dict.values())))
        for cls in (0, 1, 2):
            class_weight_dict.setdefault(cls, mean_weight)

        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            class_weight=class_weight_dict,
            verbose=verbose,
            # use_multiprocessing=True,
            # workers=4
        )

        # Clean up temp checkpoint file
        try:
            import os

            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
        except OSError:
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return RAW model predictions (continuous values).

        For 3-class softmax output, returns the signed confidence:
          positive = model leans bullish, negative = model leans bearish.
        The magnitude indicates confidence (e.g., 0.95 vs 0.51).

        Signal conversion is handled by strategy_executor._convert_to_binary_signal().
        """
        predictions = self.model.predict(X, batch_size=128)
        if predictions.shape[-1] == 3:
            # 3-class softmax: [sell_prob, hold_prob, buy_prob]
            # Return buy_prob - sell_prob as signed confidence in [-1, 1]
            return predictions[:, 2] - predictions[:, 0]
        elif predictions.shape[-1] == 1:
            # Single output (regression or sigmoid)
            return predictions.flatten()
        else:
            # Fallback: return as-is
            return predictions.flatten()

    def predict_signals(self, X: np.ndarray) -> np.ndarray:
        """Convert predictions to discrete trading signals {-1, 0, 1}.

        Used by backtesting code where discrete positions are needed directly.
        Live trading should use predict() + strategy_executor._convert_to_binary_signal().
        """
        predictions = self.model.predict(X, batch_size=128)
        return np.argmax(predictions, axis=1) - 1

    # -- Pickle support -------------------------------------------------------
    # Keras models are NOT natively picklable.  When persistence.py serialises
    # this wrapper via ``pickle.dump(model_obj, f)`` we must strip the Keras
    # model and save its weights in-band so that the .pkl is self-contained and
    # portable across environments (no file-path dependency).

    def __getstate__(self):
        """Exclude the Keras model; store its weights + config instead."""
        import tempfile
        import os

        state = self.__dict__.copy()
        if "model" in state and state["model"] is not None:
            keras_model = state.pop("model")
            # Persist architecture as JSON so we can rebuild the graph.
            state["_keras_config"] = keras_model.get_config()
            # Keras save_weights requires a real file path ending in .weights.h5.
            # Write to a temp file, read back the bytes, then delete.
            tmp = tempfile.NamedTemporaryFile(
                suffix=".weights.h5", prefix="lstm_pkl_", delete=False
            )
            tmp_path = tmp.name
            tmp.close()
            try:
                keras_model.save_weights(tmp_path)
                with open(tmp_path, "rb") as wf:
                    state["_keras_weights"] = wf.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        # Remove non-picklable training history if present
        state.pop("history", None)
        return state

    def __setstate__(self, state):
        """Reconstruct the Keras model from saved config + weights."""
        import tempfile
        import os

        keras_config = state.pop("_keras_config", None)
        keras_weights_bytes = state.pop("_keras_weights", None)
        self.__dict__.update(state)
        if keras_config is not None:
            self.model = models.Model.from_config(keras_config)
            if keras_weights_bytes is not None:
                # Write bytes to a temp file so Keras can load them.
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".weights.h5", prefix="lstm_pkl_", delete=False
                )
                tmp_path = tmp.name
                tmp.close()
                try:
                    with open(tmp_path, "wb") as wf:
                        wf.write(keras_weights_bytes)
                    self.model.load_weights(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            # Re-compile so predict() works (metrics not needed for inference)
            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
            )
        else:
            self.model = None


def run_lstm_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    sequence_length: int = 20,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    log_prefix: str = "LSTM_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Optimized LSTM trading strategy implementation with embargo.

    Note: LSTM sequences create a natural buffer of `sequence_length` samples.
    The embargo is applied ON TOP of this buffer for additional protection.
    """
    # Enable GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    # Input validation
    if data.empty or "returns" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Fatal Error: Invalid input data for {log_prefix}")
        sys.exit(1)

    # Initialize model
    lstm_model = OptimizedLSTMModel(
        input_shape=(sequence_length, 1),
        units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    # Prepare sequences (pass train_split_ratio to avoid test-data leakage in threshold)
    X, y = lstm_model.prepare_sequences(
        data, sequence_length, train_split_ratio=initial_train_split_ratio
    )

    if len(X) < 100:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Fatal Error: Insufficient data for LSTM training")
        sys.exit(1)

    # Train/test split with embargo
    split = int(len(X) * initial_train_split_ratio)

    if embargo_pct > 0:
        # Calculate embargo size
        embargo_size = calculate_embargo_size(
            len(X),
            embargo_pct=embargo_pct,
            min_samples=max(5, sequence_length // 4),  # At least 1/4 of sequence length
        )

        # Apply embargo
        X_train, X_test = X[:split], X[split + embargo_size :]
        y_train, y_test = y[:split], y[split + embargo_size :]

        # Validate
        validate_embargo_split(
            len(X_train), embargo_size, len(X_test), min_test_samples=30
        )

        # Log embargo (note: LSTM has natural buffer from sequence_length)
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"\nLSTM Embargo Configuration:")
            print(f"  Sequence buffer: {sequence_length} samples (natural from LSTM)")
            print(f"  Embargo buffer: {embargo_size} samples (additional protection)")
            print(f"  Total gap: {sequence_length + embargo_size} samples")
            print(get_embargo_message(embargo_size, len(X), len(X_train), len(X_test)))
    else:
        # No embargo
        embargo_size = 0
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(
            "\nNote: Embargo disabled (embargo_pct=0). Using simple train/test split."
        )

    # Scale features
    X_train_scaled = lstm_model.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(
        X_train.shape
    )
    X_test_scaled = lstm_model.scaler.transform(X_test.reshape(-1, 1)).reshape(
        X_test.shape
    )

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nTraining Optimized LSTM Model ({log_prefix}):\n")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Sequence length: {sequence_length}")
        print(f"LSTM units: {lstm_units}")
        print(f"Batch size: {batch_size}")
        print("=" * 50 + "\n")

    # Explicit temporal validation split (last 20% of training data)
    val_size = max(1, int(len(X_train_scaled) * 0.2))
    X_train_fit = X_train_scaled[:-val_size]
    y_train_fit = y_train[:-val_size]
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]

    # Train model
    lstm_model.train(
        X_train_fit,
        y_train_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Make predictions for backtest.
    #
    # CRITICAL: use the same decision function as live trading (see
    # execution/strategy_executor.py::_convert_to_binary_signal, lines
    # 813-828 for lstm/lstm_optimized/dqn/cnn/tcn/rnn/dnn). Live uses
    # `predict()` (signed confidence = P(buy) - P(sell) in [-1, 1]) then
    # np.sign with 0 -> 1. If the backtest instead used argmax on a
    # 3-class softmax (old behaviour via predict_signals), position would
    # be 0 on every day the model's most likely class is "hold" --
    # divergent from what live trading actually does. This caused GLD's
    # 2026-04-21 run to output position=0 on all 10 test days despite the
    # signed-confidence signal being ±1 (see the H2 2026 GLD incident).
    #
    # predict_signals() is kept for callers that explicitly want the
    # 3-class argmax, but it MUST NOT be used for strategy generation.
    raw_confidence = lstm_model.predict(X_test_scaled)
    # np.sign returns 0 when confidence is exactly 0; map those to +1 to
    # match strategy_executor._convert_to_binary_signal (comment on line
    # 823-828 of strategy_executor.py). This also matches the legacy
    # lstm_model.py behaviour: `predictions[predictions == 0] = 1`.
    test_predictions = np.sign(raw_confidence).astype(int)
    test_predictions = np.where(test_predictions == 0, 1, test_predictions)

    # Create test DataFrame with proper indexing
    test_start_idx = split + embargo_size + sequence_length
    test_df = data.iloc[test_start_idx : test_start_idx + len(test_predictions)].copy()
    test_df["position"] = test_predictions

    # Calculate strategy returns (shift position by 1 bar to avoid look-ahead bias)
    test_df["strategy"] = test_df["position"].shift(1) * test_df["returns"]

    # Apply transaction costs
    transaction_cost_log_impact = np.log(1 - PTC)
    test_df["strategy_tc"] = np.where(
        test_df["position"].shift(1).diff().fillna(0) != 0,
        test_df["strategy"] + transaction_cost_log_impact,
        test_df["strategy"],
    )
    # Remove first row NaN from position shift
    test_df.dropna(subset=["strategy"], inplace=True)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("\nTraining Complete!")
        print(
            f"Final training accuracy: {lstm_model.history.history['accuracy'][-1]:.4f}"
        )
        if "val_accuracy" in lstm_model.history.history:
            print(
                f"Final validation accuracy: {lstm_model.history.history['val_accuracy'][-1]:.4f}"
            )

        # Calculate test accuracy (use post-dropna index to align predictions)
        # Execution-aligned: prediction[t] predicts direction[t+1]
        aligned_preds = test_predictions[: len(test_df)]
        test_accuracy = np.mean(
            aligned_preds[:-1] == test_df["direction"].iloc[1:].values
        )
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("=" * 50 + "\n")

    return test_df, lstm_model
