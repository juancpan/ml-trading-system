"""
DQN Model Wrapper for Live Trading System
Handles DQN (Deep Q-Network) models trained with Keras for trading signal generation
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Try to import tensorflow/keras
try:
    import tensorflow as tf
    from tensorflow import keras

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print(
        "Warning: TensorFlow/Keras not installed. Install with: pip install tensorflow"
    )


class DQNModelWrapper:
    """
    Wrapper for DQN models to make them compatible with the live trading system.
    Handles feature preprocessing and signal generation from DQN models.
    """

    def __init__(self, model_path=None, model=None, lags=5):
        """
        Initialize DQN wrapper with model and preprocessing parameters.

        Args:
            model_path: Path to .keras or .h5 model file
            model: Pre-loaded Keras model object
            lags: Number of lag features used in training (default: 5)
        """
        self.logger = logging.getLogger("DQNModelWrapper")
        self.lags = lags
        self.scaler = StandardScaler()
        self._fitted_scaler = False

        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required but not installed")

        if model_path:
            self._model_path = model_path
            try:
                self.model = keras.models.load_model(model_path)
                self.logger.info(f"Loaded DQN model from {model_path}")
            except Exception as e:
                # Try loading without compilation
                self.logger.warning(
                    f"Failed to load model normally, trying with compile=False: {e}"
                )
                self.model = keras.models.load_model(model_path, compile=False)
                self.logger.info(
                    f"Loaded DQN model from {model_path} (without compilation)"
                )
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")

        # Store model metadata
        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape

        # Verify model architecture matches DQN expectations
        self._verify_model_architecture()

    def _verify_model_architecture(self):
        """Verify the model has the expected DQN architecture."""
        expected_input_features = self.lags
        actual_input_features = self.input_shape[-1] if self.input_shape else None

        if actual_input_features != expected_input_features:
            self.logger.warning(
                f"Model input shape {actual_input_features} doesn't match expected lag features {expected_input_features}. "
                f"Will adapt features as needed."
            )

        # Check output layer - should output single value with tanh activation
        if self.output_shape[-1] != 1:
            self.logger.warning(
                f"DQN model output shape is {self.output_shape[-1]}, expected 1. "
                f"Will handle prediction conversion accordingly."
            )

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data matching DQN training process.

        Args:
            df: DataFrame with OHLCV data and 'returns' column

        Returns:
            Feature array ready for DQN model prediction
        """
        # Ensure we have returns
        if "returns" not in df.columns:
            df = df.copy()
            df["returns"] = np.log(df["close"] / df["close"].shift(1))
            df.dropna(inplace=True)

        # Create lag features
        feature_cols = []
        for lag in range(1, self.lags + 1):
            col = f"lag_{lag}"
            df[col] = df["returns"].shift(lag)
            feature_cols.append(col)

        # Drop NaN values created by lag features
        df.dropna(inplace=True)

        if df.empty:
            self.logger.warning("No valid data after creating lag features")
            return np.array([])

        # Extract features
        X = df[feature_cols].values

        # Handle scaler fitting
        if not self._fitted_scaler:
            # Fit scaler on available data (ideally should use training data stats)
            self.scaler.fit(X)
            self._fitted_scaler = True
            self.logger.info("Fitted StandardScaler on feature data")

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, X):
        """
        Generate trading signals from DQN model predictions.

        Args:
            X: Input features (numpy array or pandas DataFrame)

        Returns:
            Array of trading signals (-1, 0, 1)
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            X = self.prepare_features(X)
            if len(X) == 0:
                return np.array([0])  # Return neutral signal if no valid data

        # Ensure X is numpy array
        if hasattr(X, "values"):
            X = X.values

        # Ensure correct shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Ensure we have the right number of features
        expected_features = self.input_shape[-1] if self.input_shape else self.lags
        if X.shape[1] != expected_features:
            self.logger.warning(
                f"Feature mismatch: got {X.shape[1]}, expected {expected_features}. "
                f"Using last {expected_features} features."
            )
            if X.shape[1] > expected_features:
                X = X[:, -expected_features:]
            else:
                # Pad with zeros if not enough features
                padded = np.zeros((X.shape[0], expected_features))
                padded[:, -X.shape[1] :] = X
                X = padded

        # Get predictions from DQN model
        predictions = self.model.predict(X, verbose=0)

        # Return RAW predictions (continuous values from tanh output, range [-1, 1]).
        # Signal conversion is handled by strategy_executor._convert_to_binary_signal().
        # This matches the KerasModelWrapper.predict() design contract.
        return predictions.flatten()

    def _convert_to_signals(self, predictions):
        """
        Convert DQN model predictions to trading signals.

        DQN models with tanh output produce values between -1 and 1.
        We use np.sign to convert to discrete signals.
        """
        # Flatten predictions if needed
        if predictions.shape[-1] == 1:
            predictions = predictions.flatten()

        # Convert continuous predictions to discrete signals
        # DQN uses tanh output, so predictions are already in [-1, 1] range
        signals = np.sign(predictions)

        # Handle near-zero predictions as neutral (optional)
        threshold = 0.1  # Configurable threshold for neutral zone
        signals[np.abs(predictions) < threshold] = 0

        return signals.astype(int)

    def predict_single(self, returns_history):
        """
        Generate a single trading signal from recent returns history.

        Args:
            returns_history: List or array of recent returns (length should be >= lags)

        Returns:
            Single trading signal (-1, 0, or 1)
        """
        if len(returns_history) < self.lags:
            self.logger.warning(
                f"Not enough history: {len(returns_history)} < {self.lags}"
            )
            return 0  # Return neutral signal

        # Take the last 'lags' returns
        features = np.array(returns_history[-self.lags :]).reshape(1, -1)

        # Scale features if scaler is fitted
        if self._fitted_scaler:
            features = self.scaler.transform(features)
        else:
            self.logger.warning("Scaler not fitted, using raw features")

        # Get prediction
        signal = self.predict(features)

        return signal[0] if len(signal) > 0 else 0

    def save_as_pkl(self, output_path):
        """Save the wrapper as a pickle file for compatibility."""
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"Saved DQN wrapper as pickle to {output_path}")

    def __getstate__(self):
        """Custom pickle method to handle Keras model serialization."""
        state = self.__dict__.copy()
        # Save model path and weights instead of the model object
        if hasattr(self, "model"):
            if hasattr(self, "_model_path"):
                state["_model_path"] = self._model_path
            # Save model weights to restore later
            state["_model_weights"] = self.model.get_weights()
            state["_model_config"] = self.model.get_config()
            # Remove the actual model from pickle
            del state["model"]
        return state

    def __setstate__(self, state):
        """Custom unpickle method to reload the Keras model."""
        self.__dict__.update(state)
        # Reload the model
        if hasattr(self, "_model_path") and self._model_path:
            try:
                self.model = keras.models.load_model(self._model_path)
            except:
                self.model = keras.models.load_model(self._model_path, compile=False)
        elif hasattr(self, "_model_config") and hasattr(self, "_model_weights"):
            # Recreate model from config and weights
            self.model = keras.models.Sequential.from_config(self._model_config)
            self.model.set_weights(self._model_weights)
            # Clean up temporary attributes
            del self._model_config
            del self._model_weights


def convert_dqn_to_pkl(keras_model_path, output_pkl_path, lags=5):
    """
    Utility function to convert a DQN Keras model to pickle format.

    Args:
        keras_model_path: Path to .keras or .h5 file
        output_pkl_path: Where to save the pickle file
        lags: Number of lag features used in model training

    Example:
        convert_dqn_to_pkl('models/dqn_model.keras',
                          'strategy_models/upro_dqn_model.pkl',
                          lags=5)
    """
    wrapper = DQNModelWrapper(model_path=keras_model_path, lags=lags)
    wrapper.save_as_pkl(output_pkl_path)
    print(f"Successfully converted DQN model {keras_model_path} to {output_pkl_path}")
    return wrapper


# Test functionality
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Convert model specified as command line argument
        model_path = sys.argv[1]
        output_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else model_path.replace(".keras", ".pkl").replace(".h5", ".pkl")
        )
        lags = int(sys.argv[3]) if len(sys.argv) > 3 else 5

        convert_dqn_to_pkl(model_path, output_path, lags)
    else:
        print("Usage: python dqn_model_wrapper.py <model_path> [output_path] [lags]")
        print("\nExample test:")

        if KERAS_AVAILABLE:
            # Create a simple DQN-like model for testing
            model = keras.Sequential(
                [
                    keras.layers.Dense(64, activation="relu", input_shape=(5,)),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(1, activation="tanh"),  # DQN output
                ]
            )

            model.compile(optimizer="adam", loss="mse")

            # Save and convert
            model.save("test_dqn.keras")
            wrapper = DQNModelWrapper(model_path="test_dqn.keras", lags=5)

            # Test prediction
            test_returns = np.random.randn(10, 5) * 0.01  # 10 samples, 5 features
            signals = wrapper.predict(test_returns)
            print(f"Test predictions: {signals}")

            # Clean up
            import os

            os.remove("test_dqn.keras")
            print("Test completed successfully!")
