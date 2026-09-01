"""
Keras Model Wrapper for Algo Trading System
Allows .keras and .h5 models to work with the existing trading infrastructure
"""

import numpy as np
import pickle
import logging
from pathlib import Path

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

# Try to import TCN if available
try:
    from tcn import TCN

    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False
    # Not critical - only needed if loading TCN models


class KerasModelWrapper:
    """
    Wrapper to make Keras models compatible with the sklearn-style predict interface.
    """

    def __init__(self, model_path=None, model=None):
        """
        Initialize with either a path to a saved model or a model object.

        Args:
            model_path: Path to .keras or .h5 model file
            model: Pre-loaded Keras model object
        """
        self.logger = logging.getLogger("KerasModelWrapper")

        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required but not installed")

        if model_path:
            # Save the model path for pickling
            self._model_path = model_path

            # Check if this is a TCN model and prepare custom objects
            model_name_lower = Path(model_path).stem.lower()
            custom_objects = {}

            if "tcn" in model_name_lower:
                if TCN_AVAILABLE:
                    custom_objects = {"TCN": TCN}
                    self.logger.info(f"Loading TCN model with custom layers")
                else:
                    self.logger.warning(
                        "TCN library not installed but loading TCN model. Install with: pip install keras-tcn"
                    )

            # Load model with custom objects if needed
            try:
                if custom_objects:
                    self.model = keras.models.load_model(
                        model_path, custom_objects=custom_objects
                    )
                else:
                    self.model = keras.models.load_model(model_path)
                self.logger.info(f"Loaded Keras model from {model_path}")
            except Exception as e:
                # Try loading with compile=False as fallback
                self.logger.warning(
                    f"Failed to load model normally, trying with compile=False: {e}"
                )
                if custom_objects:
                    self.model = keras.models.load_model(
                        model_path, custom_objects=custom_objects, compile=False
                    )
                else:
                    self.model = keras.models.load_model(model_path, compile=False)
                self.logger.info(
                    f"Loaded Keras model from {model_path} (without compilation)"
                )
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")

        # Store model metadata
        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape

    def predict(self, X):
        """
        Predict method compatible with sklearn interface.

        FIXED: Returns RAW predictions (continuous values), NOT binary signals.
        This matches the behavior of other model types (GBM, Linear Regression, VAR).
        Signal conversion is handled by strategy_executor._convert_to_binary_signal().

        Args:
            X: Input features (numpy array or pandas DataFrame)

        Returns:
            Array of RAW predictions (continuous values like 0.142100)
        """
        # Ensure X is numpy array
        if hasattr(X, "values"):  # pandas DataFrame
            X = X.values

        # Ensure correct shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get raw predictions from Keras model
        predictions = self.model.predict(X, verbose=0)

        # Return RAW predictions (do NOT convert to signals)
        # Let strategy_executor handle signal conversion for consistency
        return predictions.flatten()

    def predict_signals(self, X):
        """
        Alternative method to get binary signals directly.

        Most users should use predict() and let strategy_executor handle conversion.
        This method is provided for backwards compatibility or special use cases.

        Returns:
            Array of binary trading signals (-1, 0, 1)
        """
        predictions = self.predict(X)
        return self._convert_to_signals(predictions.reshape(-1, 1))

    def _convert_to_signals(self, predictions):
        """
        Convert model predictions to trading signals.

        Override this method for different signal conversion logic.
        """
        # FIX: Use zero threshold to match backtesting
        # In backtesting, the signal is: 1 if prediction > 0 else -1
        # These thresholds were causing the discrepancy
        threshold = 0.0  # Match backtesting threshold

        # Log raw predictions BEFORE conversion for transparency
        self.logger.info(
            f"[Keras Wrapper] Raw model output (BEFORE conversion): {predictions.flatten()}"
        )

        signals = np.zeros(len(predictions))

        # Handle both binary classification and regression outputs
        if predictions.shape[-1] == 1:  # Single output (regression or binary)
            predictions_flat = predictions.flatten()
            # Simple binary conversion matching backtesting
            signals[predictions_flat > threshold] = 1  # Buy
            signals[predictions_flat <= threshold] = -1  # Sell

            # Log conversion details
            for i, (pred, sig) in enumerate(zip(predictions_flat, signals)):
                self.logger.debug(
                    f"[Keras Wrapper] Prediction[{i}]: {pred:.8f} → Signal: {int(sig)}"
                )
        else:  # Multi-class output
            # Assume 3 classes: [sell_prob, hold_prob, buy_prob]
            predicted_classes = np.argmax(predictions, axis=1)
            signals[predicted_classes == 0] = -1  # Sell
            signals[predicted_classes == 1] = 0  # Hold
            signals[predicted_classes == 2] = 1  # Buy

            self.logger.info(
                f"[Keras Wrapper] Multi-class probabilities: {predictions[0]}"
            )
            self.logger.info(
                f"[Keras Wrapper] Predicted class: {predicted_classes[0]} → Signal: {int(signals[0])}"
            )

        return signals.astype(int)

    def save_as_pkl(self, output_path):
        """
        Save the wrapper (with model) as a pickle file for compatibility.
        """
        # Save model path instead of the actual model to avoid serialization issues
        self._model_path = getattr(self, "_model_path", None)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"Saved Keras wrapper as pickle to {output_path}")

    def __getstate__(self):
        """Custom pickle method to avoid serializing the Keras model directly."""
        state = self.__dict__.copy()
        # Save model path and remove the actual model
        if hasattr(self, "model"):
            # Save the model temporarily if we have a path
            if hasattr(self, "_model_path"):
                state["_model_path"] = self._model_path
            # Remove the model from pickle
            del state["model"]
        return state

    def __setstate__(self, state):
        """Custom unpickle method to reload the Keras model."""
        self.__dict__.update(state)
        # Reload the model if we have a path
        if hasattr(self, "_model_path") and self._model_path:
            # Import TCN if needed
            model_name_lower = Path(self._model_path).stem.lower()
            custom_objects = {}

            if "tcn" in model_name_lower:
                try:
                    from tcn import TCN

                    custom_objects = {"TCN": TCN}
                except ImportError as e:
                    import logging

                    logging.error(
                        f"TCN library not installed! Install with: pip install keras-tcn"
                    )
                    logging.error(f"Import error: {e}")
                    # Still try to set empty custom_objects to trigger better error
                    custom_objects = {}

            # Load the model
            try:
                if custom_objects:
                    self.model = keras.models.load_model(
                        self._model_path, custom_objects=custom_objects
                    )
                else:
                    self.model = keras.models.load_model(self._model_path)
            except Exception as e:
                # Check if it's a TCN-specific error
                if "TCN" in str(e) and not custom_objects:
                    import logging

                    logging.error("Cannot load TCN model without keras-tcn library!")
                    logging.error("Run on VPS: pip install keras-tcn")
                    raise ImportError(
                        "keras-tcn library required but not installed. Install with: pip install keras-tcn"
                    )
                # Try with compile=False as fallback
                if custom_objects:
                    self.model = keras.models.load_model(
                        self._model_path, custom_objects=custom_objects, compile=False
                    )
                else:
                    self.model = keras.models.load_model(
                        self._model_path, compile=False
                    )


class LSTMTradingModel(KerasModelWrapper):
    """
    Specialized wrapper for LSTM/TCN/RNN sequence models with proper sequence handling.
    """

    def __init__(self, model_path=None, model=None, sequence_length=60):
        # Override parent __init__ to handle TCN models
        self.logger = logging.getLogger("LSTMTradingModel")
        self._config_sequence_length = (
            sequence_length  # Store config value for reference
        )

        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required but not installed")

        if model_path:
            # Save the model path for pickling
            self._model_path = model_path

            # Check if this is a TCN model and prepare custom objects
            model_name_lower = Path(model_path).stem.lower()
            custom_objects = {}

            if "tcn" in model_name_lower:
                if TCN_AVAILABLE:
                    custom_objects = {"TCN": TCN}
                    self.logger.info(f"Loading TCN model with custom layers")
                else:
                    self.logger.warning(
                        "TCN library not installed but loading TCN model. Install with: pip install keras-tcn"
                    )

            # Load model with custom objects if needed
            try:
                if custom_objects:
                    self.model = keras.models.load_model(
                        model_path, custom_objects=custom_objects
                    )
                else:
                    self.model = keras.models.load_model(model_path)
                self.logger.info(f"Loaded sequence model from {model_path}")
            except Exception as e:
                # Try loading with compile=False as fallback
                self.logger.warning(
                    f"Failed to load model normally, trying with compile=False: {e}"
                )
                if custom_objects:
                    self.model = keras.models.load_model(
                        model_path, custom_objects=custom_objects, compile=False
                    )
                else:
                    self.model = keras.models.load_model(model_path, compile=False)
                self.logger.info(
                    f"Loaded sequence model from {model_path} (without compilation)"
                )
        elif model:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided")

        # Store model metadata
        self.input_shape = self.model.input_shape
        self.output_shape = self.model.output_shape

        # AUTO-DETECT sequence_length from model's input shape
        # Input shape is typically (None, timesteps, features) for LSTM/TCN
        if len(self.input_shape) == 3 and self.input_shape[1] is not None:
            detected_length = self.input_shape[1]
            if detected_length != sequence_length:
                self.logger.warning(
                    f"Config sequence_length={sequence_length} does not match model's expected "
                    f"input shape {self.input_shape}. Auto-correcting to {detected_length}."
                )
            self.sequence_length = detected_length
            self.logger.info(
                f"Sequence length auto-detected from model: {self.sequence_length}"
            )
        else:
            # Fallback to config value if shape detection fails
            self.sequence_length = sequence_length
            self.logger.info(f"Using config sequence_length: {self.sequence_length}")

    def predict(self, X):
        """
        LSTM/TCN-specific prediction handling.
        """
        # Ensure X is numpy array
        if hasattr(X, "values"):
            X = X.values

        # Get expected input shape from model
        expected_shape = self.model.input_shape

        # Handle different input shape scenarios
        if len(expected_shape) == 3:  # (batch, timesteps, features)
            expected_timesteps = expected_shape[1]
            expected_features = expected_shape[2]

            # Reshape input data appropriately
            if len(X.shape) == 1:
                # Single row of features
                X = X.reshape(1, -1)

            if len(X.shape) == 2:
                # Check if we have the right shape already
                if (
                    X.shape[0] == 1
                    and X.shape[1] == expected_timesteps * expected_features
                ):
                    # Reshape flat input to (1, timesteps, features)
                    X = X.reshape(1, expected_timesteps, expected_features)
                elif X.shape[1] == expected_timesteps:
                    # Already has timesteps, just add batch dimension
                    X = X.reshape(1, X.shape[0], X.shape[1])
                else:
                    # Use the last expected_timesteps of data
                    if X.shape[0] >= expected_timesteps:
                        # For TCN with (5, 1) shape - take last 5 values
                        if expected_features == 1:
                            # Take last column or last expected_timesteps rows
                            if X.shape[1] == 1:
                                X = X[-expected_timesteps:].reshape(
                                    1, expected_timesteps, 1
                                )
                            else:
                                # Use first column of last expected_timesteps rows
                                X = X[-expected_timesteps:, 0].reshape(
                                    1, expected_timesteps, 1
                                )
                        else:
                            # Standard LSTM case
                            X = X[-expected_timesteps:].reshape(
                                1, expected_timesteps, X.shape[1]
                            )
                    else:
                        # Pad with zeros if not enough data
                        if expected_features == 1:
                            # For TCN case
                            padded = np.zeros((expected_timesteps, 1))
                            if X.shape[1] == 1:
                                padded[-X.shape[0] :] = X
                            else:
                                padded[-X.shape[0] :, 0] = X[:, 0]
                            X = padded.reshape(1, expected_timesteps, 1)
                        else:
                            padded = np.zeros((expected_timesteps, X.shape[1]))
                            padded[-X.shape[0] :] = X
                            X = padded.reshape(1, expected_timesteps, X.shape[1])

        # Get predictions
        predictions = self.model.predict(X, verbose=0)

        # Return RAW predictions (continuous values), NOT binary signals.
        # This matches the parent KerasModelWrapper.predict() design:
        # signal conversion is handled by strategy_executor._convert_to_binary_signal().
        return predictions.flatten()


def convert_keras_to_pkl(keras_model_path, output_pkl_path, model_type="standard"):
    """
    Utility function to convert a Keras model to pickle format.

    Args:
        keras_model_path: Path to .keras or .h5 file
        output_pkl_path: Where to save the pickle file
        model_type: 'standard' or 'lstm'

    Example:
        convert_keras_to_pkl('models/my_lstm.keras',
                           'strategy_models/spy_lstm_model.pkl',
                           model_type='lstm')
    """
    if model_type == "lstm":
        wrapper = LSTMTradingModel(model_path=keras_model_path)
    else:
        wrapper = KerasModelWrapper(model_path=keras_model_path)

    wrapper.save_as_pkl(output_pkl_path)
    print(f"Successfully converted {keras_model_path} to {output_pkl_path}")
    return wrapper


# Example usage
if __name__ == "__main__":
    # Example 1: Convert existing Keras model
    """
    convert_keras_to_pkl(
        'path/to/your/model.keras',
        'strategy_models/spy_keras_model.pkl'
    )
    """

    # Example 2: Create and save a simple Keras model for trading
    if KERAS_AVAILABLE:
        # Create a simple model
        model = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(5,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(
                    3, activation="softmax"
                ),  # 3 classes: sell, hold, buy
            ]
        )

        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Save as .keras
        model.save("example_model.keras")

        # Convert to pkl
        wrapper = KerasModelWrapper(model_path="example_model.keras")
        wrapper.save_as_pkl("example_model.pkl")

        print("Example model created and converted successfully!")
