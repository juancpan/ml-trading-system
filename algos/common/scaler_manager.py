"""
Scaler Manager for persisting StandardScaler objects from model training.
This ensures live trading uses the exact same scaling as backtesting.
Also saves feature metadata alongside scalers for deployment validation.
"""

import json
import pickle
import os
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


class ScalerManager:
    """Manages saving and loading of StandardScaler objects for model deployment."""

    @staticmethod
    def save_scaler(
        scaler,
        symbol,
        model_type,
        save_dir="../execution/strategy_models",
        feature_names: Optional[list[str]] = None,
    ):
        """
        Save a fitted StandardScaler for later use in live trading.
        Optionally saves feature metadata JSON alongside the scaler.

        Args:
            scaler: Fitted StandardScaler object
            symbol: Trading symbol (e.g., 'NVDA', 'AVGO')
            model_type: Type of model (e.g., 'lstm', 'li_reg', 'arima')
            save_dir: Directory to save the scaler
            feature_names: Ordered list of feature column names (for metadata)
        """
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create filename for scaler
        scaler_filename = f"{model_type}_scaler_{symbol}.pkl"
        scaler_path = save_path / scaler_filename

        # Save the scaler (protocol=4 for cross-version compatibility)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f, protocol=4)

        print(f"Saved scaler for {symbol} ({model_type}) to {scaler_path}")

        # Save feature metadata alongside scaler
        n_features = getattr(scaler, "n_features_in_", None)
        meta = {
            "n_features": n_features,
            "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
            "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "var": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
            "n_samples_seen": int(scaler.n_samples_seen_)
            if hasattr(scaler, "n_samples_seen_")
            else None,
        }

        # Include feature names if provided
        if feature_names is not None:
            meta["feature_names"] = feature_names
            meta["n_features"] = len(feature_names)

        meta_path = scaler_path.with_suffix(".json")
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved scaler metadata to {meta_path}")
        except Exception as e:
            print(f"Warning: Could not save scaler metadata: {e}")

        return scaler_path

    @staticmethod
    def load_scaler(symbol, model_type, load_dir="strategy_models"):
        """
        Load a saved StandardScaler for use in live trading.

        Args:
            symbol: Trading symbol (e.g., 'NVDA', 'AVGO')
            model_type: Type of model (e.g., 'lstm', 'li_reg', 'arima')
            load_dir: Directory to load the scaler from

        Returns:
            StandardScaler object or None if not found
        """
        # Try multiple possible paths
        scaler_paths = [
            Path(load_dir) / f"{model_type}_scaler_{symbol}.pkl",
            Path(load_dir) / f"scaler_{symbol}.pkl",
            Path(load_dir) / f"{symbol}_scaler.pkl",
        ]

        for scaler_path in scaler_paths:
            if scaler_path.exists():
                try:
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                        print(f"Loaded scaler for {symbol} from {scaler_path}")
                        return scaler
                except Exception as e:
                    print(f"Error loading scaler from {scaler_path}: {e}")

        print(f"No scaler found for {symbol} ({model_type})")
        return None

    @staticmethod
    def create_identity_scaler():
        """
        Create a dummy scaler that doesn't transform data (identity transformation).
        Useful as fallback when no scaler is available.

        Returns:
            StandardScaler object that performs identity transformation
        """
        scaler = StandardScaler()
        # Set mean to 0 and scale to 1 for identity transformation
        scaler.mean_ = np.array([0.0])
        scaler.scale_ = np.array([1.0])
        scaler.var_ = np.array([1.0])
        scaler.n_features_in_ = 1
        scaler.n_samples_seen_ = 1
        return scaler


def integrate_scaler_saving(model_func_name):
    """
    Decorator to automatically save scalers when training models.

    Usage:
        @integrate_scaler_saving('lstm')
        def run_lstm_strategy(data, ...):
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Run the original function
            result = func(*args, **kwargs)

            # Extract symbol and scaler if available
            # This assumes the function returns (test_df, model_object)
            # and that scaler is accessible somehow
            # You'll need to modify model training functions to return scaler too

            return result

        return wrapper

    return decorator


# Helper function to update existing model training scripts
def update_model_with_scaler_saving(train_function, symbol, model_type, scaler):
    """
    Helper to save scaler after model training.
    Call this in your model training scripts.

    Example:
        # In lstm_model.py, after fitting scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[cols])

        # Save the scaler
        from algos.common.scaler_manager import ScalerManager
        ScalerManager.save_scaler(scaler, symbol, 'lstm')
    """
    ScalerManager.save_scaler(scaler, symbol, model_type)
