"""
Universal scaler integration for all backtesting models.
This module patches model functions to automatically save scalers.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import inspect
import functools
from typing import Any, Tuple, Optional


class ScalerAutoSaver:
    """Automatically saves scalers when models are trained."""

    def __init__(self):
        self.project_root = self._find_project_root()
        self.scalers_dir = self.project_root / "algos" / "scalers"
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        self.current_scaler = None
        self.current_config = {}

    def _find_project_root(self) -> Path:
        """Find the project root project root."""
        current = Path.cwd()
        for _ in range(5):
            if (current / "algos").is_dir():
                return current
            if (current / "algos").exists() and (current / "execution").exists():
                return current
            current = current.parent
        return Path.cwd()

    def capture_scaler(self, scaler: StandardScaler, config: dict) -> StandardScaler:
        """Capture a scaler for later saving."""
        self.current_scaler = scaler
        self.current_config = config
        return scaler

    def save_captured_scaler(
        self, model_name: str, ticker: str, timestamp: str
    ) -> Optional[Path]:
        """Save the captured scaler with proper naming."""
        if self.current_scaler is None:
            return None

        # Create filename following the model naming convention
        scaler_filename = f"{model_name}_scaler_{ticker}_{timestamp}.pkl"
        scaler_path = self.scalers_dir / scaler_filename

        # Save the scaler (protocol=4 for cross-version compatibility)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.current_scaler, f, protocol=4)

        # Also save as 'latest' for easy access
        latest_path = self.scalers_dir / f"{model_name}_scaler_{ticker}_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(self.current_scaler, f, protocol=4)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "ticker": ticker,
            "timestamp": timestamp,
            "saved_at": datetime.now().isoformat(),
            "n_features": self.current_scaler.n_features_in_,
            "n_samples": int(self.current_scaler.n_samples_seen_),
            "feature_means": self.current_scaler.mean_.tolist(),
            "feature_scales": self.current_scaler.scale_.tolist(),
            "feature_vars": self.current_scaler.var_.tolist(),
            **self.current_config,
        }

        meta_path = scaler_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save metadata for latest
        latest_meta_path = latest_path.with_suffix(".json")
        with open(latest_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Scaler saved to: {scaler_path}")
        print(f"  Also saved as: {latest_path}")

        # Reset for next model
        self.current_scaler = None
        self.current_config = {}

        return scaler_path


# Global instance
scaler_saver = ScalerAutoSaver()


def patch_standardscaler():
    """Monkey-patch StandardScaler to capture instances automatically."""
    original_fit = StandardScaler.fit
    original_fit_transform = StandardScaler.fit_transform

    @functools.wraps(original_fit)
    def patched_fit(self, X, y=None, sample_weight=None):
        result = original_fit(self, X, y, sample_weight)
        # Capture this scaler
        frame = inspect.currentframe()
        if frame and frame.f_back:
            local_vars = frame.f_back.f_locals
            config = {
                "function": frame.f_back.f_code.co_name,
                "module": frame.f_back.f_globals.get("__name__", "unknown"),
                "data_shape": X.shape if hasattr(X, "shape") else len(X),
            }
            scaler_saver.capture_scaler(self, config)
        return result

    @functools.wraps(original_fit_transform)
    def patched_fit_transform(self, X, y=None, **fit_params):
        result = original_fit_transform(self, X, y, **fit_params)
        # Capture this scaler
        frame = inspect.currentframe()
        if frame and frame.f_back:
            local_vars = frame.f_back.f_locals
            config = {
                "function": frame.f_back.f_code.co_name,
                "module": frame.f_back.f_globals.get("__name__", "unknown"),
                "data_shape": X.shape if hasattr(X, "shape") else len(X),
            }
            scaler_saver.capture_scaler(self, config)
        return result

    StandardScaler.fit = patched_fit
    StandardScaler.fit_transform = patched_fit_transform


def wrap_model_function(original_func, model_name: str):
    """
    Wrap a model function to automatically save its scaler.

    Args:
        original_func: The original model training function
        model_name: Name of the model for scaler naming
    """

    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs):
        # Reset scaler saver
        scaler_saver.current_scaler = None
        scaler_saver.current_config = {}

        # Run the original function
        result = original_func(*args, **kwargs)

        # The scaler should have been captured if StandardScaler was used
        # We don't save here as the main runner will handle it

        return result

    return wrapped_func


def integrate_scaler_saving(save_model_func):
    """
    Decorator for the save_model function to also save scalers.

    Usage:
        @integrate_scaler_saving
        def save_model(model_obj, model_name, ticker, ...):
            ...
    """

    @functools.wraps(save_model_func)
    def wrapped_save(*args, **kwargs):
        # Extract parameters
        if len(args) >= 3:
            model_name = (
                args[1] if len(args) > 1 else kwargs.get("model_name", "unknown")
            )
            ticker = args[2] if len(args) > 2 else kwargs.get("ticker", "unknown")
            timestamp = (
                args[7]
                if len(args) > 7
                else kwargs.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            )
        else:
            model_name = kwargs.get("model_name", "unknown")
            ticker = kwargs.get("ticker", "unknown")
            timestamp = kwargs.get(
                "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        # Save the model first
        result = save_model_func(*args, **kwargs)

        # Now save the scaler if one was captured
        if scaler_saver.current_scaler is not None:
            scaler_saver.save_captured_scaler(model_name, ticker, timestamp)

        return result

    return wrapped_save


# Auto-patch on import
patch_standardscaler()


def get_model_scaler_path(
    model_name: str, ticker: str, use_latest: bool = True
) -> Optional[Path]:
    """
    Get the path to a saved scaler for a model.

    Args:
        model_name: Model type (lstm, svm, etc.)
        ticker: Trading symbol
        use_latest: Whether to get the latest version

    Returns:
        Path to scaler file or None if not found
    """
    scalers_dir = ScalerAutoSaver().scalers_dir

    if use_latest:
        scaler_path = scalers_dir / f"{model_name}_scaler_{ticker}_latest.pkl"
    else:
        # Find most recent timestamped version
        pattern = f"{model_name}_scaler_{ticker}_*.pkl"
        files = list(scalers_dir.glob(pattern))
        files = [f for f in files if "latest" not in f.name]
        if not files:
            return None
        scaler_path = max(files, key=lambda f: f.stat().st_mtime)

    return scaler_path if scaler_path.exists() else None


def load_model_scaler(
    model_name: str, ticker: str, use_latest: bool = True
) -> Optional[StandardScaler]:
    """
    Load a saved scaler for a model.

    Args:
        model_name: Model type
        ticker: Trading symbol
        use_latest: Whether to load the latest version

    Returns:
        Loaded StandardScaler or None
    """
    scaler_path = get_model_scaler_path(model_name, ticker, use_latest)

    if scaler_path is None:
        return None

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return None
