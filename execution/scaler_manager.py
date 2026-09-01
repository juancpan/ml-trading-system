#!/usr/bin/env python3
"""
Scaler management system for consistent feature preprocessing between backtesting and live trading.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from typing import Optional, Dict, Any, Union


class ScalerManager:
    """
    Manages scalers for ML models, ensuring consistent preprocessing
    between training and inference.
    """

    def __init__(self, base_path: str = "strategy_models"):
        """
        Initialize the ScalerManager.

        Args:
            base_path: Directory where scalers and metadata are stored
        """
        self.base_path = Path(base_path)
        self.scalers_dir = self.base_path / "scalers"
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ScalerManager")

    def save_scaler(
        self,
        scaler: StandardScaler,
        symbol: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a fitted scaler with metadata.

        Args:
            scaler: Fitted sklearn scaler
            symbol: Trading symbol (e.g., 'NVDA')
            model_type: Model type (e.g., 'lstm', 'li_reg')
            metadata: Additional metadata to save

        Returns:
            Path to saved scaler file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_filename = f"{model_type}_scaler_{symbol}_{timestamp}.pkl"
        scaler_path = self.scalers_dir / scaler_filename

        # Also save as latest (for easy access)
        latest_path = self.scalers_dir / f"{model_type}_scaler_{symbol}_latest.pkl"

        # Prepare metadata
        meta = {
            "symbol": symbol,
            "model_type": model_type,
            "timestamp": timestamp,
            "feature_names": metadata.get("feature_names", []) if metadata else [],
            "n_features": scaler.n_features_in_,
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "training_date_range": metadata.get("training_date_range")
            if metadata
            else None,
            "additional_info": metadata if metadata else {},
        }

        # Save scaler
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(latest_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save metadata
        meta_path = scaler_path.with_suffix(".json")
        latest_meta_path = latest_path.with_suffix(".json")

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        with open(latest_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.logger.info(f"Saved scaler for {symbol} ({model_type}) to {scaler_path}")
        self.logger.info(f"Also saved as latest: {latest_path}")

        return scaler_path

    def load_scaler(
        self, symbol: str, model_type: str, use_latest: bool = True
    ) -> Optional[StandardScaler]:
        """
        Load a saved scaler.

        Args:
            symbol: Trading symbol
            model_type: Model type
            use_latest: Whether to use the latest version

        Returns:
            Loaded scaler or None if not found
        """
        if use_latest:
            scaler_path = self.scalers_dir / f"{model_type}_scaler_{symbol}_latest.pkl"
        else:
            # Find most recent timestamped version
            pattern = f"{model_type}_scaler_{symbol}_*.pkl"
            files = list(self.scalers_dir.glob(pattern))
            files = [f for f in files if "latest" not in f.name]
            if not files:
                self.logger.warning(f"No scaler found for {symbol} ({model_type})")
                return None
            scaler_path = max(files, key=lambda f: f.stat().st_mtime)

        if not scaler_path.exists():
            # Try alternative paths
            alt_paths = [
                self.base_path / f"{model_type}_scaler_{symbol}.pkl",
                self.base_path / f"lstm_scaler_{symbol}.pkl",  # Legacy path
                self.base_path / f"scaler_{symbol}.pkl",
            ]

            for alt_path in alt_paths:
                if alt_path.exists():
                    scaler_path = alt_path
                    break
            else:
                self.logger.warning(f"No scaler found for {symbol} in any location")
                return None

        try:
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                for w in caught:
                    self.logger.warning(
                        f"Scaler load warning for {symbol}: {w.message}"
                    )
            self.logger.info(f"Loaded scaler for {symbol} from {scaler_path}")

            # Load and log metadata if available
            meta_path = scaler_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                self.logger.debug(
                    f"Scaler metadata: {metadata.get('timestamp', 'unknown')}, "
                    f"features: {metadata.get('n_features', 'unknown')}"
                )

            return scaler

        except Exception as e:
            self.logger.error(f"Error loading scaler from {scaler_path}: {e}")
            return None

    def create_and_save_scaler_from_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        symbol: str,
        model_type: str,
        feature_cols: Optional[list] = None,
        train_split: float = 0.5,
    ) -> StandardScaler:
        """
        Create, fit, and save a scaler from data.

        Args:
            data: Training data (DataFrame or array)
            symbol: Trading symbol
            model_type: Model type
            feature_cols: Column names for features
            train_split: Proportion of data to use for fitting

        Returns:
            Fitted scaler
        """
        if isinstance(data, pd.DataFrame):
            if feature_cols:
                train_data = data[feature_cols]
            else:
                # Assume all numeric columns except 'direction', 'returns', etc.
                exclude_cols = [
                    "direction",
                    "returns",
                    "log_returns",
                    "Date",
                    "price",
                    "close",
                ]
                feature_cols = [col for col in data.columns if col not in exclude_cols]
                train_data = data[feature_cols]
        else:
            train_data = data

        # Use only training portion for fitting
        split_idx = int(len(train_data) * train_split)
        fit_data = (
            train_data.iloc[:split_idx]
            if isinstance(train_data, pd.DataFrame)
            else train_data[:split_idx]
        )

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(fit_data)

        # Prepare metadata
        metadata = {
            "feature_names": feature_cols if feature_cols else [],
            "train_samples": split_idx,
            "total_samples": len(train_data),
            "train_split": train_split,
            "data_shape": train_data.shape,
        }

        # Save scaler
        self.save_scaler(scaler, symbol, model_type, metadata)

        return scaler

    def validate_scaler(self, scaler: StandardScaler, expected_features: int) -> bool:
        """
        Validate that a scaler matches expected configuration.

        Args:
            scaler: Scaler to validate
            expected_features: Expected number of features

        Returns:
            True if valid, False otherwise
        """
        if scaler.n_features_in_ != expected_features:
            self.logger.warning(
                f"Scaler feature mismatch: expected {expected_features}, "
                f"got {scaler.n_features_in_}"
            )
            return False
        return True

    def copy_scaler_to_strategy_models(self, symbol: str, model_type: str) -> bool:
        """
        Copy the latest scaler to the main strategy_models directory
        for backward compatibility.

        Args:
            symbol: Trading symbol
            model_type: Model type

        Returns:
            True if successful
        """
        source_path = self.scalers_dir / f"{model_type}_scaler_{symbol}_latest.pkl"

        if not source_path.exists():
            self.logger.error(f"Source scaler not found: {source_path}")
            return False

        # Copy to multiple legacy locations for compatibility
        dest_paths = [
            self.base_path / f"{model_type}_scaler_{symbol}.pkl",
            self.base_path / f"lstm_scaler_{symbol}.pkl"
            if model_type == "lstm"
            else None,
        ]

        for dest_path in dest_paths:
            if dest_path:
                try:
                    import shutil

                    shutil.copy2(source_path, dest_path)
                    self.logger.info(f"Copied scaler to {dest_path}")
                except Exception as e:
                    self.logger.error(f"Failed to copy scaler to {dest_path}: {e}")
                    return False

        return True


def create_scalers_for_deployed_models():
    """
    Utility function to create scalers for currently deployed models.
    """
    import yfinance as yf
    from config import ASSET_SPECIFIC_CONFIGS

    logger = logging.getLogger("ScalerCreation")
    manager = ScalerManager()

    for symbol, config in ASSET_SPECIFIC_CONFIGS.items():
        model_type = config.get("model_type", "unknown")
        logger.info(f"Creating scaler for {symbol} ({model_type} model)")

        # Fetch historical data
        end_date = datetime.now().date()
        start_date = datetime(2020, 9, 1).date()  # Match training period

        df = yf.download(
            symbol, start=start_date, end=end_date, auto_adjust=False, progress=False
        )

        if df.empty:
            logger.error(f"No data available for {symbol}")
            continue

        # Calculate returns and features
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Create lagged features
        lags = config.get("lags", 5)
        feature_cols = []
        for lag in range(1, lags + 1):
            col = f"returns_lag_{lag}"
            df[col] = df["returns"].shift(lag)
            feature_cols.append(col)

        df.dropna(inplace=True)

        # Create and save scaler
        scaler = manager.create_and_save_scaler_from_data(
            data=df,
            symbol=symbol,
            model_type=model_type,
            feature_cols=feature_cols,
            train_split=0.5,  # Match backtesting split
        )

        # Copy to legacy locations
        manager.copy_scaler_to_strategy_models(symbol, model_type)

        logger.info(f"Successfully created and saved scaler for {symbol}")

    logger.info("Scaler creation complete")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create scalers for deployed models
    create_scalers_for_deployed_models()
