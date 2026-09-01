"""
Configuration loader that ensures consistency between backtesting and live trading.
Automatically loads model configs, scalers, seeds, and other settings.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """
    Loads and manages all configurations needed for consistent live trading.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the config loader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.strategy_models_dir = Path("strategy_models")
        self.configs = {}
        self.scalers = {}
        self.seeds = {}
        self.model_metadata = {}

    def load_all_configs(self, symbols: list) -> Dict[str, Any]:
        """
        Load all configurations for the specified symbols.
        SKIPS loading model configs for buy-and-hold strategy tickers.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary with all loaded configurations
        """
        self.logger.info("Loading comprehensive configurations for live trading")

        import config as local_config

        for symbol in symbols:
            self.logger.info(f"Loading configs for {symbol}")

            # Check strategy type - skip model config loading for buy-and-hold
            asset_config = local_config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
            strategy_type = asset_config.get(
                "strategy_type", local_config.DEFAULT_STRATEGY_TYPE
            )

            if strategy_type == "buy_and_hold":
                self.logger.info(
                    f"{symbol}: Buy-and-hold strategy - skipping model/scaler/seed loading"
                )
                # Still load deployment manifest
                manifest = self.load_deployment_manifest()
                if manifest:
                    self.configs["deployment_manifest"] = manifest
                continue

            # For ml_signal strategy, load all configs
            # Load scaler
            scaler = self.load_scaler(symbol)
            if scaler:
                self.scalers[symbol] = scaler

            # Load seed info
            seed_info = self.load_seed_info(symbol)
            if seed_info:
                self.seeds[symbol] = seed_info

            # Load model metadata
            metadata = self.load_model_metadata(symbol)
            if metadata:
                self.model_metadata[symbol] = metadata

            # Load deployment manifest
            manifest = self.load_deployment_manifest()
            if manifest:
                self.configs["deployment_manifest"] = manifest

        # Compile all configs
        all_configs = {
            "scalers": self.scalers,
            "seeds": self.seeds,
            "model_metadata": self.model_metadata,
            "deployment_info": self.configs.get("deployment_manifest", {}),
        }

        self.logger.info(
            f"Loaded configs for {len(self.scalers)} scalers, "
            f"{len(self.seeds)} seeds, {len(self.model_metadata)} metadata"
        )

        return all_configs

    def load_scaler(self, symbol: str) -> Optional[Any]:
        """
        Load the StandardScaler for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            StandardScaler object or None
        """
        # Check if this is an ARIMA model - they don't use scalers
        import config as local_config

        model_config = local_config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = model_config.get("model_type", "")

        if model_type == "arima":
            self.logger.debug(f"Skipping scaler load for {symbol} (ARIMA model)")
            return None

        # Try multiple naming conventions
        scaler_paths = [
            self.strategy_models_dir / f"lstm_scaler_{symbol}.pkl",
            self.strategy_models_dir / f"svm_scaler_{symbol}.pkl",
            self.strategy_models_dir / f"li_reg_scaler_{symbol}.pkl",
            self.strategy_models_dir / f"scaler_{symbol}.pkl",
            self.strategy_models_dir / f"{symbol}_scaler.pkl",
        ]

        for scaler_path in scaler_paths:
            if scaler_path.exists():
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
                    self.logger.info(
                        f"Loaded scaler for {symbol} from {scaler_path.name}"
                    )

                    # Also load metadata if available
                    json_path = scaler_path.with_suffix(".json")
                    if json_path.exists():
                        with open(json_path, "r") as f:
                            metadata = json.load(f)
                        self.logger.debug(
                            f"Scaler metadata: {metadata.get('n_features')} features, "
                            f"{metadata.get('n_samples')} samples"
                        )

                    return scaler
                except Exception as e:
                    self.logger.error(f"Error loading scaler from {scaler_path}: {e}")

        self.logger.warning(f"No scaler found for {symbol} ({model_type} model)")
        return None

    def load_seed_info(self, symbol: str) -> Optional[Dict]:
        """
        Load seed information for reproducibility.

        Args:
            symbol: Trading symbol

        Returns:
            Seed info dictionary or None
        """
        seed_path = self.strategy_models_dir / f"seed_info_{symbol}.json"

        if seed_path.exists():
            try:
                with open(seed_path, "r") as f:
                    seed_info = json.load(f)
                self.logger.info(
                    f"Loaded seed info for {symbol}: seed={seed_info.get('seed')}"
                )
                return seed_info
            except Exception as e:
                self.logger.error(f"Error loading seed info: {e}")

        return None

    def load_model_metadata(self, symbol: str) -> Optional[Dict]:
        """
        Load model training metadata.

        Args:
            symbol: Trading symbol

        Returns:
            Model metadata dictionary or None
        """
        # Try to find any metadata file for this symbol
        metadata_files = list(self.strategy_models_dir.glob(f"*{symbol}*.json"))

        for metadata_file in metadata_files:
            if "seed" in metadata_file.name or "manifest" in metadata_file.name:
                continue  # Skip seed and manifest files

            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                self.logger.info(
                    f"Loaded model metadata for {symbol} from {metadata_file.name}"
                )
                return metadata
            except Exception as e:
                self.logger.error(f"Error loading metadata from {metadata_file}: {e}")

        return None

    def load_deployment_manifest(self) -> Optional[Dict]:
        """
        Load the deployment manifest.

        Returns:
            Deployment manifest dictionary or None
        """
        manifest_path = self.strategy_models_dir / "deployment_manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                self.logger.info(
                    f"Loaded deployment manifest from {manifest.get('deployed_at')}"
                )
                return manifest
            except Exception as e:
                self.logger.error(f"Error loading deployment manifest: {e}")

        return None

    def get_model_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive configuration for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Configuration dictionary
        """
        config = {
            "symbol": symbol,
            "scaler": self.scalers.get(symbol),
            "seed_info": self.seeds.get(symbol, {}),
            "model_metadata": self.model_metadata.get(symbol, {}),
            "has_scaler": symbol in self.scalers,
            "model_type": None,
            "n_features": None,
        }

        # Extract model type from seed info, asset config, or metadata
        if config["seed_info"]:
            config["model_type"] = config["seed_info"].get("model_name")
        if not config["model_type"]:
            # Fallback: get model_type from ASSET_SPECIFIC_CONFIGS (always available)
            try:
                from config import ASSET_SPECIFIC_CONFIGS

                asset_cfg = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
                config["model_type"] = asset_cfg.get("model_type")
            except ImportError:
                pass

        # Extract feature count from scaler
        if config["scaler"] and hasattr(config["scaler"], "n_features_in_"):
            config["n_features"] = config["scaler"].n_features_in_

        return config

    def validate_config(self, symbol: str) -> bool:
        """
        Validate that all necessary configs are present for a symbol.
        SKIPS validation for buy-and-hold strategy tickers (they don't need models/scalers).

        Args:
            symbol: Trading symbol

        Returns:
            True if valid, False otherwise
        """
        import config as local_config

        # Check if this is a buy-and-hold ticker
        asset_config = local_config.ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        strategy_type = asset_config.get(
            "strategy_type", local_config.DEFAULT_STRATEGY_TYPE
        )

        if strategy_type == "buy_and_hold":
            self.logger.info(
                f"Configuration valid for {symbol}: "
                f"buy-and-hold strategy (no model/scaler needed)"
            )
            return True

        # For ml_signal strategy, validate normally
        config = self.get_model_config(symbol)

        issues = []

        # Only require scaler for models that consume scaled features.
        # ARIMA and VAR forecast from internal state (see strategy_executor.py
        # line ~752: `if model_type in ["arima", "var"]`) and never use a
        # scaler, so flagging them "Missing scaler" was a false validation
        # failure (e.g. TELEKOM.BD, a working VAR model, was logged as
        # "Configuration validation failed" yet traded normally).
        model_type = config.get("model_type", "")
        if model_type not in ("arima", "var") and not config["has_scaler"]:
            issues.append("Missing scaler")

        if not config["seed_info"]:
            # Missing seed info is a warning, not a blocker, if model_type is known
            if config["model_type"]:
                self.logger.debug(
                    f"{symbol}: No seed_info file (seed_info_{symbol}.json), "
                    f"but model_type '{config['model_type']}' resolved from config"
                )
            else:
                issues.append("Missing seed info")

        if not config["model_type"]:
            issues.append("Unknown model type")

        if issues:
            self.logger.warning(
                f"Configuration issues for {symbol}: {', '.join(issues)}"
            )
            return False

        # Log appropriate message based on model type
        if model_type in ("arima", "var"):
            self.logger.info(
                f"Configuration valid for {symbol}: "
                f"model_type={config['model_type']} "
                f"({model_type.upper()} - forecasts from internal state, no scaler needed)"
            )
        else:
            self.logger.info(
                f"Configuration valid for {symbol}: "
                f"model_type={config['model_type']}, "
                f"n_features={config['n_features']}"
            )
        return True


# Singleton instance
_config_loader = None


def get_config_loader(logger=None) -> ConfigLoader:
    """
    Get the singleton ConfigLoader instance.

    Args:
        logger: Optional logger

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(logger)
    return _config_loader
