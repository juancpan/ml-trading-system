# strategy_executor.py

import pickle
import warnings
import numpy as np
from typing import Optional
from config import ASSET_SPECIFIC_CONFIGS, SYMBOLS
from utils import setup_logger
import os
from pathlib import Path
import sys

# Add project root to path and import seed manager
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from algos.common.seed_manager import set_seed, get_model_seed

# Import config loader for comprehensive configuration management
from config_loader import get_config_loader

# Revision Protocol — Phase 1.1 signal_history logging.
try:
    from signal_history import log_signal as _log_signal_history
except Exception:  # pragma: no cover - keep trading loop resilient
    _log_signal_history = None  # type: ignore

# Feature engine imports (optional — graceful fallback to legacy)
try:
    from algos.common.feature_engine import FeatureConfig
    from algos.common.external_data import ExternalDataCache

    _HAS_FEATURE_ENGINE = True
except ImportError:
    _HAS_FEATURE_ENGINE = False


class DummyAlgorithm:
    """
    Fallback algorithm when model files are missing.
    Supports all model types: regression, classification, ARIMA, VAR.
    """

    def predict(self, features):
        """Standard predict() for regression/classification models"""
        if isinstance(features, np.ndarray) and features.size > 0:
            return np.array([1 if features[0][0] > 0 else -1])
        return np.array([0])

    def forecast(self, steps=1):
        """ARIMA/VAR-style forecast() method - returns slightly bullish prediction"""
        # Return small positive value (bullish bias for safety)
        return np.array([0.001])  # Slightly positive → signal +1

    def get_forecast(self, steps=1):
        """Statsmodels ARIMA get_forecast() method"""

        class ForecastResult:
            predicted_mean = np.array([0.001])

        return ForecastResult()


class StrategyExecutor:
    """
    Loads trained ML algorithms (one per asset) and generates trading signals.
    Supports automatic conversion of Keras models to pickle format.
    """

    def __init__(self, data_manager, logger, lags=5):  # Lags default set to 5
        self.logger = logger
        self.data_manager = data_manager
        self.lags = lags  # Set lags BEFORE loading algorithms
        # Revision Protocol — set by main.py per loop iteration so
        # signal_history rows are tagged with the active region.
        self.current_region = "UNKNOWN"
        # When False, generate_signal() does NOT append to signal_history.
        # The shadow check (shadow_check.py) sets this False: it re-runs
        # generate_signal purely to COMPARE against the live recording, and
        # must not write back into the data it is auditing (doing so poisoned
        # signal_history with synthetic "SHADOW" rows and broke dedup).
        self.log_signals = True

        # Load comprehensive configurations
        self.config_loader = get_config_loader(logger)
        self.all_configs = self.config_loader.load_all_configs(SYMBOLS)
        self.logger.info(
            f"Loaded configurations: {len(self.all_configs['scalers'])} scalers, "
            f"{len(self.all_configs['seeds'])} seeds"
        )

        # Validate configurations
        for symbol in SYMBOLS:
            if not self.config_loader.validate_config(symbol):
                self.logger.warning(f"Configuration validation failed for {symbol}")

        # Load ARIMA settings for threshold-based signal generation
        self.arima_settings = self._load_arima_settings()

        self.algorithms = self._load_algorithms()

        # Initialize feature engine (optional — graceful fallback to legacy)
        self._feature_configs = {}  # symbol -> FeatureConfig
        self._external_data_cache = None
        self._init_feature_engine()

    def _init_feature_engine(self):
        """Initialize the feature engine and per-model configs."""
        if not _HAS_FEATURE_ENGINE:
            self.logger.info(
                "Feature engine not available, using legacy lagged features"
            )
            return

        try:
            # Find feature_config.yaml
            config_path = project_root / "feature_config.yaml"
            if not config_path.exists():
                self.logger.info(
                    "No feature_config.yaml found, using legacy lagged features"
                )
                return

            from config import DEFAULT_STRATEGY_TYPE

            # Build per-symbol FeatureConfig based on model type
            for symbol in SYMBOLS:
                asset_config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
                strategy_type = asset_config.get("strategy_type", DEFAULT_STRATEGY_TYPE)

                if strategy_type == "buy_and_hold":
                    continue

                model_type = asset_config.get("model_type", "standard")

                # Check if this symbol has deployed feature metadata
                meta = self.data_manager.load_feature_metadata(symbol)
                if meta and meta.get("n_features", 0) > 5:
                    # Model was trained with feature engineering — load matching config
                    fc = FeatureConfig(
                        config_path=str(config_path),
                        model_name=model_type,
                        ticker=symbol,
                    )
                    self._feature_configs[symbol] = fc
                    self.logger.info(
                        f"Feature engine enabled for {symbol} ({model_type}): "
                        f"config hash={fc.config_hash}"
                    )

            # Initialize external data cache if any symbol uses features
            if self._feature_configs:
                # Collect external configs from any feature config
                sample_config = next(iter(self._feature_configs.values()))
                ext_configs = sample_config.external_configs
                if ext_configs:
                    self._external_data_cache = ExternalDataCache(
                        external_configs=ext_configs,
                        lookback_days=300,
                        cache_ttl_hours=12,
                        logger=self.logger,
                    )
                    self.logger.info(
                        f"External data cache initialized for {len(ext_configs)} series"
                    )

        except Exception as e:
            self.logger.warning(
                f"Feature engine initialization failed: {e}. Using legacy features."
            )
            self._feature_configs = {}

    def refresh_external_data(self):
        """Refresh external data cache. Call before daily signal generation."""
        if self._external_data_cache is not None:
            try:
                self._external_data_cache.refresh()
            except Exception as e:
                self.logger.warning(f"Failed to refresh external data: {e}")

    def _load_arima_settings(self):
        """
        Load ARIMA model settings (signal_method, threshold, etc.) from JSON files.
        These settings must match those used in backtesting for consistent behavior.
        SKIPS loading for buy-and-hold strategy tickers.

        Returns:
            Dictionary mapping symbol -> settings dict
        """
        import json

        arima_settings = {}

        from config import DEFAULT_STRATEGY_TYPE

        for symbol in SYMBOLS:
            config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
            model_type = config.get("model_type", "")
            strategy_type = config.get("strategy_type", DEFAULT_STRATEGY_TYPE)

            # Skip buy-and-hold tickers (no ML model)
            if strategy_type == "buy_and_hold":
                continue

            if model_type == "arima":
                # Try multiple naming conventions for settings files
                settings_paths = [
                    Path("strategy_models")
                    / f"arima_settings_{symbol.replace('.', '_')}*.json",
                    Path("strategy_models") / f"arima_settings_{symbol}*.json",
                ]

                settings_file = None
                for pattern_path in settings_paths:
                    # Use glob to find matching files
                    pattern_str = str(pattern_path)
                    matching_files = list(
                        Path("strategy_models").glob(pattern_path.name)
                    )
                    if matching_files:
                        settings_file = matching_files[
                            0
                        ]  # Take the first (most recent)
                        break

                if settings_file and settings_file.exists():
                    try:
                        with open(settings_file, "r") as f:
                            settings = json.load(f)
                        arima_settings[symbol] = settings
                        self.logger.info(
                            f"Loaded ARIMA settings for {symbol}: "
                            f"signal_method={settings.get('signal_method')}, "
                            f"threshold={settings.get('threshold')}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error loading ARIMA settings for {symbol}: {e}"
                        )
                        # Set default settings to avoid crashes
                        arima_settings[symbol] = {
                            "signal_method": "simple",
                            "threshold": 0.0002,
                        }
                else:
                    self.logger.warning(
                        f"ARIMA settings file not found for {symbol}, using defaults"
                    )
                    arima_settings[symbol] = {
                        "signal_method": "simple",
                        "threshold": 0.0002,
                    }

        return arima_settings

    def _check_version_metadata(self, model_path: str):
        """Log a warning if the model was saved with different library versions."""
        import json as _json

        versions_path = Path(model_path).with_name(
            Path(model_path).stem + "_versions.json"
        )
        if not versions_path.exists():
            return  # No metadata (pre-fix model) — nothing to check

        try:
            with open(versions_path, "r") as f:
                saved_versions = _json.load(f)

            # Check sklearn specifically (most common source of breakage)
            saved_sklearn = saved_versions.get("sklearn")
            if saved_sklearn:
                try:
                    import sklearn

                    current_sklearn = sklearn.__version__
                    if saved_sklearn != current_sklearn:
                        self.logger.warning(
                            f"sklearn version mismatch for {Path(model_path).name}: "
                            f"saved={saved_sklearn}, current={current_sklearn}"
                        )
                except ImportError:
                    pass
        except Exception as e:
            self.logger.debug(f"Could not read version metadata: {e}")

    def _convert_keras_to_pkl(self, keras_path, symbol):
        """
        Automatically converts Keras models to pickle format.
        Caches the converted model for future use.
        """
        try:
            # Import the wrappers (will fail gracefully if TensorFlow not installed)
            from keras_model_wrapper import KerasModelWrapper, LSTMTradingModel
            from dqn_model_wrapper import DQNModelWrapper

            # Import TCN to ensure it's available for all Keras operations
            try:
                from tcn import TCN

                TCN_AVAILABLE = True
            except ImportError:
                TCN_AVAILABLE = False

            # Determine model type from filename
            model_name_lower = Path(keras_path).stem.lower()

            # Check model type
            is_dqn_model = "dqn" in model_name_lower
            sequence_models = ["lstm", "gru", "tcn", "rnn", "temporal"]
            is_sequence_model = any(seq in model_name_lower for seq in sequence_models)

            # Get optional model configuration from config
            model_config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
            model_type = model_config.get("model_type", "auto")
            sequence_length = model_config.get("sequence_length", 60)

            # Create cache directory
            cache_dir = Path("strategy_models/.cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Compute a hash of wrapper source files so cache invalidates when
            # predict() logic changes (not just when the .keras model changes).
            import hashlib

            wrapper_sources = [
                Path(__file__).parent / "keras_model_wrapper.py",
                Path(__file__).parent / "dqn_model_wrapper.py",
            ]
            wrapper_hash = hashlib.md5()
            for ws in wrapper_sources:
                if ws.exists():
                    wrapper_hash.update(ws.read_bytes())
            wrapper_hash_hex = wrapper_hash.hexdigest()[:12]

            # Generate cache filename (includes wrapper hash for invalidation)
            keras_file = Path(keras_path)
            cache_name = f"{symbol}_{keras_file.stem}_{wrapper_hash_hex}_converted.pkl"
            cache_path = cache_dir / cache_name

            # Check if already converted and cached
            if cache_path.exists():
                # Check if keras file is newer than cache
                if keras_file.stat().st_mtime <= cache_path.stat().st_mtime:
                    self.logger.info(
                        f"Using cached converted model for {symbol} from {cache_path}"
                    )

                    # Import TCN if this is a TCN model (needed for unpickling)
                    if "tcn" in keras_file.stem.lower():
                        try:
                            from tcn import TCN
                        except ImportError:
                            self.logger.warning(
                                "TCN library not available, may have issues loading TCN models"
                            )

                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
                else:
                    self.logger.info(f"Keras model updated, reconverting {symbol}")

            # Convert the model
            self.logger.info(f"Converting Keras model for {symbol}: {keras_path}")

            if model_type == "dqn" or (model_type == "auto" and is_dqn_model):
                # Handle DQN models
                lags = model_config.get("lags", self.lags)
                wrapper = DQNModelWrapper(model_path=keras_path, lags=lags)
                self.logger.info(f"Loaded as DQN model with {lags} lag features")
            elif model_type == "lstm" or (model_type == "auto" and is_sequence_model):
                wrapper = LSTMTradingModel(
                    model_path=keras_path, sequence_length=sequence_length
                )
                self.logger.info(
                    f"Loaded as sequence model with length {sequence_length}"
                )
            else:
                wrapper = KerasModelWrapper(model_path=keras_path)
                self.logger.info(f"Loaded as standard feedforward model")

            # Cache the converted model
            with open(cache_path, "wb") as f:
                pickle.dump(wrapper, f)
            self.logger.info(f"Cached converted model at {cache_path}")

            return wrapper

        except ImportError as e:
            self.logger.error(
                f"TensorFlow/Keras not installed. Cannot load Keras models. Install with: pip install tensorflow"
            )
            raise e
        except Exception as e:
            self.logger.error(
                f"Error converting Keras model for {symbol}: {e}", exc_info=True
            )
            raise e

    def _load_algorithms(self):
        """
        Loads the persisted algorithm objects for all symbols.
        Automatically handles both pickle and Keras model formats.
        SKIPS loading for buy-and-hold strategy tickers (they don't need models).
        """
        loaded_algos = {}

        from config import DEFAULT_STRATEGY_TYPE

        for symbol in SYMBOLS:
            if symbol not in ASSET_SPECIFIC_CONFIGS:
                self.logger.error(
                    f"No config found for symbol {symbol}. Skipping strategy loading."
                )
                continue

            # Check strategy type - skip model loading for buy-and-hold
            asset_config = ASSET_SPECIFIC_CONFIGS[symbol]
            strategy_type = asset_config.get("strategy_type", DEFAULT_STRATEGY_TYPE)

            if strategy_type == "buy_and_hold":
                self.logger.info(
                    f"Skipping model load for {symbol} (buy-and-hold strategy - no ML model needed)"
                )
                continue

            # ML signal strategy - load model
            if "strategy_model_path" not in asset_config:
                self.logger.error(
                    f"ml_signal strategy for {symbol} missing strategy_model_path. Skipping."
                )
                continue

            model_path = asset_config["strategy_model_path"]

            # Handle both running from project root and from execution directory
            if not os.path.exists(model_path):
                # Try from execution directory if not found
                alt_path = Path(__file__).parent / model_path
                if alt_path.exists():
                    model_path = str(alt_path)

            try:
                if not os.path.exists(model_path):
                    self.logger.warning(
                        f"Strategy model file not found for {symbol} at {model_path}. Creating dummy model."
                    )
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    dummy_algo = DummyAlgorithm()

                    # Save dummy as pickle regardless of requested format
                    dummy_path = (
                        model_path
                        if model_path.endswith(".pkl")
                        else model_path.replace(".keras", ".pkl").replace(".h5", ".pkl")
                    )
                    with open(dummy_path, "wb") as f:
                        pickle.dump(dummy_algo, f)
                    self.logger.info(
                        f"Dummy algorithm created for {symbol} at {dummy_path}."
                    )
                    loaded_algos[symbol] = dummy_algo
                else:
                    # Check file extension to determine format
                    file_ext = Path(model_path).suffix.lower()

                    if file_ext in [".keras", ".h5"]:
                        # Keras model - convert automatically
                        algorithm = self._convert_keras_to_pkl(model_path, symbol)
                        self.logger.info(
                            f"Successfully loaded and converted Keras model for {symbol}"
                        )
                    elif file_ext == ".pkl":
                        # Standard pickle format
                        # Log (but don't crash on) sklearn version mismatches.
                        # Models trained with a newer sklearn can usually still
                        # predict correctly on a slightly older version.
                        self._check_version_metadata(model_path)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            with open(model_path, "rb") as f:
                                algorithm = pickle.load(f)
                            for w in caught:
                                self.logger.warning(
                                    f"Pickle load warning for {symbol}: {w.message}"
                                )

                        # Check if this is a Keras model that needs sequence_length detection
                        model_type = asset_config.get("model_type", "auto")
                        if (
                            model_type in ["lstm", "lstm_optimized"]
                            or "lstm" in model_path.lower()
                        ):
                            # Try to detect sequence_length from model's input shape
                            if hasattr(algorithm, "model") and hasattr(
                                algorithm.model, "input_shape"
                            ):
                                # It's a wrapped model (LSTMTradingModel or similar)
                                input_shape = algorithm.model.input_shape
                                if len(input_shape) == 3 and input_shape[1] is not None:
                                    detected_seq_len = input_shape[1]
                                    if (
                                        not hasattr(algorithm, "sequence_length")
                                        or algorithm.sequence_length != detected_seq_len
                                    ):
                                        algorithm.sequence_length = detected_seq_len
                                        self.logger.info(
                                            f"Auto-detected sequence_length={detected_seq_len} for {symbol} from model input shape"
                                        )
                            elif hasattr(algorithm, "input_shape"):
                                # Direct Keras model
                                input_shape = algorithm.input_shape
                                if len(input_shape) == 3 and input_shape[1] is not None:
                                    algorithm.sequence_length = input_shape[1]
                                    self.logger.info(
                                        f"Auto-detected sequence_length={input_shape[1]} for {symbol} from Keras model"
                                    )

                        self.logger.info(
                            f"Successfully loaded ML algorithm for {symbol} from {model_path} "
                            f"(type: {type(algorithm).__module__}.{type(algorithm).__name__})"
                        )
                    elif file_ext == ".joblib":
                        # Joblib format (used by some scikit-learn models)
                        try:
                            import joblib

                            algorithm = joblib.load(model_path)
                            self.logger.info(
                                f"Successfully loaded joblib algorithm for {symbol} from {model_path}"
                            )
                        except ImportError:
                            self.logger.warning(
                                f"joblib not available, trying pickle for {model_path}"
                            )
                            with open(model_path, "rb") as f:
                                algorithm = pickle.load(f)
                    else:
                        # Try pickle by default
                        with open(model_path, "rb") as f:
                            algorithm = pickle.load(f)
                        self.logger.info(
                            f"Successfully loaded ML algorithm for {symbol} from {model_path}"
                        )

                    loaded_algos[symbol] = algorithm

            except Exception as e:
                self.logger.error(
                    f"Error loading algorithm for {symbol} from {model_path}: {e}",
                    exc_info=True,
                )
                # Create dummy as fallback
                dummy_algo = DummyAlgorithm()
                loaded_algos[symbol] = dummy_algo
                self.logger.warning(
                    f"Using dummy algorithm for {symbol} due to loading error"
                )

        return loaded_algos

    def generate_signal(self, symbol):
        """
        Generates a trading signal (+1 for buy, -1 for sell) for a given symbol
        using its specific loaded ML model.
        CRITICAL: Returns signals matching backtest behavior EXACTLY.

        Returns:
            int: Binary signal (-1 or +1)
                 May return 0 ONLY if Linear Regression predicts exactly 0.0
                 (to match backtest behavior)
        """
        algorithm = self.algorithms.get(symbol)
        if algorithm is None:
            self.logger.error(
                f"CRITICAL: ML algorithm not loaded for {symbol}. "
                f"Check model file exists and is valid."
            )
            self.logger.error(
                f"Expected path: {ASSET_SPECIFIC_CONFIGS.get(symbol, {}).get('strategy_model_path')}"
            )
            raise RuntimeError(f"Algorithm not loaded for {symbol}")

        # Get model configuration from both sources
        config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
        model_type = config.get("model_type", "standard")
        lags = config.get("lags", self.lags)  # Use symbol-specific lags or default

        # Get loaded configuration for this symbol
        loaded_config = self.config_loader.get_model_config(symbol)

        # Use seed from loaded config if available, otherwise generate
        if loaded_config["seed_info"] and "seed" in loaded_config["seed_info"]:
            seed = loaded_config["seed_info"]["seed"]
            self.logger.debug(f"Using loaded seed {seed} for {symbol}")
            set_seed(seed)  # Set the exact seed from training
        else:
            model_name = loaded_config.get("model_type", model_type) or "default"
            seed = set_seed(model_name=model_name, ticker=symbol)
            self.logger.debug(f"Set seed {seed} for {model_name} model on {symbol}")

        try:
            # Handle ARIMA models differently - they use forecast() not predict()
            if model_type == "arima":
                # ARIMA models don't use external features, they forecast based on their internal state
                self.logger.debug(f"Using ARIMA forecast for {symbol}")
                try:
                    # ARIMA uses forecast() method for out-of-sample predictions
                    forecast_result = algorithm.forecast(steps=1)
                    # Handle both array and scalar returns
                    if hasattr(forecast_result, "__len__"):
                        raw_signal = forecast_result[0]
                    else:
                        raw_signal = forecast_result

                    self.logger.debug(f"ARIMA forecast for {symbol}: {raw_signal}")
                except AttributeError:
                    # If forecast() doesn't exist, try get_forecast() for statsmodels ARIMAResults
                    self.logger.debug(f"Trying get_forecast() for {symbol}")
                    forecast_obj = algorithm.get_forecast(steps=1)
                    raw_signal = forecast_obj.predicted_mean[0]
                    self.logger.debug(f"ARIMA get_forecast for {symbol}: {raw_signal}")

            elif model_type == "var":
                # VAR models (Vector AutoRegression from statsmodels)
                # CRITICAL: VAR.forecast() REQUIRES initial values (y parameter)
                self.logger.debug(f"Using VAR forecast for {symbol}")

                try:
                    # Get lag order from model
                    lag_order = algorithm.k_ar

                    # Get initial values: last k_ar observations from embedded training data
                    # This is a pragmatic solution - uses training data endpoint
                    # Future enhancement: Use fresh market data for better accuracy
                    if hasattr(algorithm, "endog") and algorithm.endog is not None:
                        initial_values = algorithm.endog[-lag_order:, :]
                        self.logger.debug(
                            f"Using embedded training data for {symbol} VAR forecast (last {lag_order} obs, shape {initial_values.shape})"
                        )
                    else:
                        raise RuntimeError(f"VAR model for {symbol} missing endog data")

                    # VAR.forecast(y, steps) returns shape (steps, n_vars)
                    forecast_result = algorithm.forecast(y=initial_values, steps=1)

                    # Extract first variable's forecast (typically 'returns' or 'y1')
                    # forecast_result shape: (1, n_vars) → extract [0, 0] for first variable, first step
                    raw_signal = forecast_result[0, 0]

                    var_names = (
                        algorithm.names if hasattr(algorithm, "names") else ["y1"]
                    )
                    self.logger.debug(
                        f"VAR forecast for {symbol}: {raw_signal:.6f} (variable '{var_names[0]}', {algorithm.neqs} total variables)"
                    )

                except AttributeError as e:
                    self.logger.error(
                        f"VAR model for {symbol} missing required attributes: {e}"
                    )
                    raise RuntimeError(f"VAR model incompatible for {symbol}: {e}")
                except Exception as e:
                    self.logger.error(f"VAR forecast failed for {symbol}: {e}")
                    raise RuntimeError(f"VAR model prediction failed for {symbol}: {e}")

            elif (
                model_type in ["lstm", "lstm_optimized"]
                or "lstm" in config.get("strategy_model_path", "").lower()
            ):
                # Use sequence data for LSTM models (both standard and optimized)
                # CRITICAL: Use model's sequence_length if available (auto-detected from model shape)
                # This ensures data shape matches what the model was trained with
                model_seq_length = getattr(algorithm, "sequence_length", None)
                if model_seq_length is not None and model_seq_length != lags:
                    self.logger.info(
                        f"Using model's sequence_length={model_seq_length} instead of config lags={lags} for {symbol}"
                    )
                    effective_lags = model_seq_length
                else:
                    effective_lags = lags
                features = self.data_manager.create_sequence_data(
                    symbol, lags=effective_lags
                )
                self.logger.debug(
                    f"Using LSTM sequence data for {symbol}: shape {features.shape}, lags={effective_lags}"
                )

                if features is None or (
                    hasattr(features, "size") and features.size == 0
                ):
                    self.logger.error(
                        f"CRITICAL: Not enough data for {symbol} to create LSTM sequence. "
                        f"Required: {lags} timesteps, Available: check data_manager"
                    )
                    raise RuntimeError(f"Insufficient data for LSTM model {symbol}")

                # Get raw prediction from model
                raw_signal = algorithm.predict(features)[0]

            else:
                # Use regular features for all other models
                # Includes: li_reg, svm, svm_optimized, xgb, xgb_optimized, linear_optimized,
                #           rf_optimized, ensemble_optimized, logistic_optimized, sgd_optimized

                # Try feature engine first (if model was trained with it)
                feature_config = self._feature_configs.get(symbol)
                if feature_config is not None:
                    ext_data = (
                        self._external_data_cache.get_data()
                        if self._external_data_cache is not None
                        else None
                    )
                    features, feat_names = self.data_manager.get_engineered_features(
                        symbol,
                        feature_config=feature_config,
                        external_data=ext_data,
                        lags=lags,
                    )
                    if features is not None:
                        self.logger.debug(
                            f"Using engineered features for {symbol}: shape {features.shape}, "
                            f"n_features={len(feat_names)}, model_type={model_type}"
                        )
                    else:
                        self.logger.warning(
                            f"Feature engine failed for {symbol}, falling back to legacy lags"
                        )
                        features = self.data_manager.get_data_for_strategy(symbol, lags)
                else:
                    # Legacy path: lagged returns only
                    features = self.data_manager.get_data_for_strategy(symbol, lags)

                if features is not None:
                    self.logger.debug(
                        f"Features for {symbol}: shape {features.shape}, model_type={model_type}"
                    )

                if features is None or (
                    hasattr(features, "size") and features.size == 0
                ):
                    self.logger.error(
                        f"CRITICAL: Not enough data for {symbol} to generate features. "
                        f"Required: {lags} periods, Check data fetching"
                    )
                    raise RuntimeError(f"Insufficient data for model {symbol}")

                # Get raw prediction from model
                # For optimized models, predict() is defined in BaseStrategyModel
                raw_signal = algorithm.predict(features)[0]

            # Convert to binary signal based on model type
            binary_signal = self._convert_to_binary_signal(
                raw_signal, model_type, symbol
            )

            # Log with features shape only if features were used
            # ARIMA and VAR don't use features - they use forecast() based on internal state
            if model_type in ["arima", "var"]:
                self.logger.info(
                    f"ML signal for {symbol}: {binary_signal} (raw: {raw_signal:.6f}, model_type: {model_type})"
                )
            else:
                self.logger.info(
                    f"ML signal for {symbol}: {binary_signal} (raw: {raw_signal:.6f}, model_type: {model_type}, features shape: {features.shape})"
                )

            # Revision Protocol — structured signal log (Phase 1.1).
            # Best-effort; never crashes the trading loop. Skipped entirely
            # when log_signals is False (e.g. shadow_check re-runs).
            if _log_signal_history is not None and getattr(self, "log_signals", True):
                try:
                    _features_for_log = None
                    if model_type not in ("arima", "var"):
                        _features_for_log = features
                    _asset_cfg = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
                    _target_weight = 0.0
                    try:
                        from config import TARGET_ALLOCATION as _TA
                        _target_weight = float(_TA.get(symbol, 0.0))
                    except Exception:
                        pass
                    _log_signal_history(
                        region=getattr(self, "current_region", "UNKNOWN"),
                        ticker=symbol,
                        model_type=str(model_type),
                        strategy_type=str(_asset_cfg.get("strategy_type", "ml_signal")),
                        raw_score=float(raw_signal),
                        signal=int(binary_signal),
                        features=_features_for_log,
                        target_weight=_target_weight,
                        kelly_fraction_used=float(_asset_cfg.get("kelly_fraction", 1.0)),
                    )
                except Exception as _exc:
                    self.logger.debug(
                        "signal_history log failed for %s: %s", symbol, _exc
                    )

            return binary_signal

        except RuntimeError:
            # Re-raise runtime errors (data/model loading issues)
            raise
        except Exception as e:
            self.logger.error(
                f"CRITICAL: Unexpected error predicting signal for {symbol}: {e}",
                exc_info=True,
            )
            self.logger.error(f"Model type: {model_type}, Config: {config}")
            raise RuntimeError(f"Signal generation failed for {symbol}: {e}") from e

    def _convert_to_binary_signal(self, raw_signal, model_type, symbol):
        """
        Convert model output to binary trading signal (-1 or +1).
        CRITICAL: Must match backtest signal generation EXACTLY.

        Args:
            raw_signal: The raw output from the model (predicted return or signal)
            model_type: Type of model (lstm, arima, li_reg, svm, etc.)
            symbol: Trading symbol for logging

        Returns:
            Binary signal: -1 (sell) or +1 (buy)
        """
        # Handle numpy arrays or scalars
        if hasattr(raw_signal, "item"):
            raw_value = raw_signal.item()
        else:
            raw_value = float(raw_signal)

        # === ARIMA Models: Implement threshold-based logic ===
        if model_type == "arima":
            # Load ARIMA settings for this symbol
            arima_config = self.arima_settings.get(symbol, {})
            signal_method = arima_config.get("signal_method", "simple")
            threshold = arima_config.get("threshold", 0.0002)

            if signal_method == "threshold":
                # Match backtest logic from arima_model.py lines 162-167
                # if prediction > threshold: signal = 1
                # elif prediction < -threshold: signal = -1
                # else: signal = 0 (then forward-fill or default to 1)

                if raw_value > threshold:
                    binary_signal = 1
                elif raw_value < -threshold:
                    binary_signal = -1
                else:
                    # Prediction within threshold range: default to 1 (buy-and-hold)
                    # This matches backtest: .replace(0, np.nan).fillna(1)
                    binary_signal = 1

                self.logger.debug(
                    f"ARIMA {symbol}: prediction={raw_value:.6f}, "
                    f"threshold=±{threshold}, signal={binary_signal}"
                )
            else:
                # Other signal methods (simple, z_score, percentile): use np.sign
                binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        # === LSTM Models: Use np.sign with zero → 1 ===
        elif model_type in [
            "lstm",
            "lstm_optimized",
            "dqn",
            "cnn",
            "tcn",
            "rnn",
            "dnn",
        ]:
            # Match backtest logic from lstm_model.py lines 129-134
            # Uses np.sign(), then converts 0 → 1
            binary_signal = int(np.sign(raw_value))
            if binary_signal == 0:
                binary_signal = 1  # Match backtest: predictions[predictions == 0] = 1
                self.logger.debug(f"{model_type.upper()} {symbol}: converted 0 to 1")

        # === Linear Models: Use np.sign (can be 0) ===
        elif model_type in [
            "li_reg",
            "linear_regression",
            "linear_optimized",
            "log_reg",
            "logistic_optimized",
            "sgd_optimized",
        ]:
            # Match backtest logic from linear_regression_model.py lines 95-96
            # Uses np.sign() which can return 0
            binary_signal = int(np.sign(raw_value))

            # Note: Backtest doesn't handle zero case explicitly
            # If raw_value is exactly 0.0, this will return 0
            # This is intentional to match backtest behavior
            if binary_signal == 0:
                self.logger.warning(
                    f"{model_type} {symbol}: prediction is exactly 0, "
                    f"returning 0 signal (matches backtest)"
                )

        # === SVM Models: Use np.sign ===
        elif model_type in ["svm", "svm_optimized"]:
            # SVM typically outputs -1 or 1 already, but use np.sign to be safe
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        # === Naive Bayes / Classification Models: Use np.sign ===
        elif model_type in [
            "gnb",
            "gnb_optimized",
            "naive_bayes",
            "kmeans",
            "kmeans_optimized",
        ]:
            # GNB and other classifiers typically output class labels directly
            # Use np.sign for consistency
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        # === Tree-based Models: Use np.sign ===
        elif model_type in [
            "gbm",
            "rf",
            "rf_optimized",
            "xgboost",
            "xgb_optimized",
            "random_forest_optimized",
        ]:
            # Tree models: use np.sign
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        # === Ensemble Models: Use np.sign ===
        elif model_type in ["ensemble_optimized", "ensemble_adaptive", "ensemble"]:
            # Ensemble models return aggregated predictions, use np.sign
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        # === VAR Models: Use np.sign ===
        elif model_type == "var":
            # VAR models: forecast returns numeric value, use np.sign()
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1

        else:
            # Unknown model type: use np.sign and log warning
            binary_signal = int(np.sign(raw_value)) if raw_value != 0 else 1
            self.logger.warning(
                f"Unknown model type '{model_type}' for {symbol}, "
                f"using default np.sign() conversion"
            )

        return int(binary_signal)

    # ========================================================================
    # Carry Trade Signal Generation
    # ========================================================================

    def generate_carry_signal(
        self, pair_ticker: str, model_cfg: dict
    ) -> Optional[float]:
        """Generate ML carry trade signal for a forex pair.

        Loads the carry trade model and generates a conversion fraction based
        on predict_proba() output. This is called by CashPortfolioManager
        during Phase 2 of the cash rebalancing pipeline.

        Args:
            pair_ticker: Forex pair ticker (e.g., 'USDJPY').
            model_cfg: Model config dict from CASH_PORTFOLIO_CONFIG['carry_trade_models'].
                Must contain: 'model_type', 'strategy_model_path', 'lags'.

        Returns:
            Conversion fraction [0.0, 1.0]:
                0.0 = hold (do not convert to JPY)
                1.0 = fully convert to JPY
                Between = partial conversion based on model confidence
            None on error.
        """
        model_path_str = model_cfg.get("strategy_model_path")
        model_type = model_cfg.get("model_type", "gnb")
        lags = model_cfg.get("lags", 5)

        if not model_path_str:
            self.logger.error(f"No model path for {pair_ticker}")
            return None

        model_path = Path(__file__).parent / model_path_str
        if not model_path.exists():
            self.logger.error(f"Carry model file not found: {model_path}")
            return None

        try:
            # Load model (catch sklearn version warnings)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                for w in caught:
                    self.logger.warning(
                        f"Pickle load warning for carry model {pair_ticker}: {w.message}"
                    )

            # Generate features from recent price data
            features = self._get_carry_features(pair_ticker, lags)
            if features is None:
                return None

            # Apply scaler if available
            scaler_path = model_path.parent / f"carry_{pair_ticker}_scaler.pkl"
            if scaler_path.exists():
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                features = scaler.transform(features)

            # Get prediction with probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)
                p_up = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]

                # Convert probability to conversion fraction:
                # p_up > 0.6: JPY likely weakening -> convert now (lock in rate)
                # p_up < 0.4: JPY likely strengthening -> hold (wait for better rate)
                # 0.4-0.6: partial conversion proportional to confidence
                if p_up > 0.6:
                    fraction = min(1.0, (p_up - 0.5) * 4)
                elif p_up < 0.4:
                    fraction = 0.0
                else:
                    fraction = (p_up - 0.4) * 5

                self.logger.info(
                    f"  Carry signal {pair_ticker}: P(up)={p_up:.3f}, "
                    f"fraction={fraction:.2f} (model={model_type})"
                )
                return fraction

            else:
                # Binary prediction fallback
                prediction = model.predict(features)
                signal = int(prediction[0])
                self.logger.info(
                    f"  Carry signal {pair_ticker}: signal={signal} "
                    f"(binary, model={model_type})"
                )
                return 1.0 if signal == 1 else 0.0

        except Exception as e:
            self.logger.error(f"Error generating carry signal for {pair_ticker}: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def _get_carry_features(self, pair_ticker: str, lags: int) -> Optional[np.ndarray]:
        """Generate lagged return features for a carry trade pair from recent data.

        Args:
            pair_ticker: Forex pair ticker (e.g., 'USDJPY').
            lags: Number of lag periods.

        Returns:
            Numpy array of shape (1, lags) or None on error.
        """
        try:
            from algos.common.market_data_store import MarketDataStore
            from datetime import datetime, timedelta

            store = MarketDataStore()
            yf_ticker = f"{pair_ticker}=X"

            if not store.has_ticker(yf_ticker):
                self.logger.warning(f"No parquet data for {yf_ticker}")
                return None

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

            df = store.get_ohlcv(yf_ticker, start_date, end_date)
            if df is None or len(df) < lags + 2:
                self.logger.warning(
                    f"Insufficient data for {yf_ticker}: "
                    f"{len(df) if df is not None else 0} rows"
                )
                return None

            # Compute log returns
            close = df["Close"] if "Close" in df.columns else df["close"]
            returns = np.log(close / close.shift(1)).dropna()

            # Create lagged features
            import pandas as pd

            feature_data = {}
            for lag in range(1, lags + 1):
                feature_data[f"lag_{lag}"] = returns.shift(lag)

            features_df = pd.DataFrame(feature_data, index=returns.index).dropna()

            if features_df.empty:
                self.logger.warning(
                    f"No features after lag computation for {yf_ticker}"
                )
                return None

            # Return most recent row
            return features_df.iloc[[-1]].values

        except Exception as e:
            self.logger.error(f"Error generating carry features for {pair_ticker}: {e}")
            return None
