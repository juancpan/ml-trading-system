"""
Base class for all trading strategy models.
Provides common functionality to reduce code duplication across models.

Supports two feature engineering modes:
  1. Legacy: Lagged log-returns only (backward compatible, feature_config=None)
  2. Feature Engine: Full technical indicators via FeatureConfig (feature_config provided)
"""

import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)

# Feature engine imports (optional — graceful fallback if not available)
try:
    from algos.common.feature_engine import (
        FeatureEngine,
        FeatureConfig,
        save_feature_metadata,
    )

    _HAS_FEATURE_ENGINE = True
except ImportError:
    _HAS_FEATURE_ENGINE = False


class BaseStrategyModel(ABC):
    """
    Abstract base class for trading strategy models.
    Handles common preprocessing, validation, and strategy calculation.
    """

    def __init__(self, model_name: str = "BaseModel"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()

    def validate_data(self, data: pd.DataFrame, log_prefix: str) -> None:
        """Validate input data has required columns and is not empty."""
        if data.empty:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
                )
            sys.exit(1)

        required_cols = ["returns", "direction"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Fatal Error: Input data for {log_prefix} is missing columns: {missing_cols}. Exiting."
                )
            sys.exit(1)

    def create_lagged_features(
        self, data: pd.DataFrame, lags: int, log_prefix: str
    ) -> tuple:
        """Create lagged return features for the model (legacy method)."""
        cols = []
        for lag in range(1, lags + 1):
            col = f"lag_{lag}"
            data[col] = data["returns"].shift(lag)
            cols.append(col)

        data.dropna(inplace=True)

        if data.empty:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Fatal Error: DataFrame empty after creating lags for {log_prefix}. Exiting."
                )
            sys.exit(1)

        return data, cols

    def create_features(
        self,
        data: pd.DataFrame,
        lags: int,
        log_prefix: str,
        feature_config: Optional[object] = None,
        external_data: Optional[dict] = None,
    ) -> tuple:
        """
        Create features using FeatureEngine (if config provided) or legacy lags.

        When feature_config is provided, uses the centralized FeatureEngine for full
        technical indicator computation. Otherwise falls back to legacy lagged returns.

        Args:
            data: Input DataFrame with OHLCV + returns + direction columns
            lags: Number of lag features (used only in legacy fallback mode)
            log_prefix: Logging prefix
            feature_config: FeatureConfig instance (None = legacy mode)
            external_data: Dict mapping feature_name -> pd.Series

        Returns:
            (data, feature_cols) — augmented DataFrame and list of feature column names
        """
        if feature_config is not None and _HAS_FEATURE_ENGINE:
            engine = FeatureEngine()
            data, feature_cols = engine.compute_features(
                data, feature_config, external_data
            )

            if not feature_cols:
                print(
                    f"Warning: FeatureEngine produced no features for {log_prefix}. "
                    f"Falling back to legacy lagged features."
                )
                return self.create_lagged_features(data, lags, log_prefix)

            if data.empty:
                with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                    print(
                        f"Fatal Error: DataFrame empty after feature engineering "
                        f"for {log_prefix}. Exiting."
                    )
                sys.exit(1)

            # Store feature metadata on the instance for downstream saving
            self._feature_cols = feature_cols
            self._feature_config = feature_config

            print(
                f"Feature engineering: {len(feature_cols)} features computed "
                f"({len(data)} rows after warmup)"
            )
            return data, feature_cols
        else:
            # Legacy fallback: just lagged returns
            return self.create_lagged_features(data, lags, log_prefix)

    def split_data(
        self,
        data: pd.DataFrame,
        split_ratio: float,
        log_prefix: str,
        embargo_pct: float = 0.02,
        interval: str = "1d",
        train_end_date: Optional[str] = None,
        test_start_date: Optional[str] = None,
    ) -> tuple:
        """
        Split data into train and test sets with embargo buffer.

        Supports two modes:
        1. Date-based split (WFOV walk-forward modes): Uses explicit train_end_date
           and test_start_date boundaries. Overrides ratio-based splitting.
        2. Ratio-based split (default): Uses split_ratio with optional embargo buffer.

        Args:
            data: Input DataFrame
            split_ratio: Proportion of data for training (e.g., 0.5 = 50%)
            log_prefix: Logging prefix
            embargo_pct: Embargo as percentage of total data (default 2%)
                        Set to 0 to disable embargo (backward compatibility)
            interval: Data interval for auto-adjusting minimums ('1d', '1wk', '1mo')
            train_end_date: Explicit train end date (overrides split_ratio)
            test_start_date: Explicit test start date (after embargo gap)

        Returns:
            (train, test): DataFrames with embargo buffer applied

        Example with embargo_pct=0.02, split_ratio=0.5, 1000 samples:
            Train:   samples 0-499 (500 samples)
            Embargo: samples 500-519 (20 samples) - DISCARDED
            Test:    samples 520-999 (480 samples)
        """
        # Date-based split (used by WFOV walk-forward modes)
        if train_end_date and test_start_date:
            train = data.loc[:train_end_date].copy()
            test = data.loc[test_start_date:].copy()

            if len(train) < 50 or len(test) < 10:
                with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                    print(
                        f"Fatal Error: Insufficient data for date-based split in {log_prefix}. "
                        f"Train: {len(train)}, Test: {len(test)}"
                    )
                sys.exit(1)

            print(
                f"Date-based split: train {train.index[0].strftime('%Y-%m-%d')} to "
                f"{train.index[-1].strftime('%Y-%m-%d')} ({len(train)} samples), "
                f"test {test.index[0].strftime('%Y-%m-%d')} to "
                f"{test.index[-1].strftime('%Y-%m-%d')} ({len(test)} samples)"
            )
            return train, test

        # Original ratio-based split (backward compatibility)
        # Calculate initial split point
        initial_split = int(len(data) * split_ratio)

        # Apply embargo if enabled
        if embargo_pct > 0:
            # Calculate embargo size (interval-aware)
            embargo_size = calculate_embargo_size(
                len(data), embargo_pct=embargo_pct, interval=interval
            )

            # Split with embargo buffer
            # Train ends at initial_split
            # Embargo: samples [initial_split, initial_split + embargo_size) - DISCARDED
            # Test starts at initial_split + embargo_size
            train = data.iloc[:initial_split].copy()
            test = data.iloc[initial_split + embargo_size :].copy()

            # Validate the split (interval-aware)
            validate_embargo_split(
                train_size=len(train),
                embargo_size=embargo_size,
                test_size=len(test),
                interval=interval,
            )

            # Log embargo configuration (direct print, no nested redirect)
            print(
                get_embargo_message(
                    embargo_size=embargo_size,
                    data_length=len(data),
                    train_size=len(train),
                    test_size=len(test),
                )
            )
        else:
            # No embargo (backward compatibility)
            train = data.iloc[:initial_split].copy()
            test = data.iloc[initial_split:].copy()

            # Log no embargo (direct print, no nested redirect)
            print(
                "\nNote: Embargo disabled (embargo_pct=0). Using simple train/test split."
            )

        # Original validation
        if train.empty or test.empty:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Fatal Error: Insufficient data for train/test split in {log_prefix}."
                )
            sys.exit(1)

        return train, test

    def scale_features(
        self, train: pd.DataFrame, test: pd.DataFrame, cols: list
    ) -> tuple:
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(train[cols])
        X_test_scaled = self.scaler.transform(test[cols])
        return X_train_scaled, X_test_scaled

    @abstractmethod
    def train_model(
        self, X_train: np.ndarray, y_train: pd.Series, log_prefix: str
    ) -> None:
        """Train the specific model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass

    def calculate_performance_metrics(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        train_predictions: np.ndarray,
        test_predictions: np.ndarray,
        log_prefix: str,
    ) -> None:
        """Calculate and log performance metrics.

        Hit ratios are aligned to execution timing: position[t] is compared
        against direction[t+1] since position.shift(1) * returns gives the
        actual strategy return (signal at t earns return at t+1).
        """
        train_hit_ratio = accuracy_score(train["direction"], train_predictions)

        if (
            not test["direction"].empty
            and not test["position"].empty
            and len(test["direction"]) == len(test["position"])
        ):
            # Execution-aligned hit ratio: position[t] predicts direction[t+1]
            # Discretize continuous positions (from conviction sizing) to {-1, 1}
            shifted_pos = np.sign(test["position"].iloc[:-1].values)
            shifted_pos[shifted_pos == 0] = 1  # Treat zero as long (tie-break)
            next_dir = test["direction"].iloc[1:].values
            test_hit_ratio = accuracy_score(next_dir, shifted_pos)
        else:
            test_hit_ratio = np.nan

        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print("=" * 50)
            print("\nModel Performance (Hit Ratios):\n")
            print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
            print(f"Test Hit Ratio (execution-aligned): {test_hit_ratio:.4f}")
            print("=" * 50 + "\n" * 3)

    def calculate_strategy_returns(
        self, test: pd.DataFrame, log_prefix: str
    ) -> pd.DataFrame:
        """Calculate strategy returns with transaction costs."""
        test.dropna(subset=["position", "returns"], inplace=True)

        if test.empty:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Error: Test DataFrame empty after dropping NaNs for {log_prefix}."
                )
            sys.exit(1)

        # Calculate raw strategy returns (shift position by 1 bar to avoid look-ahead bias)
        test["strategy"] = test["position"].shift(1) * test["returns"]

        # Apply transaction costs
        transaction_cost_log_impact = np.log(1 - PTC)
        position_shifted = test["position"].shift(1)
        position_change = position_shifted.diff().fillna(0).abs()
        # Scale TC by magnitude of position change:
        # 0->+1 = 1 unit cost, -1->+1 = 2 units cost, +1->+1 = 0 cost
        tc_per_bar = position_change * transaction_cost_log_impact
        test["strategy_tc"] = test["strategy"] + tc_per_bar
        # Remove first row NaN from position shift
        test.dropna(subset=["strategy"], inplace=True)

        return test

    def run_strategy(
        self,
        data: pd.DataFrame,
        initial_train_split_ratio: float = 0.5,
        lags: int = 5,
        log_prefix: str = None,
        embargo_pct: float = 0.02,
        interval: str = "1d",
        feature_config: Optional[object] = None,
        external_data: Optional[dict] = None,
        train_end_date: Optional[str] = None,
        test_start_date: Optional[str] = None,
        **kwargs,
    ) -> tuple:
        """
        Main method to run the complete strategy pipeline.

        Args:
            data: Input DataFrame with price and returns data
            initial_train_split_ratio: Training split ratio (default 0.5)
            lags: Number of lag features to create (default 5)
            log_prefix: Logging prefix
            embargo_pct: Embargo percentage (default 0.02 = 2%). Set to 0 to disable.
            interval: Data interval for auto-adjusting minimums ('1d', '1wk', '1mo')
            feature_config: FeatureConfig instance for full feature engineering.
                           If None, falls back to legacy lagged returns.
            external_data: Dict mapping feature_name -> pd.Series for external data
                          (e.g., VIX close). Only used when feature_config is provided.
            train_end_date: Explicit train end date for date-based splitting (WFOV walk-forward)
            test_start_date: Explicit test start date for date-based splitting (WFOV walk-forward)
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (test_results, trained_model)
        """
        if log_prefix is None:
            log_prefix = self.model_name

        # Validate input data
        self.validate_data(data, log_prefix)

        # Create features (uses FeatureEngine if feature_config provided, else legacy lags)
        data, feature_cols = self.create_features(
            data,
            lags,
            log_prefix,
            feature_config=feature_config,
            external_data=external_data,
        )

        # Split data WITH embargo (interval-aware, or date-based for WFOV walk-forward)
        train, test = self.split_data(
            data,
            initial_train_split_ratio,
            log_prefix,
            embargo_pct=embargo_pct,
            interval=interval,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
        )

        # Store feature names for downstream analysis (SHAP, importance)
        self._feature_names = list(feature_cols)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(train, test, feature_cols)

        # Get labels - ensure they are numpy arrays with correct type
        y_train = train["direction"].values
        y_test = test["direction"].values

        # Train model
        self.train_model(X_train_scaled, y_train, log_prefix, **kwargs)

        # Make predictions
        train_predictions = self.predict(X_train_scaled)
        test_predictions = self.predict(X_test_scaled)

        # Add predictions to test DataFrame with conviction-based sizing
        # Models with predict_proba get continuous positions in [-1, +1]
        # Models without get standard binary {-1, +1} positions
        if hasattr(self, "predict_proba") and callable(self.predict_proba):
            try:
                probas = self.predict_proba(X_test_scaled)
                if probas.ndim == 2 and probas.shape[1] >= 2:
                    # Conviction = distance from 0.5, scaled to [0, 1]
                    p_long = probas[:, 1]
                    conviction = np.abs(p_long - 0.5) * 2
                    direction = np.where(p_long > 0.5, 1.0, -1.0)
                    test["position"] = pd.Series(
                        direction * conviction, index=test.index
                    )
                    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                        print(
                            f"Conviction sizing: mean |pos|={conviction.mean():.3f}, "
                            f"max={conviction.max():.3f}, min={conviction.min():.3f}"
                        )
                else:
                    test["position"] = pd.Series(test_predictions, index=test.index)
            except Exception:
                test["position"] = pd.Series(test_predictions, index=test.index)
        else:
            test["position"] = pd.Series(test_predictions, index=test.index)

        # Calculate performance metrics
        self.calculate_performance_metrics(
            train, test, train_predictions, test_predictions, log_prefix
        )

        # Calculate strategy returns
        test = self.calculate_strategy_returns(test, log_prefix)

        return test, self.model
