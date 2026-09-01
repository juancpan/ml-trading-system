"""
Optimized Random Forest trading strategy model.
Includes feature importance analysis and parallel processing.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile
from typing import Optional, Dict, Any


class RandomForestStrategyModel(BaseStrategyModel):
    """Optimized Random Forest trading strategy with enhanced features."""

    def __init__(self):
        super().__init__(model_name="RandomForest")
        self.feature_importance = None
        self.oob_score = None

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        log_prefix: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Train Random Forest model with optimized parameters.

        Args:
            n_jobs: Number of parallel jobs (-1 uses all cores)
            oob_score: Whether to compute out-of-bag score
            Other args: Standard RandomForest parameters
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,  # Parallel processing
            random_state=random_state,
            class_weight="balanced",  # Handle imbalanced classes
        )

        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print("=" * 50)
            print(f"\nTraining Optimized Random Forest ({log_prefix}):\n")
            print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
            print(f"Using {n_jobs if n_jobs > 0 else 'all'} CPU cores")

            try:
                self.model.fit(X_train, y_train)
                print(f"Model fitted successfully")

                # SHAP feature importance analysis (diagnostic, logged to output)
                try:
                    from algos.common.feature_pruner import compute_shap_importance

                    if hasattr(self, "_feature_names") and self._feature_names:
                        shap_df = compute_shap_importance(
                            self.model, X_train, self._feature_names, max_samples=300
                        )
                        if not shap_df.empty:
                            self.shap_importance = shap_df
                            print(f"\nSHAP Feature Importance (top 15):")
                            print(shap_df.head(15).to_string(index=False))
                            # Show noise threshold
                            from algos.common.feature_pruner import prune_features

                            keep, drop = prune_features(shap_df, method="cumulative_90")
                            if drop:
                                print(
                                    f"\nNoise features (below 90% cumulative): {drop}"
                                )
                except Exception:
                    pass  # SHAP is optional

                # Store feature importance
                self.feature_importance = self.model.feature_importances_
                print(f"\nFeature Importance:")
                for i, importance in enumerate(self.feature_importance):
                    print(f"  Lag {i + 1}: {importance:.4f}")

                # Store OOB score if available
                if oob_score and bootstrap:
                    self.oob_score = self.model.oob_score_
                    print(f"\nOut-of-Bag Score: {self.oob_score:.4f}")

                # Model complexity info
                print(f"\nModel Complexity:")
                print(f"  Total trees: {len(self.model.estimators_)}")
                print(f"  Max features per split: {self.model.max_features}")

            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print("=" * 50 + "\n")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for risk management."""
        return self.model.predict_proba(X)

    def get_model_insights(self) -> Dict[str, Any]:
        """Return model insights for analysis."""
        return {
            "feature_importance": self.feature_importance,
            "oob_score": self.oob_score,
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
        }


def run_random_forest_strategy(
    data,
    initial_train_split_ratio=0.5,
    lags=5,
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    log_prefix="RF_model",
    embargo_pct=0.02,
    **kwargs,
):
    """
    Wrapper function for backward compatibility with embargo.
    Now uses all CPU cores by default for faster training.
    """
    model = RandomForestStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        **kwargs,
    )
