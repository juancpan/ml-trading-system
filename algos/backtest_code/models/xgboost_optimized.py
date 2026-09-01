"""
Optimized XGBoost trading strategy with GPU support and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile
from typing import Dict, Any, Optional


class XGBoostStrategyModel(BaseStrategyModel):
    """
    Optimized XGBoost trading strategy with advanced features.
    """

    def __init__(self):
        super().__init__(model_name="XGBoost")
        self.feature_importance = None
        self.best_params = None

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        log_prefix: str,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        use_gpu: bool = False,
        auto_tune: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Train XGBoost model with optimization features.

        Args:
            use_gpu: Whether to use GPU acceleration
            auto_tune: Whether to perform hyperparameter tuning
            Other args: Standard XGBoost parameters
        """
        # Convert labels from {-1, 1} to {0, 1} for XGBoost
        y_train_binary = np.where(y_train == -1, 0, 1)

        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print("=" * 50)
            print(f"\nTraining Optimized XGBoost ({log_prefix}):\n")

            # Determine tree method based on GPU availability
            # XGBoost 2.0+ supports 'device' parameter for Apple Silicon GPU via Metal
            if use_gpu == "auto":
                import platform

                if platform.machine() == "arm64" and platform.system() == "Darwin":
                    # Apple Silicon: use hist with device=cuda (XGBoost 2.0+)
                    tree_method = "hist"
                    # Note: XGBoost on Apple Silicon doesn't use gpu_hist;
                    # GPU acceleration is handled via the 'device' param in XGBClassifier
                    use_gpu = True
                    print("Auto-detected Apple Silicon, enabling GPU acceleration")
                else:
                    use_gpu = False
                    tree_method = "hist"
            else:
                tree_method = "gpu_hist" if use_gpu else "hist"

            if auto_tune:
                print("Performing hyperparameter tuning...")
                best_params = self._tune_hyperparameters(
                    X_train, y_train_binary, tree_method, n_jobs
                )
                self.best_params = best_params

                # Update parameters with tuned values
                n_estimators = best_params.get("n_estimators", n_estimators)
                max_depth = best_params.get("max_depth", max_depth)
                learning_rate = best_params.get("learning_rate", learning_rate)
                subsample = best_params.get("subsample", subsample)
                colsample_bytree = best_params.get("colsample_bytree", colsample_bytree)
                gamma = best_params.get("gamma", gamma)
                reg_alpha = best_params.get("reg_alpha", reg_alpha)
                reg_lambda = best_params.get("reg_lambda", reg_lambda)

                print(f"Best parameters found: {best_params}")

            # Create optimized XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                tree_method=tree_method,
                n_jobs=n_jobs,
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                enable_categorical=False,
                early_stopping_rounds=10,
                verbosity=0,
            )

            print(f"Training with tree_method='{tree_method}', n_jobs={n_jobs}")

            try:
                # Temporal validation split: use last 20% of training data for early stopping
                val_split_idx = int(len(X_train) * 0.8)
                X_train_fit, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
                y_train_fit, y_val = (
                    y_train_binary[:val_split_idx],
                    y_train_binary[val_split_idx:],
                )

                eval_set = [(X_val, y_val)]
                self.model.fit(
                    X_train_fit, y_train_fit, eval_set=eval_set, verbose=False
                )

                print(f"Model fitted successfully")
                print(f"Best iteration: {self.model.best_iteration}")
                print(f"Best score: {self.model.best_score:.4f}")

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
                print(f"\nFeature Importance (Gain):")
                for i, importance in enumerate(self.feature_importance):
                    print(f"  Lag {i + 1}: {importance:.4f}")

                # Get additional importance metrics
                importance_types = ["weight", "gain", "cover"]
                for imp_type in importance_types:
                    try:
                        imp = self.model.get_booster().get_score(
                            importance_type=imp_type
                        )
                        if imp:
                            print(f"\nFeature Importance ({imp_type}):")
                            for feat, score in sorted(
                                imp.items(), key=lambda x: x[1], reverse=True
                            ):
                                print(f"  {feat}: {score:.4f}")
                    except:
                        pass

            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print("=" * 50 + "\n")

    def _tune_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, tree_method: str, n_jobs: int
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        """
        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_alpha": [0, 0.1, 1, 10],
            "reg_lambda": [0.1, 1, 10, 100],
        }

        # Create base model
        base_model = xgb.XGBClassifier(
            tree_method=tree_method,
            n_jobs=n_jobs,
            random_state=42,
            objective="binary:logistic",
            use_label_encoder=False,
            verbosity=0,
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Random search
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,  # Number of parameter combinations to try
            cv=tscv,
            scoring="accuracy",
            n_jobs=1,  # Use single job for search itself
            random_state=42,
            verbose=0,
        )

        random_search.fit(X_train, y_train)

        return random_search.best_params_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        # Convert 0/1 predictions to -1/1 for trading signals
        predictions = self.model.predict(X)
        return np.where(predictions == 1, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def get_model_insights(self) -> Dict[str, Any]:
        """Return model insights for analysis."""
        insights = {
            "feature_importance": self.feature_importance,
            "best_iteration": self.model.best_iteration
            if hasattr(self.model, "best_iteration")
            else None,
            "best_score": self.model.best_score
            if hasattr(self.model, "best_score")
            else None,
            "n_estimators_used": self.model.n_estimators,
        }

        if self.best_params:
            insights["tuned_parameters"] = self.best_params

        return insights


def run_xgboost_strategy(
    data,
    initial_train_split_ratio=0.5,
    lags=5,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_gpu=False,
    auto_tune=False,
    n_jobs=-1,
    log_prefix="XGB_model",
    embargo_pct=0.02,
    **kwargs,
):
    """
    Wrapper function for backward compatibility with optimization and embargo.

    Args:
        use_gpu: Enable GPU acceleration if available
        auto_tune: Enable automatic hyperparameter tuning
        n_jobs: Number of parallel threads (-1 for all cores)
        embargo_pct: Embargo percentage (default 2%)
    """
    model = XGBoostStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_gpu=use_gpu,
        auto_tune=auto_tune,
        n_jobs=n_jobs,
        **kwargs,
    )
