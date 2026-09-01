"""
SHAP-based feature importance analysis and pruning.

Computes SHAP values for tree-based models, identifies noise features,
and returns a ranked importance DataFrame.

Usage:
    from algos.common.feature_pruner import compute_shap_importance, prune_features

    # After training a tree-based model:
    importance_df = compute_shap_importance(model, X_train, feature_names)
    keep, drop = prune_features(importance_df, method="cumulative_90")
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance for a trained tree-based model.

    Args:
        model: Trained sklearn/xgboost model (must support TreeExplainer).
        X_train: Training features (numpy array).
        feature_names: List of feature column names.
        max_samples: Max samples for SHAP computation (speed vs accuracy).

    Returns:
        DataFrame with columns [feature, mean_abs_shap, rank, pct_of_total]
        sorted by importance descending. Empty DataFrame if SHAP fails.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Run: pip install shap")
        return pd.DataFrame()

    # Subsample if too large
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), max_samples, replace=False)
        X_sample = X_train[indices]
    else:
        X_sample = X_train

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (positive/long)

        # Handle 3D arrays (some models return [n_samples, n_features, n_classes])
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        mean_abs = np.abs(shap_values).mean(axis=0)

        # Trim to match feature_names length (in case of mismatch)
        n_features = min(len(mean_abs), len(feature_names))

        importance_df = pd.DataFrame(
            {
                "feature": feature_names[:n_features],
                "mean_abs_shap": mean_abs[:n_features],
            }
        )
        importance_df = importance_df.sort_values(
            "mean_abs_shap", ascending=False
        ).reset_index(drop=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        total = importance_df["mean_abs_shap"].sum()
        if total > 0:
            importance_df["pct_of_total"] = importance_df["mean_abs_shap"] / total
            importance_df["cumulative_pct"] = importance_df["pct_of_total"].cumsum()
        else:
            importance_df["pct_of_total"] = 0.0
            importance_df["cumulative_pct"] = 0.0

        return importance_df

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return pd.DataFrame()


def get_noise_threshold(
    importance_df: pd.DataFrame, method: str = "cumulative_90"
) -> float:
    """
    Determine the SHAP importance threshold below which features are noise.

    Methods:
        'cumulative_90': Keep features that explain 90% of total SHAP importance.
        'cumulative_95': Keep features that explain 95% of total importance.
        'mean': Drop features below mean importance.
        'elbow': Second derivative method on sorted importance curve.
    """
    if importance_df.empty:
        return 0.0

    values = importance_df["mean_abs_shap"].values

    if method.startswith("cumulative_"):
        pct = int(method.split("_")[1]) / 100.0
        total = values.sum()
        if total == 0:
            return 0.0
        cumsum = np.cumsum(values)
        cutoff_idx = np.searchsorted(cumsum, total * pct)
        return values[min(cutoff_idx, len(values) - 1)]

    elif method == "mean":
        return float(values.mean())

    elif method == "elbow":
        if len(values) < 3:
            return float(values.min()) if len(values) > 0 else 0.0
        diffs = np.diff(values)
        second_diffs = np.diff(diffs)
        if len(second_diffs) == 0:
            return float(values.min())
        elbow_idx = np.argmax(np.abs(second_diffs)) + 1
        return float(values[min(elbow_idx, len(values) - 1)])

    return 0.0


def prune_features(
    importance_df: pd.DataFrame,
    method: str = "cumulative_90",
    min_features: int = 5,
) -> Tuple[List[str], List[str]]:
    """
    Return (features_to_keep, features_to_drop) based on SHAP importance.

    Always keeps at least min_features.
    """
    if importance_df.empty:
        return [], []

    threshold = get_noise_threshold(importance_df, method=method)

    keep = importance_df[importance_df["mean_abs_shap"] >= threshold][
        "feature"
    ].tolist()
    drop = importance_df[importance_df["mean_abs_shap"] < threshold]["feature"].tolist()

    # Ensure minimum features
    if len(keep) < min_features:
        keep = importance_df.head(min_features)["feature"].tolist()
        drop = importance_df.iloc[min_features:]["feature"].tolist()

    return keep, drop
