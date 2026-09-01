"""
Out-of-Fold Stacking Ensemble using temporal cross-validation.

Architecture:
1. Split training data into K temporal folds (TimeSeriesSplit).
2. For each fold k:
   - Train each base model on folds 1..k-1
   - Predict fold k (out-of-fold predictions)
3. Stack: Train meta-learner on OOF predictions (no data leakage).
4. For test data: Train all base models on full train, predict test,
   feed to meta-learner.

Base models: GNB, SVM (linear), XGBoost (the 3 fastest from comprehensive list)
Meta-learner: LogisticRegression with L2 regularization
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile
from typing import Optional


class OOFStackingModel(BaseStrategyModel):
    """Out-of-fold stacking ensemble with temporal cross-validation."""

    def __init__(self):
        super().__init__(model_name="EnsembleOOF")
        self.base_models = None
        self.meta_model = None
        self.base_scalers = None
        self.n_folds = 3

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        log_prefix: str,
        n_folds: int = 3,
        **kwargs,
    ) -> None:
        """
        Train stacking ensemble using out-of-fold predictions.

        Phase 1: Generate OOF predictions via temporal CV
        Phase 2: Train meta-learner on OOF predictions
        Phase 3: Refit all base models on full training data (for test prediction)
        """
        self.n_folds = n_folds

        # Convert labels for XGBoost: {-1, 1} -> {0, 1}
        y_binary = np.where(y_train == -1, 0, 1)

        def make_base_models():
            return [
                ("gnb", GaussianNB()),
                (
                    "svc",
                    SVC(
                        kernel="linear",
                        probability=True,
                        C=1.0,
                        random_state=42,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        verbosity=0,
                        eval_metric="logloss",
                    ),
                ),
            ]

        n_base = 3
        oof_predictions = np.zeros((len(X_train), n_base))
        oof_mask = np.zeros(len(X_train), dtype=bool)

        tscv = TimeSeriesSplit(n_splits=n_folds)

        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"\n{'=' * 50}")
            print(f"OOF Stacking: {n_folds} temporal folds, {n_base} base models")
            print(f"Training samples: {len(X_train)}")

        # Phase 1: Generate OOF predictions
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_train = y_binary[train_idx]

            fold_scaler = StandardScaler()
            X_fold_train_s = fold_scaler.fit_transform(X_fold_train)
            X_fold_val_s = fold_scaler.transform(X_fold_val)

            base_models = make_base_models()
            for i, (name, model) in enumerate(base_models):
                try:
                    model.fit(X_fold_train_s, y_fold_train)
                    proba = model.predict_proba(X_fold_val_s)[:, 1]
                    oof_predictions[val_idx, i] = proba
                except Exception as e:
                    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                        print(f"  Fold {fold_idx}: {name} failed: {e}")
                    oof_predictions[val_idx, i] = 0.5  # neutral fallback

            oof_mask[val_idx] = True

            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"  Fold {fold_idx + 1}/{n_folds}: "
                    f"train={len(train_idx)}, val={len(val_idx)}"
                )

        # Phase 2: Train meta-learner on OOF predictions
        meta_X = oof_predictions[oof_mask]
        meta_y = y_binary[oof_mask]

        self.meta_model = LogisticRegression(
            solver="liblinear", C=1.0, penalty="l2", random_state=42
        )
        self.meta_model.fit(meta_X, meta_y)

        meta_train_acc = self.meta_model.score(meta_X, meta_y)

        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"\nMeta-learner OOF accuracy: {meta_train_acc:.4f}")
            coefs = self.meta_model.coef_[0]
            names = ["GNB", "SVC", "XGB"]
            for name, c in zip(names, coefs):
                print(f"  {name} weight: {c:.4f}")
            print(f"{'=' * 50}\n")

        # Phase 3: Refit all base models on FULL training data
        self.base_models = make_base_models()
        self.base_scalers = StandardScaler()
        X_train_s = self.base_scalers.fit_transform(X_train)

        for name, model in self.base_models:
            model.fit(X_train_s, y_binary)

        # Store meta_model as the primary model for persistence
        self.model = self.meta_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the stacking ensemble.
        Scale -> base model probas -> meta-learner -> {-1, 1} signal.
        """
        X_scaled = self.base_scalers.transform(X)

        base_probas = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            base_probas[:, i] = model.predict_proba(X_scaled)[:, 1]

        meta_pred = self.meta_model.predict(base_probas)
        return np.where(meta_pred == 1, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get meta-learner probabilities for conviction sizing."""
        X_scaled = self.base_scalers.transform(X)

        base_probas = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            base_probas[:, i] = model.predict_proba(X_scaled)[:, 1]

        return self.meta_model.predict_proba(base_probas)


def run_ensemble_oof_strategy(
    data,
    initial_train_split_ratio=0.5,
    lags=5,
    log_prefix="EnsembleOOF_model",
    embargo_pct=0.02,
    n_folds=3,
    **kwargs,
):
    """Wrapper function for backward compatibility."""
    model = OOFStackingModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        n_folds=n_folds,
        **kwargs,
    )
