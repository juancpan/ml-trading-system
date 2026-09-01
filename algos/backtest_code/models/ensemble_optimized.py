"""
Optimized Ensemble Model combining multiple strategies with voting and stacking.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import Parallel, delayed
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile
from typing import List, Dict, Any, Optional, Tuple
import pickle


class EnsembleStrategyModel(BaseStrategyModel):
    """
    Advanced ensemble model combining multiple strategies.
    """
    
    def __init__(self):
        super().__init__(model_name="Ensemble")
        self.base_models = {}
        self.model_weights = None
        self.model_performances = {}
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, log_prefix: str,
                   ensemble_type: str = 'voting', base_models: Optional[List[str]] = None,
                   use_probabilities: bool = True, n_jobs: int = -1,
                   auto_weight: bool = True, **kwargs) -> None:
        """
        Train ensemble model with multiple base models.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending')
            base_models: List of base models to use
            use_probabilities: Use soft voting with probabilities
            auto_weight: Automatically weight models based on performance
            n_jobs: Number of parallel jobs
        """
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print('=' * 50)
            print(f"\nTraining Optimized Ensemble Model ({log_prefix}):\n")
            print(f"Ensemble type: {ensemble_type}")
            
            # Default base models if not specified
            if base_models is None:
                base_models = ['rf', 'xgb', 'svm', 'logreg']
            
            print(f"Base models: {base_models}")
            
            # Create base model instances
            self._create_base_models(base_models, n_jobs)
            
            # Train and evaluate base models for weighting
            if auto_weight:
                self.model_weights = self._calculate_model_weights(
                    X_train, y_train, n_jobs
                )
                print(f"\nAuto-calculated weights: {self.model_weights}")
            
            try:
                if ensemble_type == 'voting':
                    self._train_voting_ensemble(
                        X_train, y_train, 
                        use_probabilities, 
                        n_jobs
                    )
                elif ensemble_type == 'stacking':
                    self._train_stacking_ensemble(
                        X_train, y_train, 
                        n_jobs
                    )
                elif ensemble_type == 'blending':
                    self._train_blending_ensemble(
                        X_train, y_train, 
                        n_jobs
                    )
                else:
                    raise ValueError(f"Unknown ensemble type: {ensemble_type}")
                
                print(f"\nEnsemble model fitted successfully")
                
                # Print individual model performances
                if self.model_performances:
                    print("\nIndividual Model Performances:")
                    for name, perf in self.model_performances.items():
                        print(f"  {name}: {perf:.4f}")
                
            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print('=' * 50 + '\n')
    
    def _create_base_models(self, model_names: List[str], n_jobs: int):
        """Create instances of base models."""
        model_configs = {
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                n_jobs=n_jobs, random_state=42
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                tree_method='hist', n_jobs=n_jobs,
                random_state=42, use_label_encoder=False,
                eval_metric='logloss'
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, 
                probability=True, random_state=42
            ),
            'logreg': LogisticRegression(
                penalty='l2', C=1.0, 
                max_iter=1000, n_jobs=n_jobs,
                random_state=42
            )
        }
        
        for name in model_names:
            if name in model_configs:
                self.base_models[name] = model_configs[name]
            else:
                print(f"Warning: Unknown model {name}, skipping")
    
    def _calculate_model_weights(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_jobs: int) -> Dict[str, float]:
        """
        Calculate weights for models based on cross-validation performance.
        """
        from sklearn.model_selection import cross_val_score
        
        # Convert labels for compatibility
        y_train_binary = np.where(y_train == -1, 0, 1)
        
        weights = {}
        
        for name, model in self.base_models.items():
            # Use 3-fold CV for speed
            scores = cross_val_score(
                model, X_train, y_train_binary, 
                cv=3, scoring='accuracy', 
                n_jobs=n_jobs
            )
            avg_score = scores.mean()
            self.model_performances[name] = avg_score
            weights[name] = avg_score
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _train_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                              use_probabilities: bool, n_jobs: int):
        """Train voting classifier ensemble."""
        # Convert labels from {-1, 1} to {0, 1} for compatibility
        y_train_binary = np.where(y_train == -1, 0, 1)
        
        estimators = list(self.base_models.items())
        
        voting_type = 'soft' if use_probabilities else 'hard'
        
        if self.model_weights:
            # Convert weights to list in same order as estimators
            weights = [self.model_weights[name] for name, _ in estimators]
        else:
            weights = None
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            weights=weights,
            n_jobs=n_jobs
        )
        
        self.model.fit(X_train, y_train_binary)
    
    def _train_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_jobs: int):
        """Train stacking classifier ensemble."""
        # Convert labels from {-1, 1} to {0, 1} for compatibility
        y_train_binary = np.where(y_train == -1, 0, 1)
        
        estimators = list(self.base_models.items())
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(
            penalty='l2', C=1.0, 
            max_iter=1000, random_state=42
        )
        
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,  # Use 3-fold CV for training meta-learner
            n_jobs=n_jobs,
            passthrough=False  # Don't include original features
        )
        
        self.model.fit(X_train, y_train_binary)
    
    def _train_blending_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_jobs: int):
        """
        Train blending ensemble (similar to stacking but with holdout set).
        """
        # Convert labels from {-1, 1} to {0, 1} for compatibility
        y_train_binary = np.where(y_train == -1, 0, 1)
        
        # Split training data for blending
        blend_split = int(0.8 * len(X_train))
        X_blend_train = X_train[:blend_split]
        y_blend_train = y_train_binary[:blend_split]
        X_blend_val = X_train[blend_split:]
        y_blend_val = y_train_binary[blend_split:]
        
        # Train base models on blend training set
        blend_features_train = []
        blend_features_val = []
        
        for name, model in self.base_models.items():
            # Train model
            model.fit(X_blend_train, y_blend_train)
            
            # Get predictions for validation set
            if hasattr(model, 'predict_proba'):
                val_preds = model.predict_proba(X_blend_val)[:, 1]
                train_preds = model.predict_proba(X_blend_train)[:, 1]
            else:
                val_preds = model.predict(X_blend_val)
                train_preds = model.predict(X_blend_train)
            
            blend_features_train.append(train_preds)
            blend_features_val.append(val_preds)
        
        # Stack predictions as features
        X_blend_train_meta = np.column_stack(blend_features_train)
        X_blend_val_meta = np.column_stack(blend_features_val)
        
        # Train meta-learner
        meta_learner = LogisticRegression(
            penalty='l2', C=1.0, 
            max_iter=1000, random_state=42
        )
        meta_learner.fit(X_blend_val_meta, y_blend_val)
        
        # Store for prediction
        self.model = {
            'base_models': self.base_models,
            'meta_learner': meta_learner,
            'type': 'blending'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if isinstance(self.model, dict) and self.model.get('type') == 'blending':
            # Blending ensemble prediction
            blend_features = []
            for name, model in self.model['base_models'].items():
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                blend_features.append(preds)
            
            X_meta = np.column_stack(blend_features)
            predictions = self.model['meta_learner'].predict(X_meta)
        else:
            # Standard sklearn ensemble
            predictions = self.model.predict(X)
        
        # Convert predictions from {0, 1} back to {-1, 1} for trading signals
        return np.where(predictions == 0, -1, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if isinstance(self.model, dict) and self.model.get('type') == 'blending':
            blend_features = []
            for name, model in self.model['base_models'].items():
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                blend_features.append(preds)
            
            X_meta = np.column_stack(blend_features)
            return self.model['meta_learner'].predict_proba(X_meta)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback to predictions
            preds = self.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[preds == -1, 0] = 1
            proba[preds == 1, 1] = 1
            return proba


class AdaptiveEnsembleModel(EnsembleStrategyModel):
    """
    Adaptive ensemble that adjusts weights based on recent performance.
    """
    
    def __init__(self, window_size: int = 50):
        super().__init__()
        self.window_size = window_size
        self.performance_history = {name: [] for name in ['rf', 'xgb', 'svm', 'logreg']}
        
    def update_weights(self, X_recent: np.ndarray, y_recent: np.ndarray):
        """
        Update model weights based on recent performance.
        """
        for name, model in self.base_models.items():
            preds = model.predict(X_recent)
            accuracy = np.mean(preds == y_recent)
            
            # Update performance history
            self.performance_history[name].append(accuracy)
            if len(self.performance_history[name]) > self.window_size:
                self.performance_history[name].pop(0)
            
            # Calculate new weight
            if len(self.performance_history[name]) > 0:
                self.model_weights[name] = np.mean(self.performance_history[name])
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight 
                                for k, v in self.model_weights.items()}


def run_ensemble_strategy(data, initial_train_split_ratio=0.5, lags=5,
                         ensemble_type='voting', base_models=None,
                         use_probabilities=True, auto_weight=True,
                         n_jobs=-1, log_prefix="Ensemble_model", **kwargs):
    """
    Run optimized ensemble strategy.
    
    Args:
        ensemble_type: 'voting', 'stacking', or 'blending'
        base_models: List of model names to include
        use_probabilities: Use soft voting
        auto_weight: Automatically calculate model weights
        n_jobs: Number of parallel jobs
    """
    model = EnsembleStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        ensemble_type=ensemble_type,
        base_models=base_models,
        use_probabilities=use_probabilities,
        auto_weight=auto_weight,
        n_jobs=n_jobs,
        **kwargs
    )


def run_adaptive_ensemble_strategy(data, initial_train_split_ratio=0.5, lags=5,
                                  window_size=50, log_prefix="AdaptiveEnsemble_model",
                                  **kwargs):
    """
    Run adaptive ensemble that adjusts weights over time.
    """
    model = AdaptiveEnsembleModel(window_size=window_size)
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        ensemble_type='voting',
        auto_weight=True,
        **kwargs
    )