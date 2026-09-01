"""
Optimized Linear and Logistic Regression models with regularization and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, 
    Ridge, Lasso, ElasticNet,
    RidgeClassifier, SGDClassifier
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile
from typing import Optional, Dict, Any


class LinearRegressionStrategyModel(BaseStrategyModel):
    """
    Optimized Linear Regression with regularization options.
    """
    
    def __init__(self):
        super().__init__(model_name="LinearRegression")
        self.coefficients = None
        self.intercept = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, log_prefix: str,
                   regularization: str = 'ridge', alpha: float = 1.0,
                   polynomial_degree: int = 1, fit_intercept: bool = True,
                   max_iter: int = 1000, **kwargs) -> None:
        """
        Train linear regression with various regularization options.
        
        Args:
            regularization: Type of regularization ('none', 'ridge', 'lasso', 'elastic')
            alpha: Regularization strength
            polynomial_degree: Degree of polynomial features (1 = no polynomial)
            fit_intercept: Whether to fit intercept
        """
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print('=' * 50)
            print(f"\nTraining Optimized Linear Regression ({log_prefix}):\n")
            print(f"Regularization: {regularization}, Alpha: {alpha}")
            
            # Create pipeline with optional polynomial features
            steps = []
            
            if polynomial_degree > 1:
                poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
                steps.append(('polynomial', poly))
                print(f"Using polynomial features of degree {polynomial_degree}")
            
            # Select regression model based on regularization
            if regularization == 'none':
                regressor = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
            elif regularization == 'ridge':
                regressor = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
            elif regularization == 'lasso':
                regressor = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
            elif regularization == 'elastic':
                regressor = ElasticNet(alpha=alpha, fit_intercept=fit_intercept, 
                                      max_iter=max_iter, l1_ratio=0.5)
            else:
                raise ValueError(f"Unknown regularization: {regularization}")
            
            steps.append(('regressor', regressor))
            
            # Create pipeline
            self.model = Pipeline(steps)
            
            try:
                # Convert regression to classification
                # Use regression predictions to determine direction
                self.model.fit(X_train, y_train)
                
                # Store coefficients if available
                if hasattr(self.model.named_steps['regressor'], 'coef_'):
                    self.coefficients = self.model.named_steps['regressor'].coef_
                    self.intercept = self.model.named_steps['regressor'].intercept_
                    
                    print(f"\nModel Coefficients:")
                    for i, coef in enumerate(self.coefficients[:min(5, len(self.coefficients))]):
                        print(f"  Feature {i+1}: {coef:.6f}")
                    print(f"Intercept: {self.intercept:.6f}")
                
                print(f"Model fitted successfully")
                
            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print('=' * 50 + '\n')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and convert to trading signals."""
        # Get regression predictions
        predictions = self.model.predict(X)
        # Convert to trading signals based on predicted direction
        return np.where(predictions > 0, 1, -1)


class LogisticRegressionStrategyModel(BaseStrategyModel):
    """
    Optimized Logistic Regression with advanced features.
    """
    
    def __init__(self):
        super().__init__(model_name="LogisticRegression")
        self.coefficients = None
        self.intercept = None
        self.classes = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, log_prefix: str,
                   penalty: str = 'l2', C: float = 1.0,
                   solver: str = 'lbfgs', class_weight: str = 'balanced',
                   polynomial_degree: int = 1, max_iter: int = 1000,
                   n_jobs: int = -1, **kwargs) -> None:
        """
        Train logistic regression with optimization features.
        
        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength
            solver: Optimization algorithm
            class_weight: Class weight strategy
            polynomial_degree: Degree of polynomial features
            n_jobs: Number of parallel jobs
        """
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print('=' * 50)
            print(f"\nTraining Optimized Logistic Regression ({log_prefix}):\n")
            print(f"Penalty: {penalty}, C: {C}, Solver: {solver}")
            
            # Create pipeline
            steps = []
            
            if polynomial_degree > 1:
                poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
                steps.append(('polynomial', poly))
                print(f"Using polynomial features of degree {polynomial_degree}")
            
            # Adjust solver based on penalty
            if penalty == 'l1':
                solver = 'liblinear' if solver == 'lbfgs' else solver
            elif penalty == 'elasticnet':
                solver = 'saga'
                
            # Create logistic regression model
            log_reg = LogisticRegression(
                penalty=penalty if penalty != 'none' else None,
                C=C,
                solver=solver,
                class_weight=class_weight,
                max_iter=max_iter,
                n_jobs=n_jobs,
                random_state=42,
                l1_ratio=0.5 if penalty == 'elasticnet' else None
            )
            
            steps.append(('classifier', log_reg))
            self.model = Pipeline(steps)
            
            try:
                self.model.fit(X_train, y_train)
                
                # Store model parameters
                classifier = self.model.named_steps['classifier']
                self.coefficients = classifier.coef_
                self.intercept = classifier.intercept_
                self.classes = classifier.classes_
                
                print(f"Model fitted successfully")
                print(f"Classes: {self.classes}")
                
                # Print coefficients for first class
                if self.coefficients.ndim > 1:
                    coefs = self.coefficients[0]
                else:
                    coefs = self.coefficients
                    
                print(f"\nModel Coefficients (first {min(5, len(coefs))} features):")
                for i, coef in enumerate(coefs[:min(5, len(coefs))]):
                    print(f"  Feature {i+1}: {coef:.6f}")
                print(f"Intercept: {self.intercept[0] if self.intercept.ndim > 0 else self.intercept:.6f}")
                
                # Calculate and print convergence info
                if hasattr(classifier, 'n_iter_'):
                    print(f"Converged in {classifier.n_iter_[0]} iterations")
                
            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print('=' * 50 + '\n')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)


class SGDLinearStrategyModel(BaseStrategyModel):
    """
    Stochastic Gradient Descent classifier for large-scale learning.
    """
    
    def __init__(self):
        super().__init__(model_name="SGDLinear")
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, log_prefix: str,
                   loss: str = 'log', penalty: str = 'l2', alpha: float = 0.0001,
                   learning_rate: str = 'optimal', eta0: float = 0.01,
                   max_iter: int = 1000, early_stopping: bool = True,
                   n_jobs: int = -1, **kwargs) -> None:
        """
        Train SGD classifier for efficient large-scale learning.
        
        Args:
            loss: Loss function ('hinge', 'log', 'modified_huber', etc.)
            penalty: Regularization term
            alpha: Regularization strength
            learning_rate: Learning rate schedule
            early_stopping: Whether to use early stopping
        """
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_jobs=n_jobs,
            random_state=42,
            verbose=0
        )
        
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print('=' * 50)
            print(f"\nTraining SGD Linear Model ({log_prefix}):\n")
            print(f"Loss: {loss}, Penalty: {penalty}, Alpha: {alpha}")
            
            try:
                self.model.fit(X_train, y_train)
                print(f"Model fitted successfully")
                print(f"Number of iterations: {self.model.n_iter_}")
                
                if hasattr(self.model, 'coef_'):
                    print(f"\nCoefficients shape: {self.model.coef_.shape}")
                    
            except Exception as e:
                print(f"Error during training: {e}")
                raise
            print('=' * 50 + '\n')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)


# Wrapper functions for backward compatibility
def run_linear_regression_strategy(data, initial_train_split_ratio=0.5, lags=5,
                                  regularization='ridge', alpha=1.0,
                                  polynomial_degree=1, log_prefix="LinearReg_model",
                                  embargo_pct=0.02, **kwargs):
    """Optimized linear regression strategy with embargo."""
    model = LinearRegressionStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        regularization=regularization,
        alpha=alpha,
        polynomial_degree=polynomial_degree,
        **kwargs
    )


def run_logistic_regression_strategy(data, initial_train_split_ratio=0.5, lags=5,
                                    penalty='l2', C=1.0, solver='lbfgs',
                                    class_weight='balanced', polynomial_degree=1,
                                    log_prefix="LogReg_model", embargo_pct=0.02, **kwargs):
    """Optimized logistic regression strategy with embargo."""
    model = LogisticRegressionStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        polynomial_degree=polynomial_degree,
        **kwargs
    )


def run_sgd_linear_strategy(data, initial_train_split_ratio=0.5, lags=5,
                           loss='log', penalty='l2', alpha=0.0001,
                           log_prefix="SGD_model", embargo_pct=0.02, **kwargs):
    """SGD-based linear model strategy with embargo."""
    model = SGDLinearStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        embargo_pct=embargo_pct,
        lags=lags,
        log_prefix=log_prefix,
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        **kwargs
    )