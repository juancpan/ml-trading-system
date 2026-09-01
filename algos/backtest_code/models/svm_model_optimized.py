"""
Optimized SVM trading strategy model using base class.
"""

import numpy as np
from sklearn.svm import SVC
from algos.common.base_model import BaseStrategyModel
from algos.common.utils import RedirectStdoutToFile


class SVMStrategyModel(BaseStrategyModel):
    """Support Vector Machine trading strategy implementation."""
    
    def __init__(self):
        super().__init__(model_name="SVM")
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, log_prefix: str,
                   C: float = 1.0, kernel: str = 'rbf', random_state: int = 42, **kwargs) -> None:
        """Train the SVM model."""
        self.model = SVC(C=C, kernel=kernel, random_state=random_state, probability=False)
        
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print('=' * 50)
            print(f"\nModel Training ({log_prefix}):\n")
            try:
                self.model.fit(X_train, y_train)
                print(f"Model fitted successfully: {self.model}")
                
                if kernel == 'linear':
                    print(f"Model Coefficients: {self.model.coef_}")
                    print(f"Model Intercept: {self.model.intercept_}")
                else:
                    print(f"Support vectors per class: {self.model.n_support_}")
                    
            except Exception as e:
                print(f"Error during {log_prefix} training: {e}")
                raise
            print('=' * 50 + '\n')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)


def run_svm_strategy(data, initial_train_split_ratio=0.5, lags=5, C=1.0,
                    kernel='rbf', random_state=42, log_prefix="SVM_model",
                    embargo_pct=0.02, **kwargs):
    """
    Wrapper function for backward compatibility with embargo.
    """
    model = SVMStrategyModel()
    return model.run_strategy(
        data=data,
        initial_train_split_ratio=initial_train_split_ratio,
        lags=lags,
        log_prefix=log_prefix,
        embargo_pct=embargo_pct,
        C=C,
        kernel=kernel,
        random_state=random_state,
        **kwargs
    )