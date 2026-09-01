"""
Unified model loader for both original and optimized models.
Provides a consistent interface for loading and managing trading models.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import json


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    module_path: str
    function_name: str
    is_optimized: bool
    description: str
    parameters: Dict[str, Any]


class UnifiedModelLoader:
    """
    Centralized model loader that manages both original and optimized models.
    """
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._register_all_models()
    
    def _register_all_models(self):
        """Register all available models with metadata."""
        
        # Original models
        original_models = {
            'dqn': ModelInfo(
                name='dqn',
                module_path='algos.backtest_code.models.dqn_model',
                function_name='run_dqn_strategy',
                is_optimized=False,
                description='Deep Q-Network reinforcement learning model',
                parameters={'epochs': 50, 'batch_size': 32, 'hidden_units': 64}
            ),
            'cnn': ModelInfo(
                name='cnn',
                module_path='algos.backtest_code.models.cnn_model',
                function_name='run_cnn_strategy',
                is_optimized=False,
                description='Convolutional Neural Network for pattern recognition',
                parameters={'filters': 32, 'kernel_size': 3}
            ),
            'arima': ModelInfo(
                name='arima',
                module_path='algos.backtest_code.models.arima_model',
                function_name='run_arima_strategy',
                is_optimized=False,
                description='AutoRegressive Integrated Moving Average time series model',
                parameters={'p': 1, 'd': 1, 'q': 1}
            ),
            'lstm': ModelInfo(
                name='lstm',
                module_path='algos.backtest_code.models.lstm_model',
                function_name='run_lstm_strategy',
                is_optimized=False,
                description='Long Short-Term Memory neural network',
                parameters={'units': 50, 'sequence_length': 20}
            ),
            'rf': ModelInfo(
                name='rf',
                module_path='algos.backtest_code.models.random_forest_model',
                function_name='run_random_forest_strategy',
                is_optimized=False,
                description='Random Forest ensemble classifier',
                parameters={'n_estimators': 100, 'max_depth': 10}
            ),
            'svm': ModelInfo(
                name='svm',
                module_path='algos.backtest_code.models.svm_model',
                function_name='run_svm_strategy',
                is_optimized=False,
                description='Support Vector Machine classifier',
                parameters={'kernel': 'rbf', 'C': 1.0}
            ),
            'xgb': ModelInfo(
                name='xgb',
                module_path='algos.backtest_code.models.gbm_model',
                function_name='run_xgboost_strategy',
                is_optimized=False,
                description='XGBoost gradient boosting model',
                parameters={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
            ),
            'li_reg': ModelInfo(
                name='li_reg',
                module_path='algos.backtest_code.models.linear_regression_model',
                function_name='run_linear_regression_strategy',
                is_optimized=False,
                description='Linear Regression model',
                parameters={'fit_intercept': True}
            ),
            'log_reg': ModelInfo(
                name='log_reg',
                module_path='algos.backtest_code.models.logistic_regression_model',
                function_name='run_logistic_regression_strategy',
                is_optimized=False,
                description='Logistic Regression classifier',
                parameters={'penalty': 'l2', 'C': 1.0}
            ),
        }
        
        # Optimized models
        optimized_models = {
            'svm_optimized': ModelInfo(
                name='svm_optimized',
                module_path='algos.backtest_code.models.svm_model_optimized',
                function_name='run_svm_strategy',
                is_optimized=True,
                description='Optimized SVM with base class architecture',
                parameters={'kernel': 'rbf', 'C': 1.0, 'random_state': 42}
            ),
            'rf_optimized': ModelInfo(
                name='rf_optimized',
                module_path='algos.backtest_code.models.random_forest_optimized',
                function_name='run_random_forest_strategy',
                is_optimized=True,
                description='Optimized Random Forest with parallel processing',
                parameters={'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1, 'oob_score': True}
            ),
            'lstm_optimized': ModelInfo(
                name='lstm_optimized',
                module_path='algos.backtest_code.models.lstm_optimized',
                function_name='run_lstm_strategy',
                is_optimized=True,
                description='Optimized LSTM with GPU support and batch normalization',
                parameters={'lstm_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'epochs': 50}
            ),
            'xgb_optimized': ModelInfo(
                name='xgb_optimized',
                module_path='algos.backtest_code.models.xgboost_optimized',
                function_name='run_xgboost_strategy',
                is_optimized=True,
                description='Optimized XGBoost with GPU support and auto-tuning',
                parameters={'n_estimators': 100, 'max_depth': 6, 'use_gpu': False, 'auto_tune': False, 'n_jobs': -1}
            ),
            'linear_optimized': ModelInfo(
                name='linear_optimized',
                module_path='algos.backtest_code.models.linear_models_optimized',
                function_name='run_linear_regression_strategy',
                is_optimized=True,
                description='Optimized Linear Regression with regularization options',
                parameters={'regularization': 'ridge', 'alpha': 1.0, 'polynomial_degree': 1}
            ),
            'logistic_optimized': ModelInfo(
                name='logistic_optimized',
                module_path='algos.backtest_code.models.linear_models_optimized',
                function_name='run_logistic_regression_strategy',
                is_optimized=True,
                description='Optimized Logistic Regression with advanced features',
                parameters={'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'class_weight': 'balanced', 'n_jobs': -1}
            ),
            'sgd_optimized': ModelInfo(
                name='sgd_optimized',
                module_path='algos.backtest_code.models.linear_models_optimized',
                function_name='run_sgd_linear_strategy',
                is_optimized=True,
                description='SGD classifier for large-scale learning',
                parameters={'loss': 'log', 'penalty': 'l2', 'alpha': 0.0001, 'early_stopping': True}
            ),
            'ensemble_optimized': ModelInfo(
                name='ensemble_optimized',
                module_path='algos.backtest_code.models.ensemble_optimized',
                function_name='run_ensemble_strategy',
                is_optimized=True,
                description='Optimized ensemble with voting/stacking/blending',
                parameters={'ensemble_type': 'voting', 'base_models': None, 'auto_weight': True, 'n_jobs': -1}
            ),
            'ensemble_adaptive': ModelInfo(
                name='ensemble_adaptive',
                module_path='algos.backtest_code.models.ensemble_optimized',
                function_name='run_adaptive_ensemble_strategy',
                is_optimized=True,
                description='Adaptive ensemble that adjusts weights over time',
                parameters={'window_size': 50}
            ),
        }
        
        # Merge all models
        self._models.update(original_models)
        self._models.update(optimized_models)
    
    def get_model(self, model_name: str) -> Callable:
        """
        Get a model function by name.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            The model's run function
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.list_models()}")
        
        # Check if already loaded
        if model_name in self._loaded_functions:
            return self._loaded_functions[model_name]
        
        # Load the model
        model_info = self._models[model_name]
        try:
            module = importlib.import_module(model_info.module_path)
            func = getattr(module, model_info.function_name)
            self._loaded_functions[model_name] = func
            return func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load model '{model_name}': {e}")
    
    def list_models(self, optimized_only: bool = False, original_only: bool = False) -> List[str]:
        """
        List available models.
        
        Args:
            optimized_only: Only list optimized models
            original_only: Only list original models
            
        Returns:
            List of model names
        """
        models = []
        for name, info in self._models.items():
            if optimized_only and not info.is_optimized:
                continue
            if original_only and info.is_optimized:
                continue
            models.append(name)
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object with model details
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        return self._models[model_name]
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of default parameters
        """
        model_info = self.get_model_info(model_name)
        return model_info.parameters.copy()
    
    def compare_models(self, original_name: str, optimized_name: str) -> Dict[str, Any]:
        """
        Compare original and optimized versions of a model.
        
        Args:
            original_name: Name of original model
            optimized_name: Name of optimized model
            
        Returns:
            Comparison dictionary
        """
        original_info = self.get_model_info(original_name)
        optimized_info = self.get_model_info(optimized_name)
        
        comparison = {
            'original': {
                'name': original_info.name,
                'description': original_info.description,
                'parameters': original_info.parameters
            },
            'optimized': {
                'name': optimized_info.name,
                'description': optimized_info.description,
                'parameters': optimized_info.parameters
            },
            'new_features': [],
            'improvements': []
        }
        
        # Identify new parameters (features)
        original_params = set(original_info.parameters.keys())
        optimized_params = set(optimized_info.parameters.keys())
        new_params = optimized_params - original_params
        
        if new_params:
            comparison['new_features'] = list(new_params)
        
        # Add known improvements
        if 'n_jobs' in optimized_params:
            comparison['improvements'].append('Parallel processing support')
        if 'use_gpu' in optimized_params:
            comparison['improvements'].append('GPU acceleration')
        if 'auto_tune' in optimized_params:
            comparison['improvements'].append('Automatic hyperparameter tuning')
        if optimized_info.is_optimized:
            comparison['improvements'].append('Uses optimized base class architecture')
        
        return comparison
    
    def print_model_summary(self):
        """Print a summary of all available models."""
        original_models = self.list_models(original_only=True)
        optimized_models = self.list_models(optimized_only=True)
        
        print("\n" + "="*60)
        print("AVAILABLE TRADING MODELS")
        print("="*60)
        
        print(f"\nOriginal Models ({len(original_models)}):")
        print("-" * 30)
        for model in original_models:
            info = self.get_model_info(model)
            print(f"  • {model:20s} - {info.description}")
        
        print(f"\nOptimized Models ({len(optimized_models)}):")
        print("-" * 30)
        for model in optimized_models:
            info = self.get_model_info(model)
            print(f"  • {model:20s} - {info.description}")
        
        print(f"\nTotal Models: {len(self._models)}")
        print("="*60)
    
    def export_model_catalog(self, filepath: str):
        """
        Export model catalog to JSON file.
        
        Args:
            filepath: Path to save the catalog
        """
        catalog = {}
        for name, info in self._models.items():
            catalog[name] = {
                'module_path': info.module_path,
                'function_name': info.function_name,
                'is_optimized': info.is_optimized,
                'description': info.description,
                'parameters': info.parameters
            }
        
        with open(filepath, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"Model catalog exported to {filepath}")


# Singleton instance
_model_loader = None


def get_model_loader() -> UnifiedModelLoader:
    """Get or create the singleton model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = UnifiedModelLoader()
    return _model_loader


# Convenience functions
def load_model(model_name: str) -> Callable:
    """Load a model by name."""
    loader = get_model_loader()
    return loader.get_model(model_name)


def list_available_models(optimized_only: bool = False, original_only: bool = False) -> List[str]:
    """List available models."""
    loader = get_model_loader()
    return loader.list_models(optimized_only, original_only)


def get_model_parameters(model_name: str) -> Dict[str, Any]:
    """Get default parameters for a model."""
    loader = get_model_loader()
    return loader.get_model_parameters(model_name)


if __name__ == "__main__":
    # Demo the model loader
    loader = UnifiedModelLoader()
    loader.print_model_summary()
    
    # Example: Compare original vs optimized SVM
    print("\n" + "="*60)
    print("COMPARISON: SVM vs SVM_OPTIMIZED")
    print("="*60)
    comparison = loader.compare_models('svm', 'svm_optimized')
    print(json.dumps(comparison, indent=2))