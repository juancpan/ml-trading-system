"""
Unified seed management for reproducible model training.
Ensures consistent random number generation across all models and systems.
"""

import os
import sys
import random
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
from datetime import datetime


class SeedManager:
    """
    Manages random seeds for reproducibility across all ML models.
    
    Features:
    - Consistent seed setting across numpy, random, tensorflow, pytorch
    - Seed tracking and logging
    - Model-specific seed generation
    - Deterministic mode for full reproducibility
    """
    
    # Default global seed
    DEFAULT_SEED = 1000
    
    # Model-specific seeds (for backward compatibility)
    MODEL_SEEDS = {
        'lstm': 1000,
        'dqn': 1000,
        'dnn': 1000,
        'cnn': 1000,
        'tcn': 1000,
        'sklearn_dnn': 1000,
        'svm': 1000,
        'rf': 1000,
        'xgboost': 1000,
        'arima': 1000,
        'default': 1000
    }
    
    def __init__(self, base_seed: Optional[int] = None):
        """
        Initialize seed manager.
        
        Args:
            base_seed: Base seed to use. If None, uses DEFAULT_SEED.
        """
        self.base_seed = base_seed or self.DEFAULT_SEED
        self.seeds_used = {}
        self.initialized_libraries = set()
        
    def get_seed_for_model(self, model_name: str, ticker: Optional[str] = None) -> int:
        """
        Get a consistent seed for a specific model and ticker combination.
        
        Args:
            model_name: Name of the model
            ticker: Optional ticker symbol
            
        Returns:
            Integer seed value
        """
        # Use predefined seed if available
        if model_name in self.MODEL_SEEDS:
            seed = self.MODEL_SEEDS[model_name]
        else:
            seed = self.MODEL_SEEDS['default']
        
        # Optionally modify seed based on ticker for diversity
        if ticker:
            # Create deterministic variation based on ticker
            ticker_hash = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
            seed = (seed + ticker_hash) % (2**31 - 1)  # Keep within int32 range
        
        return seed
    
    def set_global_seed(self, seed: Optional[int] = None, model_name: Optional[str] = None,
                       ticker: Optional[str] = None) -> int:
        """
        Set seeds for all random number generators.
        
        Args:
            seed: Specific seed to use
            model_name: Model name for model-specific seeding
            ticker: Ticker for ticker-specific variation
            
        Returns:
            The seed that was set
        """
        if seed is None:
            if model_name:
                seed = self.get_seed_for_model(model_name, ticker)
            else:
                seed = self.base_seed
        
        # Set Python's random seed
        random.seed(seed)
        self.initialized_libraries.add('random')
        
        # Set NumPy's random seed
        np.random.seed(seed)
        self.initialized_libraries.add('numpy')
        
        # Set environment variable for hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set TensorFlow environment variables (before import)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        # Suppress TensorFlow warnings about version compatibility
        if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show fatal errors
        
        # Only import TensorFlow if it's already loaded or if explicitly needed
        if 'tensorflow' in sys.modules:
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
                self.initialized_libraries.add('tensorflow')
            except ImportError:
                pass
        
        # Try to set PyTorch seed
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Deterministic operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            self.initialized_libraries.add('pytorch')
        except ImportError:
            pass
        
        # Note: sklearn models should use random_state parameter directly
        # There's no global sklearn seed setting
        self.initialized_libraries.add('sklearn_note')
        
        # Track this seed
        key = f"{model_name or 'global'}_{ticker or 'all'}"
        self.seeds_used[key] = {
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'libraries': list(self.initialized_libraries)
        }
        
        return seed
    
    def save_seed_config(self, filepath: Path, model_name: str, ticker: str,
                        additional_info: Optional[Dict] = None):
        """
        Save seed configuration for reproducibility.
        
        Args:
            filepath: Where to save the config
            model_name: Model name
            ticker: Ticker symbol
            additional_info: Additional metadata
        """
        config = {
            'model_name': model_name,
            'ticker': ticker,
            'seed': self.get_seed_for_model(model_name, ticker),
            'base_seed': self.base_seed,
            'timestamp': datetime.now().isoformat(),
            'seeds_used': self.seeds_used,
            'initialized_libraries': list(self.initialized_libraries),
            'additional_info': additional_info or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_seed_config(filepath: Path) -> Dict:
        """
        Load seed configuration from file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            Seed configuration dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def ensure_reproducibility():
        """
        Set all possible flags for maximum reproducibility.
        This may impact performance but ensures consistent results.
        """
        # Python hash seed
        os.environ['PYTHONHASHSEED'] = '0'
        
        # TensorFlow settings
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # CUDA settings
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for full reproducibility
        
        # NumPy settings
        np.seterr(all='warn')  # Warn on numerical issues
        
        print("Reproducibility mode enabled. Performance may be reduced.")
    
    def ensure_tensorflow_seed(self, seed: Optional[int] = None) -> None:
        """
        Explicitly set TensorFlow seed when needed.
        Call this before using TensorFlow models.
        
        Args:
            seed: Seed to use, or None to use current base seed
        """
        if seed is None:
            seed = self.base_seed
            
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            self.initialized_libraries.add('tensorflow')
        except ImportError:
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of seed manager.
        
        Returns:
            Status dictionary
        """
        return {
            'base_seed': self.base_seed,
            'seeds_used': self.seeds_used,
            'initialized_libraries': list(self.initialized_libraries),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
global_seed_manager = SeedManager()


def set_seed(seed: Optional[int] = None, model_name: Optional[str] = None,
            ticker: Optional[str] = None) -> int:
    """
    Convenience function to set global seeds.
    
    Args:
        seed: Specific seed value
        model_name: Model name for model-specific seeding
        ticker: Ticker for additional variation
        
    Returns:
        The seed that was set
    """
    return global_seed_manager.set_global_seed(seed, model_name, ticker)


def get_model_seed(model_name: str, ticker: Optional[str] = None) -> int:
    """
    Get the appropriate seed for a model.
    
    Args:
        model_name: Model name
        ticker: Optional ticker
        
    Returns:
        Seed value
    """
    return global_seed_manager.get_seed_for_model(model_name, ticker)


def save_seed_info(filepath: Path, model_name: str, ticker: str, **kwargs):
    """
    Save seed information for reproducibility.
    
    Args:
        filepath: Where to save
        model_name: Model name
        ticker: Ticker symbol
        **kwargs: Additional metadata
    """
    global_seed_manager.save_seed_config(filepath, model_name, ticker, kwargs)


# Decorator for functions that need reproducibility
def with_seed(seed: Optional[int] = None, model_name: Optional[str] = None):
    """
    Decorator to ensure function runs with consistent seed.
    
    Usage:
        @with_seed(42)
        def train_model():
            ...
            
        @with_seed(model_name='lstm')
        def train_lstm():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set seed before function
            actual_seed = set_seed(seed, model_name)
            print(f"[Seed Manager] Set seed to {actual_seed} for {func.__name__}")
            
            # Run function
            result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test seed management
    print("Testing Seed Manager")
    print("=" * 50)
    
    # Test default seeding
    seed = set_seed()
    print(f"Default seed: {seed}")
    print(f"Random sample: {np.random.randn(3)}")
    
    # Reset and verify consistency
    seed = set_seed()
    print(f"Reset seed: {seed}")
    print(f"Random sample (should be same): {np.random.randn(3)}")
    
    # Test model-specific seeds
    print("\nModel-specific seeds:")
    for model in ['lstm', 'dqn', 'svm']:
        seed = get_model_seed(model, 'NVDA')
        print(f"  {model} + NVDA: {seed}")
    
    print("\nStatus:")
    print(json.dumps(global_seed_manager.get_status(), indent=2))