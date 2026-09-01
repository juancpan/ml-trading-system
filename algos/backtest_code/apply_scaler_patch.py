#!/usr/bin/env python3
"""
Simple patch to apply to any backtest script to automatically save scalers.
Just import this at the beginning of your backtest script:

    import apply_scaler_patch

That's it! All StandardScalers will be automatically saved.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from algos.backtest_code.scaler_integration import (
    scaler_saver, 
    integrate_scaler_saving,
    patch_standardscaler
)

# Patch is already applied when scaler_integration is imported
print("[Scaler Auto-Save] Patch applied - scalers will be saved automatically")

# Also patch the save_model function if it exists
try:
    import algos.common.persistence as persistence
    
    original_save_model = persistence.save_model
    
    @integrate_scaler_saving
    def patched_save_model(model_obj, model_name: str, ticker: str, symbol: str, 
                          start: str, end: str, interval: str, timestamp: str):
        """Patched save_model that also saves scalers."""
        return original_save_model(
            model_obj, model_name, ticker, symbol, start, end, interval, timestamp
        )
    
    # Replace the original
    persistence.save_model = patched_save_model
    
    print("[Scaler Auto-Save] save_model function patched - scalers will be saved with models")
    
except ImportError:
    print("[Scaler Auto-Save] Could not patch save_model - manual saving may be required")


def ensure_scaler_saved(model_name: str, ticker: str, timestamp: str):
    """
    Manually ensure the current scaler is saved.
    Call this after training if automatic saving doesn't work.
    
    Example:
        model = train_lstm_model(data)
        ensure_scaler_saved('lstm', 'NVDA', timestamp)
    """
    if scaler_saver.current_scaler is not None:
        path = scaler_saver.save_captured_scaler(model_name, ticker, timestamp)
        if path:
            print(f"[Scaler Auto-Save] Manually saved scaler to {path}")
    else:
        print("[Scaler Auto-Save] No scaler to save")


# Export for convenience
__all__ = ['ensure_scaler_saved', 'scaler_saver']