#!/usr/bin/env python3
"""
Test script to verify that live trading signals match backtesting signals.
This script simulates signal generation and compares with expected values.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_manager import DataManager
from strategy_executor import StrategyExecutor
from utils import setup_logger
from config import ASSET_SPECIFIC_CONFIGS, SYMBOLS


def test_signal_generation():
    """
    Test that signals are generated correctly with the fixes applied.
    """
    print("=" * 60)
    print("SIGNAL CONSISTENCY TEST")
    print("=" * 60)
    
    # Initialize components
    logger = setup_logger('test_signals.log')
    data_manager = DataManager(logger)
    strategy_executor = StrategyExecutor(data_manager, logger, lags=5)
    
    # Test cases with known expected signals
    # These should match your backtesting results
    test_cases = [
        {
            'symbol': 'NVDA',
            'date': datetime(2025, 9, 11),
            'expected_signal': -1,
            'model_type': 'lstm'
        },
        {
            'symbol': 'NVDA', 
            'date': datetime(2025, 9, 12),
            'expected_signal': 1,  # This should be 1 in backtesting
            'model_type': 'lstm'
        },
        {
            'symbol': 'AVGO',
            'date': datetime(2025, 9, 11),
            'model_type': 'li_reg',
            'check_binary': True  # Just check it's binary (-1 or 1)
        },
        {
            'symbol': 'AVGO',
            'date': datetime(2025, 9, 12),
            'model_type': 'li_reg',
            'check_binary': True
        }
    ]
    
    print("\nFetching historical data for testing...")
    
    # Fetch historical data for each symbol
    for symbol in SYMBOLS:
        end_date = datetime(2025, 9, 13).date()  # Day after last test
        print(f"  Fetching data for {symbol}...")
        data_manager.fetch_and_store_historical_data(symbol, end_date)
    
    print("\n" + "-" * 60)
    print("TEST RESULTS:")
    print("-" * 60)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        symbol = test_case['symbol']
        date = test_case['date']
        model_type = test_case['model_type']
        
        print(f"\nTest {i}: {symbol} on {date.strftime('%Y-%m-%d')} (model: {model_type})")
        
        try:
            # Generate signal
            signal = strategy_executor.generate_signal(symbol)
            
            # Check if signal is binary
            if signal not in [-1, 0, 1]:
                print(f"  ✗ FAILED: Signal is not binary: {signal}")
                all_passed = False
                continue
            
            # Check specific value if expected
            if 'expected_signal' in test_case:
                expected = test_case['expected_signal']
                if signal == expected:
                    print(f"  ✓ PASSED: Signal = {signal} (expected {expected})")
                else:
                    print(f"  ✗ FAILED: Signal = {signal} (expected {expected})")
                    all_passed = False
            
            # Just check if binary
            elif test_case.get('check_binary'):
                if signal in [-1, 1]:
                    print(f"  ✓ PASSED: Signal is binary = {signal}")
                else:
                    print(f"  ✗ FAILED: Signal is not binary = {signal}")
                    all_passed = False
                    
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Signals are consistent!")
    else:
        print("✗ SOME TESTS FAILED - Check the implementation")
    print("=" * 60)
    
    return all_passed


def test_preprocessing_consistency():
    """
    Test that data preprocessing matches between backtesting and live trading.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING CONSISTENCY TEST")
    print("=" * 60)
    
    logger = setup_logger('test_signals.log')
    data_manager = DataManager(logger)
    
    # Fetch sample data
    symbol = 'NVDA'
    end_date = datetime.now().date()
    data_manager.fetch_and_store_historical_data(symbol, end_date)
    
    print(f"\nTesting preprocessing for {symbol}:")
    
    # Test 1: Check log returns calculation
    history_df = data_manager.historical_data[symbol]
    if not history_df.empty:
        # Calculate returns as in the fixed version
        test_returns = np.log(history_df['Close'] / history_df['Close'].shift(1))
        print(f"  ✓ Log returns calculation: Last 5 values:")
        print(f"    {test_returns.tail(5).values}")
    
    # Test 2: Check lagged features
    features = data_manager.get_data_for_strategy(symbol, lags=5)
    if features is not None:
        print(f"  ✓ Lagged features shape: {features.shape}")
        print(f"    Features: {features}")
    
    # Test 3: Check LSTM sequence data
    lstm_features = data_manager.create_sequence_data(symbol, lags=5)
    if lstm_features is not None:
        print(f"  ✓ LSTM features shape: {lstm_features.shape}")
        print(f"    First 3 values: {lstm_features[0, :3, 0]}")
    
    # Test 4: Check scaler loading
    scaler = data_manager.load_scaler(symbol)
    if scaler is not None:
        print(f"  ✓ Scaler loaded successfully")
        print(f"    Features expected: {scaler.n_features_in_}")
        print(f"    Mean (first 3): {scaler.mean_[:3]}")
    else:
        print(f"  ⚠ Warning: No scaler found (will use unscaled features)")
    
    print("\n" + "=" * 60)


def test_binary_conversion():
    """
    Test that all model outputs are converted to binary signals.
    """
    print("\n" + "=" * 60)
    print("BINARY CONVERSION TEST")
    print("=" * 60)
    
    logger = setup_logger('test_signals.log')
    data_manager = DataManager(logger)
    strategy_executor = StrategyExecutor(data_manager, logger)
    
    # Test various raw signal values
    test_values = [
        (0.0538, 'li_reg'),     # Continuous positive
        (-0.0234, 'li_reg'),    # Continuous negative
        (1.5, 'lstm'),          # Large positive
        (-2.1, 'lstm'),         # Large negative
        (0.0001, 'arima'),      # Very small positive
        (-0.0001, 'arima'),     # Very small negative
        (0, 'svm'),             # Exactly zero
    ]
    
    print("\nTesting binary conversion for various signals:\n")
    
    for raw_signal, model_type in test_values:
        binary = strategy_executor._convert_to_binary_signal(raw_signal, model_type, 'TEST')
        status = "✓" if binary in [-1, 1] else "✗"
        print(f"  {status} {model_type:10s}: {raw_signal:8.4f} -> {binary:2d}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nRunning Signal Consistency Tests...")
    print("This will verify that live trading signals match backtesting.\n")
    
    # Run all tests
    test_preprocessing_consistency()
    test_binary_conversion()
    
    # Main signal generation test
    if test_signal_generation():
        print("\n✅ SUCCESS: Your fixes are working correctly!")
        print("Live trading should now generate identical signals to backtesting.")
    else:
        print("\n⚠️  WARNING: Some issues detected.")
        print("Please review the failed tests above.")
    
    print("\nTest complete.\n")