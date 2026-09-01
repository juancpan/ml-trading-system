#!/usr/bin/env python3
"""
Test LSTM model with exact features from backtesting CSV for Sep 12.
"""

import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

def test_with_exact_features():
    """Test LSTM with the exact features from backtesting."""
    
    print("=" * 70)
    print("TESTING LSTM WITH EXACT BACKTESTING FEATURES")
    print("=" * 70)
    
    # Exact features from backtesting CSV for Sep 12
    # These are the lagged returns as shown in the CSV
    features_raw = np.array([
        -0.000846,  # lag_1 (Sep 11 return)
        0.038475,   # lag_2 (Sep 10 return)  
        0.014556,   # lag_3 (Sep 9 return)
        0.007724,   # lag_4 (Sep 8 return)
        -0.027030   # lag_5 (Sep 5 return)
    ])
    
    print("\n1. RAW FEATURES (from backtesting CSV):")
    print("-" * 40)
    for i, val in enumerate(features_raw, 1):
        print(f"  lag_{i}: {val:.6f}")
    
    # Load and apply the scaler
    print("\n2. APPLYING SCALER:")
    print("-" * 40)
    scaler_path = Path('strategy_models/lstm_scaler_NVDA.pkl')
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the features
        features_scaled = scaler.transform(features_raw.reshape(1, -1))
        print("Scaled features:")
        for i, val in enumerate(features_scaled[0], 1):
            print(f"  lag_{i}_scaled: {val:.6f}")
    else:
        print("ERROR: Scaler not found!")
        return
    
    # Reshape for LSTM (batch_size=1, timesteps=5, features=1)
    lstm_input = features_scaled.reshape(1, 5, 1)
    print(f"\n3. LSTM INPUT SHAPE: {lstm_input.shape}")
    
    # Load the Keras model directly
    print("\n4. LOADING KERAS MODEL:")
    print("-" * 40)
    model_path = Path('strategy_models/lstm_algorithm_NVDA.keras')
    if model_path.exists():
        model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Make prediction
        print("\n5. PREDICTION:")
        print("-" * 40)
        raw_prediction = model.predict(lstm_input, verbose=0)
        print(f"Raw prediction value: {raw_prediction[0][0]:.6f}")
        binary_signal = 1 if raw_prediction[0][0] > 0 else -1
        print(f"Binary signal: {binary_signal}")
        print(f"Expected from backtesting: 1")
        
        if binary_signal == 1:
            print("\n✓ SUCCESS: Model returns +1 as expected!")
        else:
            print("\n✗ ISSUE: Model returns -1, not matching backtesting")
            
            # Try with slightly different feature values (what we actually got)
            print("\n6. TESTING WITH ACTUAL LIVE FEATURES:")
            print("-" * 40)
            features_live = np.array([
                -0.000846,  # lag_1 (matches)
                0.037753,   # lag_2 (slightly different)
                0.014633,   # lag_3 (slightly different)
                0.007761,   # lag_4 (slightly different)
                -0.027093   # lag_5 (slightly different)
            ])
            
            print("Live features (small differences):")
            for i, (csv_val, live_val) in enumerate(zip(features_raw, features_live), 1):
                diff = abs(csv_val - live_val)
                print(f"  lag_{i}: CSV={csv_val:.6f}, Live={live_val:.6f}, Diff={diff:.6f}")
            
            # Test with live features
            features_live_scaled = scaler.transform(features_live.reshape(1, -1))
            lstm_input_live = features_live_scaled.reshape(1, 5, 1)
            raw_pred_live = model.predict(lstm_input_live, verbose=0)
            binary_live = 1 if raw_pred_live[0][0] > 0 else -1
            print(f"\nLive features prediction: {raw_pred_live[0][0]:.6f} -> {binary_live}")
    else:
        print(f"ERROR: Model not found at {model_path}")
    
    # Also test the cached model wrapper
    print("\n7. TESTING CACHED MODEL WRAPPER:")
    print("-" * 40)
    cached_path = Path('strategy_models/.cache/NVDA_lstm_algorithm_NVDA_converted.pkl')
    if cached_path.exists():
        with open(cached_path, 'rb') as f:
            wrapper = pickle.load(f)
        
        # Test with exact CSV features
        lstm_input_csv = features_scaled.reshape(1, 5, 1)
        wrapper_pred = wrapper.predict(lstm_input_csv)
        wrapper_signal = 1 if wrapper_pred[0] > 0 else -1
        print(f"Wrapper prediction: {wrapper_pred[0]:.6f} -> {wrapper_signal}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_with_exact_features()