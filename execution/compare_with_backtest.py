#!/usr/bin/env python3
"""
Compare live trading features with backtesting features.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf

# Add algos path for backtesting imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "algos" / "common"))

def compare_features():
    print("\n" + "="*80)
    print("COMPARING LIVE VS BACKTEST FEATURE GENERATION")
    print("="*80)
    
    symbol = 'NVDA'
    lags = 5
    
    # 1. Load data same as backtesting
    print("\n1. LOADING DATA (like backtesting)...")
    print("-" * 40)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years like backtesting
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval='1d')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # 2. Generate features like backtesting
    print("\n2. GENERATING FEATURES (backtesting method)...")
    print("-" * 40)
    
    # Calculate returns (SIMPLE, not LOG!)
    df['returns'] = df['Close'].pct_change()
    
    # Create lagged features
    feature_cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        df[col] = df['returns'].shift(lag)
        feature_cols.append(col)
    
    # Drop NaN rows
    df_clean = df.dropna()
    
    # Get last row features (unscaled)
    last_features = df_clean[feature_cols].iloc[-1].values
    print(f"Last features (unscaled): {last_features}")
    
    # 3. Load scaler and apply
    print("\n3. APPLYING SCALER...")
    print("-" * 40)
    
    scaler_path = Path("strategy_models/lstm_scaler_NVDA.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    last_features_scaled = scaler.transform(last_features.reshape(1, -1))
    print(f"Last features (scaled): {last_features_scaled[0]}")
    
    # 4. Load model and predict
    print("\n4. MAKING PREDICTION...")
    print("-" * 40)
    
    model_path = Path("strategy_models/NVDA_trading_model_lstm.keras")
    model = tf.keras.models.load_model(model_path)
    
    # Reshape for LSTM
    X = last_features_scaled.reshape(1, lags, 1)
    print(f"Input shape for LSTM: {X.shape}")
    
    # Predict
    pred_raw = model.predict(X, verbose=0)
    pred_signal = 1 if pred_raw[0][0] > 0 else -1
    
    print(f"Raw prediction: {pred_raw[0][0]}")
    print(f"Signal: {pred_signal}")
    
    # 5. Now compare with live trading
    print("\n5. COMPARING WITH LIVE TRADING...")
    print("-" * 40)
    
    from data_manager import DataManager
    import logging
    
    logger = logging.getLogger(__name__)
    data_manager = DataManager(logger=logger)
    
    # Fetch data
    from datetime import date
    data_manager.fetch_and_store_historical_data(symbol, date.today())
    
    # Get sequence data
    live_features = data_manager.create_sequence_data(symbol, lags=lags)
    print(f"Live features shape: {live_features.shape}")
    print(f"Live features (scaled): {live_features.flatten()}")
    
    # Compare
    print("\n6. COMPARISON...")
    print("-" * 40)
    print(f"Backtest features: {last_features_scaled[0]}")
    print(f"Live features:     {live_features.flatten()}")
    
    diff = np.abs(last_features_scaled[0] - live_features.flatten())
    print(f"Difference:        {diff}")
    print(f"Max difference:    {diff.max()}")
    
    if diff.max() < 0.01:
        print("✅ Features match!")
    else:
        print("❌ Features don't match!")
        
        # Debug the raw returns
        print("\n7. DEBUGGING RAW RETURNS...")
        print("-" * 40)
        
        hist_df = data_manager.historical_data[symbol]
        print("\nLive trading last 5 close prices:")
        print(hist_df['Close'].tail(7))
        
        print("\nBacktest last 5 close prices:")
        print(df['Close'].tail(7))

if __name__ == "__main__":
    compare_features()