#!/usr/bin/env python3
"""
Extract and save StandardScalers from backtesting for use in live trading.
Run this script to create scaler files for your models.
"""

import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import algos modules
sys.path.append(str(Path(__file__).parent.parent))

from config import ASSET_SPECIFIC_CONFIGS


def train_and_save_scaler(symbol, model_type, lags=5):
    """
    Train a StandardScaler on historical data and save it for live trading.
    
    Args:
        symbol: Trading symbol
        model_type: Type of model (lstm, li_reg, arima, etc.)
        lags: Number of lagged features
    """
    print(f"\nProcessing {symbol} ({model_type})...")
    
    try:
        # Load historical data using yfinance (same as live trading)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
        
        if data.empty:
            print(f"No data available for {symbol}")
            return False
        
        # Use Adj Close or Close
        if 'Adj Close' in data.columns:
            data['price'] = data['Adj Close']
        elif 'Close' in data.columns:
            data['price'] = data['Close']
        else:
            print(f"No price column found for {symbol}")
            return False
        
        # Calculate log returns (same as backtesting)
        data['returns'] = np.log(data['price'] / data['price'].shift(1))
        
        # Create lagged features (same as backtesting)
        feature_cols = []
        for lag in range(1, lags + 1):
            col = f'lag_{lag}'
            data[col] = data['returns'].shift(lag)
            feature_cols.append(col)
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        if data.empty:
            print(f"Insufficient data for {symbol} after creating lags")
            return False
        
        # Split data (same ratio as backtesting)
        split_ratio = 0.8  # Common training split
        split = int(len(data) * split_ratio)
        train_data = data.iloc[:split]
        
        # Fit StandardScaler on training features
        scaler = StandardScaler()
        train_features = train_data[feature_cols].values
        scaler.fit(train_features)
        
        # Save the scaler
        scaler_dir = Path('strategy_models')
        scaler_dir.mkdir(exist_ok=True)
        
        scaler_filename = f"{model_type}_scaler_{symbol}.pkl"
        scaler_path = scaler_dir / scaler_filename
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✓ Saved scaler to {scaler_path}")
        print(f"  Mean: {scaler.mean_[:3]}..." if len(scaler.mean_) > 3 else f"  Mean: {scaler.mean_}")
        print(f"  Scale: {scaler.scale_[:3]}..." if len(scaler.scale_) > 3 else f"  Scale: {scaler.scale_}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {symbol}: {e}")
        return False


def create_scalers_from_config():
    """
    Create scalers for all symbols defined in the config.
    """
    print("Creating StandardScalers for live trading...")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    for symbol, config in ASSET_SPECIFIC_CONFIGS.items():
        model_type = config.get('model_type', 'standard')
        lags = config.get('lags', 5)
        
        total_count += 1
        if train_and_save_scaler(symbol, model_type, lags):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Scaler creation complete: {success_count}/{total_count} successful")
    
    if success_count < total_count:
        print("\nNote: For symbols that failed, the system will use unscaled features")
        print("      This may cause prediction discrepancies")


def verify_scalers():
    """
    Verify that scalers exist for all configured symbols.
    """
    print("\nVerifying scaler files...")
    print("-" * 30)
    
    scaler_dir = Path('strategy_models')
    
    for symbol, config in ASSET_SPECIFIC_CONFIGS.items():
        model_type = config.get('model_type', 'standard')
        scaler_filename = f"{model_type}_scaler_{symbol}.pkl"
        scaler_path = scaler_dir / scaler_filename
        
        if scaler_path.exists():
            # Try to load it
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✓ {symbol}: {scaler_filename} (features: {scaler.n_features_in_})")
            except Exception as e:
                print(f"✗ {symbol}: {scaler_filename} exists but failed to load: {e}")
        else:
            print(f"✗ {symbol}: {scaler_filename} NOT FOUND")


if __name__ == "__main__":
    # Create scalers for all configured symbols
    create_scalers_from_config()
    
    # Verify the created scalers
    verify_scalers()
    
    print("\nDone! Your models should now use consistent scaling in live trading.")