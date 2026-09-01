# DQN Model Integration Guide

## Overview
This guide explains how to use DQN (Deep Q-Network) models trained with Keras in your live trading system, both for IBKR stock trading and crypto trading.

## Quick Start

### For UPRO Trading with IBKR

1. **Copy your trained DQN model to the right location:**
   ```bash
   # Copy the UPRO DQN model you already have
   cp algos/model_dumps/dqn_algorithm_UPRO_*.keras execution/strategy_models/upro_dqn_model.keras
   ```

2. **Update IBKR config to use DQN:**
   ```python
   # In execution/config.py
   ASSET_SPECIFIC_CONFIGS = {
       'UPRO': {
           'model_file': 'upro_dqn_model.keras',  # Will auto-convert
           'model_type': 'dqn',  # Explicitly specify DQN
           'lags': 5,  # Number of lag features used in training
           'kelly_fraction': 0.5,
           'max_leverage': 2
       }
   }
   ```

3. **Run live trading:**
   ```bash
   cd execution
   python main.py
   ```

### For Crypto Trading

1. **Place DQN model in models directory:**
   ```bash
   # The model will be auto-detected
   cp your_dqn_model.keras algos/backtest_code/best_models/dqn_BTC.keras
   ```

2. **Use in crypto trading:**
   ```bash
   cd crypto_trading
   python main.py --model dqn --symbol BTC/USDT
   ```

## How It Works

### Automatic Conversion
- **First Load**: When a `.keras` or `.h5` DQN model is encountered, it's automatically wrapped with `DQNModelWrapper`
- **Caching**: The wrapped model is cached as `.pkl` for faster subsequent loads
- **No Manual Conversion Needed**: The system handles everything automatically

### Signal Generation
DQN models output continuous values between -1 and 1 (tanh activation):
- Values > 0.1 → Buy signal (1)
- Values < -0.1 → Sell signal (-1)  
- Values between -0.1 and 0.1 → Hold signal (0)

### Feature Preparation
DQN models expect lag features from returns:
- Automatically calculates returns from price data
- Creates 5 lag features (configurable)
- Applies StandardScaler normalization
- Handles both DataFrame and numpy array inputs

## Architecture Support

### DQN Model Requirements
Your DQN model should have:
1. **Input**: Lag features (default: 5 features)
2. **Hidden Layers**: Dense layers with ReLU activation
3. **Output**: Single neuron with tanh activation

Example architecture:
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')  # Output: -1 to 1
])
```

## File Locations

### IBKR Trading
```
execution/
├── strategy_models/
│   ├── upro_dqn_model.keras     # Your DQN model
│   └── .cache/                   # Auto-generated cached versions
├── dqn_model_wrapper.py          # DQN wrapper implementation
└── strategy_executor.py          # Handles model loading
```

### Crypto Trading
```
crypto_trading/
├── strategy/
│   └── ml_adapter.py             # Handles DQN model loading
algos/
├── backtest_code/
│   └── best_models/
│       └── dqn_BTC.keras         # Your crypto DQN models
```

## Troubleshooting

### "TensorFlow not installed"
```bash
pip install tensorflow
```

### "Model input shape doesn't match"
Check the number of lag features:
- Default is 5 lags
- Configure in config: `'lags': 5`

### "Cannot load model"
Ensure your model:
1. Is saved in Keras format (.keras or .h5)
2. Has the correct architecture (see above)
3. Is in the right directory

### Performance Issues
- Models are cached after first load
- Cache location: `strategy_models/.cache/`
- Delete cache to force re-conversion

## Manual Conversion (Optional)

If you prefer to manually convert models:

```python
from dqn_model_wrapper import convert_dqn_to_pkl

# Convert DQN model
convert_dqn_to_pkl(
    'path/to/dqn_model.keras',
    'strategy_models/upro_dqn.pkl',
    lags=5
)
```

## Testing Your Setup

Run the integration test:
```bash
python test_dqn_integration.py
```

This will:
- Test DQN wrapper functionality
- Verify ML adapter integration
- Check strategy executor compatibility
- List all found DQN models

## Advanced Configuration

### Custom Thresholds
Modify signal thresholds in `dqn_model_wrapper.py`:
```python
def _convert_to_signals(self, predictions):
    threshold = 0.1  # Adjust this value
    signals[np.abs(predictions) < threshold] = 0
```

### Different Lag Features
For models trained with different lag counts:
```python
ASSET_SPECIFIC_CONFIGS = {
    'UPRO': {
        'model_file': 'upro_dqn_model.keras',
        'lags': 10,  # If trained with 10 lags
    }
}
```

### Scaler Persistence
The StandardScaler is fitted on first use. For production, consider:
1. Saving scaler from training
2. Loading pre-fitted scaler
3. Using training data statistics

## Your Existing Models

Found DQN models in your project:
1. `dqn_algorithm_UPRO_*.keras` - Ready for UPRO trading
2. `dqn_algorithm_GLD_*.keras` - For GLD trading
3. `dqn_algorithm_BTC-USD_*.keras` - For Bitcoin trading

To use them:
1. Copy to appropriate directory (see File Locations)
2. Update config with model name
3. Run trading system

## Summary

The DQN integration is now complete and tested. Your DQN models will:
- Load automatically when referenced
- Convert to appropriate format on first use
- Generate trading signals properly
- Work with both IBKR and crypto trading systems

No manual conversion needed - just place your `.keras` files in the right location and start trading!