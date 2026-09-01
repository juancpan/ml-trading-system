# Keras Models Integration Guide

## Overview
The trading system now **automatically supports Keras models** (.keras and .h5 files) without manual conversion!

## How It Works

### Automatic Conversion
When you specify a `.keras` or `.h5` file in your config, the system:
1. **Detects** the file format automatically
2. **Converts** it to a compatible format on first load
3. **Caches** the conversion for faster subsequent loads
4. **Updates** the cache if the original model changes

### Supported Model Types

#### Standard Neural Networks
- Dense/Feed-forward networks
- Convolutional Neural Networks (CNNs)
- Any model with fixed input size

#### Sequence Models (Auto-detected)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Units)
- TCN (Temporal Convolutional Networks)
- RNN (Recurrent Neural Networks)

The system automatically detects sequence models by their filename (if it contains 'lstm', 'gru', 'tcn', 'rnn', or 'temporal').

## Configuration

### Basic Usage
Simply specify your Keras model path in `config.py`:

```python
ASSET_SPECIFIC_CONFIGS = {
    'UPRO': {
        'kelly_fraction': 2.1743,
        'strategy_model_path': 'strategy_models/your_model.keras'  # Direct path!
    }
}
```

### Advanced Configuration
For sequence models, you can specify additional parameters:

```python
ASSET_SPECIFIC_CONFIGS = {
    'SPY': {
        'kelly_fraction': 2.0,
        'strategy_model_path': 'strategy_models/lstm_model.keras',
        'model_type': 'lstm',        # Optional: 'lstm', 'standard', or 'auto' (default)
        'sequence_length': 60         # Optional: For LSTM/TCN models (default: 60)
    }
}
```

## File Formats Supported

| Extension | Description | Auto-Convert |
|-----------|-------------|--------------|
| `.pkl` | Pickle format (sklearn, xgboost) | No (native) |
| `.keras` | TensorFlow/Keras SavedModel | Yes |
| `.h5` | Keras HDF5 format | Yes |

## Cache Management

### Cache Location
Converted models are cached in: `strategy_models/.cache/`

### Cache Files
- Format: `{SYMBOL}_{model_name}_converted.pkl`
- Example: `UPRO_tcn_algorithm_UPRO_converted.pkl`

### Cache Invalidation
The cache automatically updates when:
- The original `.keras` file is modified
- You delete the cache directory
- The conversion logic changes

### Manual Cache Clear
```bash
rm -rf strategy_models/.cache/
```

## Examples

### Example 1: Using a TCN Model
```python
# config.py
ASSET_SPECIFIC_CONFIGS = {
    'UPRO': {
        'kelly_fraction': 2.1743,
        'strategy_model_path': 'strategy_models/tcn_algorithm_UPRO_Adj Close_2020-01-01_2025-06-01_1d_20250819_162044.keras'
    }
}
# The system auto-detects 'tcn' in the filename and handles it as a sequence model
```

### Example 2: Mixed Model Types
```python
# config.py
ASSET_SPECIFIC_CONFIGS = {
    'SPY': {
        'kelly_fraction': 2.0,
        'strategy_model_path': 'strategy_models/spy_svm_model.pkl'  # Traditional sklearn
    },
    'QQQ': {
        'kelly_fraction': 1.5,
        'strategy_model_path': 'strategy_models/qqq_lstm.keras'    # Keras LSTM
    },
    'UPRO': {
        'kelly_fraction': 2.1,
        'strategy_model_path': 'strategy_models/upro_tcn.h5'       # Keras TCN
    }
}
```

### Example 3: Custom Signal Conversion
If your Keras model needs custom signal conversion, create a wrapper:

```python
# custom_wrapper.py
from keras_model_wrapper import KerasModelWrapper

class MyCustomWrapper(KerasModelWrapper):
    def _convert_to_signals(self, predictions):
        # Your custom logic here
        signals = np.sign(predictions - 0.5)  # Example
        return signals.astype(int)

# Convert and save
wrapper = MyCustomWrapper(model_path='my_model.keras')
wrapper.save_as_pkl('strategy_models/my_model_custom.pkl')
```

## Testing Your Setup

### Test Script
Run the provided test script to verify Keras support:
```bash
python test_keras_autoload.py
```

### Expected Output
```
✅ SUCCESS: Keras model loaded automatically!
✅ Model prediction successful: Signal = [1]
✅ Found 1 cached conversion(s)
   - UPRO_tcn_algorithm_UPRO_Adj Close_2020-01-01_2025-06-01_1d_20250819_162044_converted.pkl
```

## Requirements

### Required Package
```bash
pip install tensorflow  # For Keras model support
```

### Optional Packages
```bash
pip install tensorflow-gpu  # For GPU acceleration
pip install tcn            # For TCN models
```

## Troubleshooting

### Error: "TensorFlow/Keras not installed"
**Solution:** Install TensorFlow
```bash
pip install tensorflow
```

### Error: "Model input shape mismatch"
**Solution:** Ensure your feature engineering matches the model's expected input shape

### Slow First Load
**Normal behavior:** First load converts and caches the model. Subsequent loads use the cache and are much faster.

### Model Not Auto-Detected as Sequence
**Solution:** Either:
1. Rename file to include 'lstm', 'tcn', etc.
2. Explicitly set in config: `'model_type': 'lstm'`

## Performance Notes

1. **First Load:** May take 10-30 seconds for large models
2. **Cached Loads:** Typically < 1 second
3. **Memory Usage:** Keras models use more RAM than sklearn
4. **GPU Support:** Automatically uses GPU if available

## Migration Guide

### From Manual Conversion
**Before (Manual):**
```python
# Had to run this first:
python -c "from keras_model_wrapper import convert_keras_to_pkl; convert_keras_to_pkl('model.keras', 'model.pkl')"

# Then in config:
'strategy_model_path': 'model.pkl'
```

**Now (Automatic):**
```python
# Just use directly in config:
'strategy_model_path': 'model.keras'  # Done!
```

## Best Practices

1. **Name models descriptively:** Include model type in filename (e.g., `spy_lstm_model.keras`)
2. **Keep originals:** Don't delete `.keras` files after caching
3. **Version control:** Track `.keras` files, ignore `.cache/` directory
4. **Test locally:** Verify models work before deploying

## Summary

✅ **No manual conversion needed**
✅ **Automatic caching for performance**
✅ **Supports all Keras model types**
✅ **Backward compatible with `.pkl` files**
✅ **Smart detection of model types**

Your trading system is now fully compatible with both traditional sklearn models and modern deep learning models!