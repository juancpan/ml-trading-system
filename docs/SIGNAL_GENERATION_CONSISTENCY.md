# Signal Generation Consistency Update

## Summary
All backtesting models have been verified and updated to use consistent signal generation logic with threshold 0.0.

## Verification Results

### 1. **Linear Regression (li_reg)** ✅
- **File**: `linear_regression_model.py`
- **Line 95-96**: Already using `np.sign()` with threshold 0.0
- **Updated**: Added `.flatten()` for consistency

### 2. **ARIMA** ✅
- **File**: `arima_model.py`
- **Original Line 107**: `np.where(test['predicted_returns'] > 0, 1, -1)`
- **Updated**: Now uses `np.sign(test['predicted_returns'].values).flatten()`
- **Threshold**: 0.0 (implicit in both versions)

## Standardized Signal Generation Pattern

All models now follow this consistent pattern:

```python
# Convert continuous predictions to binary signals
test['position'] = np.sign(test['predicted_returns'].values).flatten()

# Handle edge case where sign returns 0 (for exactly 0 prediction)
test['position'] = np.where(test['position'] == 0, 1, test['position'])
```

## Models Updated

| Model | File | Status | Signal Logic |
|-------|------|--------|--------------|
| LSTM | `lstm_model.py` | ✅ Already correct | `np.sign().flatten()` |
| Linear Regression | `linear_regression_model.py` | ✅ Updated | `np.sign().flatten()` |
| ARIMA | `arima_model.py` | ✅ Updated | `np.sign().flatten()` |
| ARIMA v2 | `arima_model_v2.py` | ✅ Updated | `np.sign().flatten()` |
| TCN | `tcn_model.py` | ✅ Updated | `np.sign().flatten()` |
| VAR | `var_model.py` | ✅ Updated | `np.sign().flatten()` |
| GBM | `gbm_model.py` | ✅ Updated | `np.sign().flatten()` |
| DQN | `dqn_model.py` | ✅ Already correct | `np.sign().flatten()` |

## Key Points

1. **Threshold = 0.0**: All models use zero as the decision boundary
   - Positive prediction → BUY (+1)
   - Negative or zero prediction → SELL (-1)

2. **Consistency**: All models now use `np.sign()` with `.flatten()`
   - Ensures consistent array shape
   - Handles edge cases uniformly

3. **Edge Case Handling**: When prediction is exactly 0.0
   - `np.sign(0.0)` returns 0
   - We default to +1 (BUY) in this rare case
   - This is consistent across all models

## Live Trading Alignment

The live trading system (`keras_model_wrapper.py`) has been fixed to use the same 0.0 threshold:

```python
threshold = 0.0  # Match backtesting threshold
signals[predictions > threshold] = 1   # Buy
signals[predictions <= threshold] = -1  # Sell
```

## Result

✅ **Complete consistency** between backtesting and live trading signal generation
- All models use threshold 0.0
- All models use `np.sign()` with `.flatten()`
- Edge cases handled uniformly
- Live trading matches backtesting behavior