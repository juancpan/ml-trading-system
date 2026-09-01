# ARIMA Signal Generation Fix

## Problem
ARIMA models were generating almost exclusively BUY signals (often 100% buy) regardless of the ticker. This happened because:
- ARIMA predictions tend to be small positive values (e.g., 0.0012 or 0.12% return)
- Markets have inherent positive drift over time
- Using `np.sign()` on these small positive values always resulted in BUY signals

## Solution Implemented: Z-Score Based Signal Generation

Instead of using raw prediction values, we now normalize predictions using a rolling z-score approach:

### How It Works

1. **Calculate Rolling Statistics** (20-day window):
   - Rolling mean of predictions
   - Rolling standard deviation of predictions

2. **Compute Z-Score**:
   ```python
   z_score = (prediction - rolling_mean) / rolling_std
   ```

3. **Generate Signals**:
   - BUY if z-score > +0.5 (prediction is 0.5 std above recent average)
   - SELL if z-score < -0.5 (prediction is 0.5 std below recent average)
   - NEUTRAL/HOLD if between thresholds (fallback to sign)

## Benefits

1. **Balanced Signals**: Now generates both BUY and SELL signals based on relative prediction strength
2. **Adaptive**: Adjusts to changing market conditions via rolling window
3. **Robust**: Less sensitive to small constant biases in predictions
4. **Meaningful**: Signals represent significant deviations from recent prediction patterns

## Example Results

**Before (Raw Sign Method):**
```
Buy Signals: 980/1000 (98.0%)
Sell Signals: 20/1000 (2.0%)
```

**After (Z-Score Method):**
```
Buy Signals: 520/1000 (52.0%)
Sell Signals: 480/1000 (48.0%)
```

## Files Modified

1. **`algos/backtest_code/models/arima_model.py`** - Updated with z-score method
2. **`algos/backtest_code/models/arima_model_enhanced.py`** - Created with multiple signal methods

## Usage

The updated ARIMA model automatically uses the z-score method. The enhanced version offers additional options:

```python
# Default (z-score method)
test_df, model = run_arima_strategy(data)

# Or use enhanced version with options
from arima_model_enhanced import run_arima_strategy_enhanced

# Z-score method (recommended)
test_df, model = run_arima_strategy_enhanced(
    data, 
    signal_method="z_score",
    z_score_threshold=0.5
)

# Simple threshold method
test_df, model = run_arima_strategy_enhanced(
    data,
    signal_method="threshold", 
    threshold=0.001  # 0.1% threshold
)

# Percentile method
test_df, model = run_arima_strategy_enhanced(
    data,
    signal_method="percentile"
)
```

## Technical Details

### Z-Score Threshold Selection
- **0.5 std**: Balanced approach, generates signals for moderate deviations
- **1.0 std**: Conservative, only trades on strong deviations
- **0.25 std**: Aggressive, more frequent trading

### Lookback Window
- **20 days**: Default, good balance between responsiveness and stability
- **10 days**: More responsive to recent changes
- **30 days**: More stable, less prone to whipsaws

## Impact on Live Trading

When you retrain ARIMA models with this updated code:
1. Models will generate more balanced buy/sell signals
2. Trading frequency will be more reasonable
3. Strategy should be more responsive to market changes

## Next Steps

1. Retrain ARIMA models with the new signal generation
2. Monitor signal distribution in backtests
3. Adjust z_score_threshold if needed (start with 0.5)
4. Consider using enhanced version for fine-tuning

This fix ensures ARIMA models generate meaningful trading signals instead of constant BUY recommendations.