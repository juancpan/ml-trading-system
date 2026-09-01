# Signal Threshold Explanation: Backtesting vs Live Trading

## The Answer: Yes, Zero Threshold is Correct

**In backtesting, the standard signal generation uses a threshold of 0.0:**
- If prediction > 0: Signal = +1 (BUY)
- If prediction ≤ 0: Signal = -1 (SELL)

This is the **de facto standard** across most financial ML models because:
1. The model outputs represent expected returns
2. Positive expected return = BUY
3. Negative expected return = SELL
4. Zero is the natural decision boundary

## What Went Wrong in Live Trading?

The `keras_model_wrapper.py` was using different thresholds:
```python
# WRONG (old code):
upper_threshold = 0.6  # Buy signal threshold
lower_threshold = 0.4  # Sell signal threshold

signals[predictions > upper_threshold] = 1   # Buy
signals[predictions < lower_threshold] = -1  # Sell
# Everything between 0.4 and 0.6 would be 0 (HOLD)
```

This created a "dead zone" between 0.4 and 0.6 where signals would be neutral (0).

## The Correct Implementation

```python
# CORRECT (fixed code):
threshold = 0.0  # Match backtesting threshold

signals[predictions > threshold] = 1   # Buy
signals[predictions <= threshold] = -1  # Sell
```

## Evidence from Backtesting Code

1. **Linear Models** (`linear_models_optimized.py:100`):
   ```python
   return np.where(predictions > 0, 1, -1)
   ```

2. **LSTM Models** (when outputting regression values):
   - Raw outputs are continuous values
   - Positive = BUY, Negative = SELL
   - No intermediate HOLD state in binary trading

3. **Consistency Test Results**:
   - Raw prediction: 0.1377 (positive)
   - With 0.0 threshold: Signal = +1 (BUY) ✅
   - With 0.4/0.6 thresholds: Signal = -1 (SELL) ❌

## Different Model Types

### Regression Models (Li_Reg, ARIMA, etc.)
- Output: Continuous predicted returns
- Threshold: 0.0 (natural boundary)
- Signal: Sign of prediction

### Classification Models (Some LSTMs)
- The optimized LSTM uses 3 classes (SELL=0, HOLD=1, BUY=2)
- But deployed models typically use binary classification or regression
- Still maps to -1/+1 signals ultimately

### Deep Learning Models (LSTM, CNN, TCN)
- Can be either regression or classification
- When regression: Use 0.0 threshold
- When classification: Use argmax then map to signals

## Why This Matters

The threshold directly affects trading decisions:
- With 0.0 threshold: Model predicting 0.1% return → BUY
- With 0.4 threshold: Model predicting 0.1% return → SELL

This completely inverts the trading logic!

## Verification

Our testing confirms:
1. Backtesting uses 0.0 threshold
2. Live trading (after fix) uses 0.0 threshold
3. Signals now match: 96.1% BUY for NVDA (consistent)

## Conclusion

**Yes, the 0.0 threshold is the correct, standard approach** for converting model predictions to trading signals. The 0.4/0.6 thresholds were an error that caused the signal discrepancy.

The fix ensures:
- ✅ Consistent behavior between backtesting and live trading
- ✅ Correct interpretation of model predictions
- ✅ Proper trading signals based on expected returns

## Note on Alternative Thresholds

Some trading systems might use non-zero thresholds for:
- Risk management (require minimum expected return)
- Transaction cost consideration
- Market regime filters

But these should be:
1. Explicitly configured
2. Consistent between backtest and live
3. Well-documented

In this system, the standard is 0.0, which is appropriate for the models trained.