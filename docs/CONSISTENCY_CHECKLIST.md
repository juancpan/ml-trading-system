
# BACKTESTING TO LIVE TRADING CONSISTENCY CHECKLIST

## Model Files
- [ ] Models trained in backtesting are copied to execution/strategy_models/
- [ ] Model filenames match configuration in config.py
- [ ] Model types (lstm, li_reg, arima) match between systems

## Data Preprocessing
- [ ] Log returns used: np.log(price / price.shift(1))
- [ ] Lagged features created: lag_1, lag_2, ..., lag_5
- [ ] Same number of lags in both systems (default: 5)

## Feature Scaling
- [ ] StandardScaler fitted during backtesting training
- [ ] Scaler saved as {model_type}_scaler_{symbol}.pkl
- [ ] Scaler loaded and applied in live trading

## Signal Generation
- [ ] All models output binary signals (-1 or +1)
- [ ] Continuous outputs converted using np.sign() or threshold at 0
- [ ] No neutral signals (0) - default to 1 if exactly 0

## LSTM Specific
- [ ] Sequence length matches (use lags parameter)
- [ ] Features reshaped to (1, lags, 1)
- [ ] Scaled features used

## Configuration Parameters
- [ ] model_type specified for each symbol
- [ ] lags parameter consistent (default: 5)
- [ ] kelly_fraction for position sizing

## Data Sources
- [ ] yfinance used for both systems
- [ ] auto_adjust=True for adjusted prices
- [ ] Same date ranges for training

## Testing
- [ ] Run test_signal_consistency.py
- [ ] Verify binary signals for all models
- [ ] Compare with backtesting results CSV
