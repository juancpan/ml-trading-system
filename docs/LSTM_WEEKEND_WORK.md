# LSTM Weekend Work — Queue

**Created:** 2026-04-21 (night before H2 2026 deploy)
**Context:** During the preflight hardening session, we surfaced two
real bugs in the LSTM pipeline and one design question. Tonight's
deploy ships with the minimum safe fix (class-weighted training +
backtest/live signal alignment via signed confidence). The items below
are deferred research and improvements for the weekend.

Assume no action on these until the live deploy has at least a few
trading days under the new GLD LSTM and class-weighted backtest path.

---

## Items (ranked by value)

### 1. Ternary vs binary decision function — A-B study

**Background:** The LSTM emits a 3-class softmax
`[P(sell), P(hold), P(buy)]`. We have two ways to collapse it to a
trading signal:

- **Binary (current):** `conf = P(buy) - P(sell)`, then `np.sign()` with
  `0 -> +1`. Output in {-1, +1}. Always takes a position.
- **Ternary:** `argmax() - 1`. Output in {-1, 0, +1}. Can hold
  (stay flat).

For GLD specifically, ternary may outperform binary because GLD
bleeds in contango; sitting flat on "I don't know" days preserves
capital. For TLT/AAA where staying exposed collects carry/beta, binary
is likely fine.

**Task:** Implement ternary as an opt-in per-ticker config flag
(`"signal_mode": "ternary"` in `ASSET_SPECIFIC_CONFIGS`). Run GLD
through both decision functions in parallel for 4–8 weeks (shadow
mode), log both signals and hypothetical P&L, compare:

- Realized P&L under each
- Turnover (ternary should be lower)
- Max drawdown
- Hit rate conditional on taking a position

**Implementation notes from the rolled-back prototype (for reference
when this work actually begins):**

- `strategy_executor.py`: add `TERNARY_CAPABLE_MODEL_TYPES` constant
  covering lstm, lstm_optimized, dqn, cnn, tcn, rnn, dnn. In
  `generate_signal()` LSTM branch, after `raw_signal = predict()[0]`,
  check config `signal_mode`. If `"ternary"` and model has
  `predict_signals`, return `int(algorithm.predict_signals(features)[0])`
  directly and log with `TERNARY` prefix. Fall through to binary on any
  exception.
- `portfolio_manager.py:1097-1099` already handles `signal == 0`
  correctly ("no trade, keep current position") — no changes needed
  downstream.
- `lstm_optimized.py::run_lstm_strategy` — add a parallel debug column
  that emits the argmax prediction alongside the signed-confidence
  position, so backtests compare both without changing the "position"
  of record.
- Config: keep `signal_mode` default `"binary"`. Only set GLD to
  `"ternary"` after the A-B study shows positive edge.

**Do NOT skip to the flip.** We intentionally did not ship this tonight
because: (a) the deployed GLD LSTM was trained without class weights,
so its argmax is ~100% "hold" — enabling ternary would silently zero
out the hedge; (b) the 6-month live track record (Sharpe ~2.0) was
earned under binary, changing the decision function is unshadowed
risk.

### 2. Retrain deployed GLD LSTM with class weights

The class-weighted `train()` change (committed as part of tonight's
backtest/live alignment fix) lives in `lstm_optimized.py`. The model
currently deployed at `execution/strategy_models/VIXM_trading_model_lstm.pkl`
was trained before this fix and has softmax heavily biased toward the
"hold" class.

Under the live signed-confidence path this still produces sensible ±1
signals (see `run_backtest_optimized.py` output from 2026-04-21 23:09:
50/50 buy/sell split over 2 years, clear regime-change bullish in last
8 days). So deployment is safe.

But if/when you enable ternary (item 1), you MUST first retrain GLD
with class weights and redeploy the .pkl. Otherwise argmax on the
current model is effectively a no-op → GLD position = 0 every day →
no tail hedge.

**Task (only when starting item 1):**
```bash
cd algos/backtest_code
python run_backtest_optimized.py --model_name lstm_optimized \
    --ticker GLD --lookback_days 730 --rf_rate 0.04 \
    --max_leverage 2.0 --train_split 0.98 --embargo_pct 0.0
# Copy the resulting .pkl from algos/model_dumps/ to
# execution/strategy_models/VIXM_trading_model_lstm.pkl
# (or use the existing deploy_models.py workflow)
```

### 3. Understand ml_signal == -1 semantics end-to-end

Open question from `STRATEGY_MODE.md`: when an `ml_signal` ticker gets
a `-1` signal, does the live system go **short** or go **flat**?

Reading `portfolio_manager.py:1053-1095`:
- If `min_position_shares` is set and current > min: sell down to min.
- Else: sell all (exit to 0).

So `-1` means **flatten**, not **go short**. The ML overlay is
effectively a long/flat filter on top of the HRP weights, not a
long/short model. This is probably what you want, but confirm:

- Does any code path submit a short order on `signal == -1`?
- Does IBKR margin account permit short positions on these specific
  tickers (TLT, AAA, GLD, TELEKOM.BD)? Some international tickers may
  be hard-to-borrow.
- Is the backtest's assumption `strategy = position.shift(1) * returns`
  (where position ∈ {-1, 0, +1}) overstating returns by assuming free
  shorts?

**Task:** Grep for "SELL SHORT" or `shortSell` in order_manager.py /
ib_client_final.py. Trace what IBKR order action is submitted when
`shares_to_trade < 0` and `current_shares == 0`. Document the answer
in `STRATEGY_MODE.md` under "Open questions".

### 4. ML model load-failure fallback behavior

If `VIXM_trading_model_lstm.pkl` fails to unpickle at startup (file
corrupt, TF version mismatch, ml_dtypes drift), what happens?

Possibilities:
- Abort main loop → no trading (safe but loud)
- Fall back to `DummyAlgorithm` (we saw one in strategy_executor.py:31)
  → signals would be arbitrary, silent failure
- Drop the ticker from signals dict → portfolio_manager skips it

**Task:** Read `strategy_executor._load_strategy_algorithms` and
`DummyAlgorithm.predict()` (line 37-41). Confirm the actual fallback
path. Decide explicitly:

- Option A: abort deployment on any model load failure (strict)
- Option B: auto-downgrade the failed ticker to `buy_and_hold` (keep
  running, lose the ML gate)
- Option C: drop the ticker entirely (keep running, lose the position)

Implement whichever you pick, with a clear log line. Add a preflight
Check G that exercises each model's load path to catch issues before
market open.

### 5. Adaptive threshold review

The current threshold for 3-class labels (`0.5 * std`) was designed to
avoid degenerate "all hold" labels on T-bill ETFs. For high-kurtosis
assets like GLD it still produces 62% hold (vs TLT's 39%).

Alternatives to study:
- **Quantile-based threshold:** set threshold so each class gets ~33%
  of samples (guaranteed balance by construction).
- **Asymmetric threshold:** different cut for buy vs sell (GLD spikes
  up sharply, bleeds slowly → buy threshold should be higher).
- **Returns-normalized labels:** label = 2 if next_return > median of
  |returns|, else 0 if < -median, else 1. No sigma dependence.

**Task:** Run all three label schemes on GLD, TLT, AAA. Report:
- Class balance per scheme
- Validation accuracy per scheme
- Backtest Sharpe per scheme

Keep the current scheme unless a clearly better one emerges.

### 6. LSTM OOS validation on 2020-2022

Tier 2 item already in `KILL_SWITCH.md`. For LSTM models specifically:

- Retrain on data up to 2019-12-31
- Walk forward through COVID crash + recovery + 2022 bear
- Compare OOS Sharpe to in-sample Sharpe
- If OOS < 0.3 * IS, the model is regime-overfit; consider smaller
  sequence length, more regularization, or an ensemble

---

## What NOT to do this weekend

- Do not change signal decision logic while a live model is running
  money. If you must experiment, build the A-B infrastructure first.
- Do not touch `_convert_to_binary_signal` directly. Add logic at the
  call site in `generate_signal()` so the conversion function stays a
  pure utility.
- Do not retrain ALL ML models in one session. One at a time, with
  preflight between each.
- Do not bypass preflight. If Check B or Check C fails after a
  retrain, investigate — don't lower the threshold.

## Related files

- `algos/backtest_code/models/lstm_optimized.py` — train() + run_lstm_strategy()
- `execution/strategy_executor.py` — generate_signal() + _convert_to_binary_signal()
- `execution/keras_model_wrapper.py` — live LSTM wrapper
- `execution/portfolio_manager.py:970-1100` — signal → trade conversion
- `execution/STRATEGY_MODE.md` — current strategy documentation
- `execution/KILL_SWITCH.md` — thresholds and procedures
- `docs/CRITICAL_ISSUES_ANALYSIS.md` — older critical-bug register
