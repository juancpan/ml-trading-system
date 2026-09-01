# Kill Switch & Risk Procedures

**Deployment:** H2 2026 HRP portfolio (17 tickers, NAV ~$10.5k, 1.3x leverage)
**Last updated:** 2026-04-21
**Owner:** solo operator

This document is a **manual procedure runbook**, not automated code. Automation is deferred to Tier 3. When a threshold is breached, *you* are the circuit breaker.

---

## Thresholds (read this section at the start of every week)

| Condition | Threshold | Action |
|---|---|---|
| Hard kill — MTD drawdown | ≥ 8% | **Flatten** all non-{BIL, TLT} to cash. Hold BIL+TLT+GLD only. Halt main loop. |
| Soft warning — MTD drawdown | ≥ 5% | **Halt new `ml_signal` entries.** `buy_and_hold` continues. Monitor hourly. |
| Daily-move alarm | Portfolio moves ≥ 4% in one day | **No rebalance.** Manual review next morning. Run preflight. |
| Model staleness | Any ML model mtime ≥ 120 days | **Disable that gate.** Hold the ticker as `buy_and_hold` target until retrain. |
| Data staleness (preflight Check B) | Any ticker parquet ≥ 5 days old | Do not run rebalance. Update market data first. |
| Covariance alarm (preflight Check C) | LedoitWolf shrinkage > 0.50 | **Reject new HRP weights.** Hold prior week's allocation. Investigate outlier. |
| FX rate drift | CURRENCY_RATE_FALLBACKS off by > 5% vs live | Update rates before next rebalance. Check F will mis-size otherwise. |
| Single-ticker 1-day log-return | > 0.30 | Preflight Check B auto-rejects. Exclude the ticker or fix the data. |
| JPY carry debt ceiling | JPY debt > 200% of NAV | Cap further conversion; do not add to the position (existing debt is not force-unwound). See `STRATEGY_MODE.md`. |

---

## Procedure: Hard Kill (MTD drawdown ≥ 8%)

**Trigger:** you observe `(current_NAV / month_start_NAV - 1) ≤ -0.08` via IBKR account summary.

**Steps (execute in order, do not skip):**

1. **Stop the automation.**
   - Comment out the cron entries in `crontab_regions.txt` (or `crontab -e`).
   - Verify no `main.py` process is running: `pgrep -f "execution/main.py"`.

2. **Freeze position book.**
   - Open IBKR TWS / Gateway manually.
   - Cancel all open orders via TWS → "Trades" → "Cancel All".

3. **Flatten risk positions.**
   - Close all positions EXCEPT BIL, TLT, GLD.
   - Use MARKET orders for US tickers.
   - Use LIMIT orders 50 bps inside spread for EU tickers (limited liquidity).
   - TELEKOM.BD may take 2–3 trading sessions to fully exit cleanly.

4. **Post-mortem (before restart).**
   - Create `logs/kill_switch_{YYYY-MM-DD}.log` with:
     - Trigger date/time, NAV trajectory for the MTD window.
     - Positions held at trigger, realized P&L per ticker on flatten.
     - Per-ticker ML signals in the 5 days leading up to trigger.
     - Macro context (Fed, geopolitics, earnings events).
   - Decide: (a) resume with unchanged config, (b) adjust weights/thresholds, (c) halt indefinitely.

5. **Resume checklist.**
   - Run `preflight_check.py --nav <new_NAV> --with-ibkr`.
   - Re-enable cron only after preflight passes.
   - Consider reducing `GENERAL_LEVERAGE` from 1.3 to 1.0 for the first month of resume.

---

## Procedure: Soft Warning (MTD drawdown ≥ 5%)

1. Do **not** flatten. This is a normal drawdown that may reverse.
2. **Halt new `ml_signal` entries:**
   - Edit `config.py`: temporarily flip each `ml_signal` ticker to `buy_and_hold` (AAA, TLT, GLD, TELEKOM.BD).
   - This prevents ML models from entering positions during stressed regimes where their training distribution may no longer apply.
3. Monitor daily. If MTD drops to −8%, execute Hard Kill.
4. If MTD recovers to −3%, revert to `ml_signal` mode.

---

## Procedure: Daily-move Alarm (portfolio ±4% in a day)

1. **Do not run the scheduled rebalance** that evening.
2. Run `preflight_check.py --nav <current_NAV> --verbose`:
   - Check B will identify any ticker with a suspicious single-day return.
   - Check C may fail if the move corrupted covariance.
3. If a single ticker drove the move (>2% contribution):
   - Verify corporate action (split, dividend, M&A) via IBKR news or the issuer's IR page.
   - If unadjusted corporate action: exclude the ticker, redistribute to BIL, re-run preflight.
4. Resume rebalance next day.

---

## Procedure: Model Staleness (any model ≥ 120d old)

1. Preflight Check D will fail.
2. Temporarily edit `config.py`: switch that ticker from `ml_signal` to `buy_and_hold`.
3. Rerun preflight — should PASS.
4. Retrain the model at your earliest opportunity (ideally within 1 week).
5. After retrain, revert to `ml_signal` and re-verify preflight.

---

## Procedure: Covariance Alarm (shrinkage > 0.50)

**This is the GVR.IR class of bug.** (Apr 2026 incident: an unadjusted reverse stock split produced an 11,892% single-day return and collapsed LedoitWolf to identity, making HRP output equal weights.)

1. Preflight Check C fails (or Check B fails first with an outlier).
2. **Do NOT accept the new HRP weights.** Keep last week's allocation.
3. Investigate:
   ```bash
   python3 -c "
   import pandas as pd, numpy as np
   for t in ['JKL.MC', 'MNO.LS', ...]:
       df = pd.read_parquet(f'data/market_data/{t}.parquet')
       ret = np.log(df['adj_close'] / df['adj_close'].shift(1)).dropna()
       if ret.abs().max() > 0.30:
           print(f'{t}: OUTLIER', ret.abs().idxmax(), ret.abs().max())
   "
   ```
4. Exclude the offending ticker (add to `BLACKLISTED_SYMBOLS` or remove from universe file).
5. Re-run HRP optimization (`portfolio_exploration_global.py`) without the bad ticker.
6. Re-run preflight.

---

## Small-NAV realities ($10.5k book)

These are structural, not emergency conditions. Monitor monthly.

### Commission drag
- IBKR minimum per-trade commissions at this NAV absorb ~30–50 bps per full rebalance.
- With 17 positions and weekly rebalance potential, cumulative annual drag could be 5–10%.
- **Track commission separately in monthly P&L.** If > 3%/month, reduce rebalance frequency.

### FX conversion minimums
- IBKR IDEALPRO has a $2 minimum per conversion.
- You transact in 4 currencies (USD, EUR, CAD, HUF).
- **Budget: max 2 FX conversion rounds per month.** Batch conversions in Phase 1/Phase 2.

### Integer-share rounding
- Preflight Check F tracks per-ticker weight drift from share rounding.
- At $10.5k NAV, drifts up to 12% (GLX) are acceptable.
- If drift exceeds 20% on any ticker, redistribute that weight to BIL.

### Minimum position viability
- Any target notional < $100 is economically fragile (commission > 2% of position).
- Current smallest positions: ENR.DE (removed 2026-04-21), GLX ($193).
- Watch for positions that drift below $100 due to price action; consider consolidation to BIL.

---

## Kill-switch contact info (fill in before deploying)

- Primary device: __________________
- Backup device (IBKR app installed): __________________
- Offline copy of this document: __________________
- IBKR account number: __________________
- IBKR customer service: +1-877-442-2757 (US) / your regional line

---

## Pre-deploy checklist (run every time)

- [ ] `python execution/preflight_check.py --nav <NAV>` passes
- [ ] `python execution/preflight_check.py --nav <NAV> --with-ibkr` passes
- [ ] Current MTD P&L reviewed against thresholds
- [ ] No model file >90 days old
- [ ] FX rates in `CURRENCY_RATE_FALLBACKS` within 5% of live spot
- [ ] This file read from top (5 minutes, cheap insurance)

## See also

- `execution/STRATEGY_MODE.md` — what this portfolio does and why
- `execution/preflight_check.py` — automated pre-deploy gate
- `execution/CONSISTENCY_CHECKLIST.md` — existing pre-trade checks
