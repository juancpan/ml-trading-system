# ML Trading System

**Walk-forward-validated ML strategies with a live-execution layer for Interactive Brokers.**

A personal research + trading platform: a 28-model strategy lab, a statistically
rigorous walk-forward validation framework (WFOV) designed to *deflate* the
results data mining produces, an anti-overfitting research governance protocol,
and a production execution engine with layered risk controls — connected
end-to-end so a model only ever trades after clearing the same validation
stack in backtest and live.

> [!IMPORTANT]
> **Read [DISCLAIMER.md](DISCLAIMER.md) first.** This repo is an educational
> portfolio piece: it is an incomplete, curated subset of a larger private
> system; it ships **no data, no trained weights, and makes no performance
> claims**. Nothing here is investment advice.

Author: [juancpan](https://github.com/juancpan) · License: [MIT](LICENSE)

---

## Why this repo exists

Most retail "ML trading bot" repos stop at a fitted backtest. The interesting
problems start *after* the backtest looks good: multiple-testing inflation,
in-sample leakage, seed luck, overfitting to one regime, and the operational
reality of trading real capital (stale data, stale models, margin, partial
fills, your own panic). This repo documents a system built to survive those
problems:

- **Validation before capital.** No model reaches the execution layer without
  clearing walk-forward validation, a Deflated Sharpe gate, and a trials
  budget (see [Research methodology](#research-methodology)).
- **Research governance.** Live strategies are governed by a written
  anti-data-mining contract (`docs/REVISION_POLICY.md`): pre-registered
  hypotheses, a minimum time-between-revisions rule, and a trials ledger —
  every test costs a trial.
- **Layered risk controls.** Kill switch → per-position circuit breaker →
  preflight gates → order guard, each independently testable.

## Research methodology

The differentiating component is **WFOV v2** (`algos/wfov/`) — a validation
framework built around the López de Prado toolkit:

| Stage | What it does |
|---|---|
| `window_generator.py` | True walk-forward windows — expanding or rolling — plus a Monte Carlo resampling mode for window-robustness checks |
| `embargo_utils.py` | 2% train/test embargo to kill leakage from overlapping labels |
| `statistical_tests.py` | Newey-West t-tests, bootstrap confidence intervals (percentile + BCa), **Deflated Sharpe Ratio**, **Probability of Backtest Overfitting (PBO)**, Bonferroni / Benjamini-Hochberg multiple-testing corrections |
| `regime_analyzer.py` | Performance conditional on market regime — a model that only works in bull markets is a bug, not a strategy |
| `model_ranker.py` | Tiered DEPLOY / REVIEW / REJECT recommendations from the validation statistics |
| `trials_ledger.py` | SQLite ledger of every validation trial — the denominator for DSR and the enforcement point for the research protocol |

Around it, a research protocol (fully written up in `docs/RESEARCH_MANUAL.md`
and `docs/REVISION_POLICY.md`):

- **Two research regimes.** Free exploration is allowed only *before*
  capital is deployed. Live strategies can only be revised through
  pre-registered hypotheses, under a minimum-time-between-revisions (MinTRL)
  rule, with every revision gated by `scripts/revision_check.py` against the
  trials ledger.
- **Reproducibility.** Fixed seeds (`algos/common/seed_manager.py`), scaler
  auto-save, pinned data windows. A result you cannot reproduce is not a
  result.
- **Falsifiable claims.** "Looks good" is not a claim; "OOS Sharpe ≥ X on a
  non-overlapping window, surviving 2× transaction costs, DSR > 0.5 at N
  trials" is.

## Architecture

```
                 ┌────────────────────────────────────────────────────────┐
                 │                     DATA LAYER                         │
                 │  algos/common/: update_market_data, data_loader,      │
                 │  ibkr_downloader, market_data_store (parquet),        │
                 │  yf_downloader (fallback), fred/cot downloaders       │
                 └────────────────────────────┬───────────────────────────┘
                                              │
                 ┌────────────────────────────▼───────────────────────────┐
                 │              FEATURE ENGINE (shared, config-driven)     │
                 │  algos/common/feature_engine.py + feature_config.yaml   │
                 │  lagged returns · trend · momentum · volatility ·       │
                 │  Hurst exponent · volume · external macro (VIX, ...)   │
                 └────────────────────────────┬───────────────────────────┘
                                              │ identical features
                 ┌────────────────────────────▼───────────────────────────┐
                 │                  MODEL LAB (28 models)                  │
                 │  algos/backtest_code/models/ — SVM, LSTM, DQN, TCN,     │
                 │  XGBoost, ARIMA/SARIMAX/VAR, RF, ensembles, ...          │
                 │  run_backtest_optimized.py · portimization.py (MIQP/HRP)│
                 └────────────────────────────┬───────────────────────────┘
                                              │
                 ┌────────────────────────────▼───────────────────────────┐
                 │            WALK-FORWARD VALIDATION (WFOV v2)             │
                 │  algos/wfov/ — expanding/rolling windows, Monte Carlo,    │
                 │  embargo, Newey-West, bootstrap CI, DSR, PBO,            │
                 │  regime analysis, model ranking, trials ledger           │
                 └────────────────────────────┬───────────────────────────┘
                                              │ DEPLOY tier only
                 ┌────────────────────────────▼───────────────────────────┐
                 │                    DEPLOYMENT                           │
                 │  scripts/model_selection_workflow.py · deploy_models.py │
                 │  retrain_models.py · revision_check.py (trials gate)    │
                 └────────────────────────────┬───────────────────────────┘
                                              │
                 ┌────────────────────────────▼───────────────────────────┐
                 │              EXECUTION LAYER (execution/)                │
                 │  main.py --region US|EUROPE|ASIA|...                     │
                 │  strategy_executor → portfolio_manager → order_guard →   │
                 │  ib_client_final (native ibapi) · limit_order_engine ·   │
                 │  cash_portfolio_manager (FX carry)                       │
                 └────────────────────────────┬───────────────────────────┘
                                              │
                 ┌────────────────────────────▼───────────────────────────┐
                 │           RISK & OVERSIGHT (execution/, scripts/)        │
                 │  kill_switch (MTD hard/soft) · position_circuit_breaker  │
                 │  preflight_check · nav_quick · run_region.sh gates      │
                 │  attribution.py · signal_history.py · shadow_check.py    │
                 │  sunday_maintenance.sh · retirement_check.py            │
                 └────────────────────────────────────────────────────────┘
```

## Key features

**Research core (`algos/`)**
- 28 model implementations registered behind a common `BaseStrategyModel`
  interface, with optimized variants (GPU/batch/caching) — LSTM, DQN, TCN,
  XGBoost, SVM, ARIMA/SARIMAX/VAR, random forest, gradient boosting,
  stacking/voting ensembles, and more
- Config-driven feature engineering shared byte-for-byte by backtest, WFOV,
  and live trading (feature_hash tracking — features can't silently drift)
- Portfolio construction: HRP, efficient frontier, budget-constrained MIQP
  (`portimization.py`), with gated OOS validation (`weekly_gate_engine.py`,
  `validate_gated_portfolio_oos.py`)

**Validation & governance (`algos/wfov/`, `scripts/`, `docs/`)**
- WFOV v2 with the full statistical stack (DSR, PBO, Newey-West, bootstrap,
  multiple-testing corrections)
- Trials ledger + pre-registration + revision/retirement checks — the
  anti-data-mining machinery is code, not vibes

**Execution layer (`execution/`)**
- IBKR integration via native `ibapi` across US, European, Asian, and other
  exchanges (region-based sessions, per-region client-id management)
- Risk stack: kill switch (hard kill / soft halt / daily-move alarm, sentinel
  files that fail safe), per-position circuit breaker, preflight gates
  (NAV, data staleness, model staleness, covariance sanity), order guard
  (pre-submission validation), transition safety (cross-margin limits)
- FX cash engine: multi-currency debt consolidation with an ML-timed
  USD/JPY carry leg
- `run_region.sh`: operator-facing gate chain — env bootstrap → kill-switch
  sentinels → NAV gate → preflight → trade, with distinct exit codes per
  failure class and Telegram alerting on every abort

**Testing discipline (`tests/`)**
- 31-module pytest suite (195 tests): kill-switch decisions, circuit-breaker
  math, trials-ledger accounting, portfolio deployer, symbol resolution,
  signal-history integrity, feature-engine edge cases
- Test isolation taken seriously: `conftest.py` scrubs real credentials at
  import time so the suite can never send a real Telegram alert (see
  [Testing](#testing))

## Quickstart

Python 3.9+ (developed on 3.11). TensorFlow 2.19 / Keras 3.9 for the deep
models; scikit-learn-classic models need none of that.

```bash
git clone https://github.com/juancpan/ml-trading-system.git
cd ml-trading-system
python -m venv .venv && source .venv/bin/activate    # or: conda create -n trading python=3.11
pip install -r requirements.txt
```

**1. Run a backtest** (data auto-downloads from Yahoo Finance on first run):

```bash
python algos/backtest_code/run_backtest_optimized.py \
    --model_name svm_optimized --ticker SPY --lookback_days 252 \
    --no-plots --skip-model-save
```

**2. Walk-forward validate the same model** (expanding windows, per-window
OOS Sharpe, full statistical summary written to `algos/wfov/results/`):

```bash
python -m algos.wfov.wfov_runner \
    --mode walk_forward_expanding --model_name svm_optimized --ticker SPY \
    --initial_train_days 504 --test_days 84 --step_days 84 --no-plots
```

**3. Run the test suite:**

```bash
pytest tests/ -q        # 195 passed, 6 skipped at time of publishing
```

The execution layer additionally requires an Interactive Brokers
TWS/Gateway session (`ibapi` connects to `127.0.0.1:4002` paper / `7497`),
plus a configured `execution/config.py` and `.env` — see the disclaimers
before pointing it at anything real.

## The pipeline, stage by stage

| Stage | Entry point | What happens |
|---|---|---|
| **Data** | `python -m algos.common.update_market_data` | Refreshes the parquet store (IBKR primary; yfinance fallback in `data_loader`) |
| **Features** | `feature_config.yaml` → `algos/common/feature_engine.py` | One feature definition, consumed identically by backtest/WFOV/live |
| **Backtest** | `algos/backtest_code/run_backtest_optimized.py` | Train + backtest any of 28 models with transaction costs, embargo, risk metrics |
| **Portfolio** | `algos/backtest_code/portimization.py`, `portfolio_exploration_global.py` | HRP / efficient-frontier / MIQP weight construction; OOS gating via `weekly_gate_engine.py` |
| **Validate** | `python -m algos.wfov.wfov_runner --mode walk_forward_expanding ...` | Walk-forward + Monte Carlo validation, DSR/PBO/Newey-West/bootstrap, regime analysis |
| **Select** | `python scripts/model_selection_workflow.py --ticker SPY --preset comprehensive` | Batch model comparison over the ticker universe, tiered ranking (DEPLOY/REVIEW/REJECT) |
| **Deploy** | `python deploy_models.py --portfolio weights.json --dry-run` | Deploys pickles + scalers + config updates; `--dry-run` previews; trials-ledger gate unless `--bypass-trials-check` (audited) |
| **Retrain** | `python scripts/retrain_models.py --tickers SPY` | Retrains stale models from the refreshed parquet store |
| **Execute** | `cd execution && python main.py --region US` or `./run_region.sh US` | Signal → position sizing → order guard → IBKR. `run_region.sh` enforces the full gate chain |
| **Maintain** | `scripts/sunday_maintenance.sh`, `scripts/revision_check.py`, `scripts/retirement_check.py` | Weekly data/model maintenance; revision protocol gates; strategy retirement rules |

## Repository structure

```
├── algos/
│   ├── backtest_code/        # 28 model implementations, runners, portfolio optimization
│   ├── common/               # data loading, feature engine, metrics, embargo, seeds, downloaders
│   ├── wfov/                 # walk-forward validation framework (the statistical core)
│   └── tests/                # unit tests for the research core
├── execution/                # IBKR live-trading engine (risk modules, order pipeline, cash/FX)
├── scripts/                  # MLOps: model selection, retrain, revision/retirement gates, weekly maintenance
├── tests/                    # cross-system pytest suite (32 modules)
├── docs/                     # technical guides: WFOV summary, research manual, revision policy, model guides
├── deploy_models.py          # validated-model deployer (portfolio weights → execution layer)
├── validate_config.py        # pre-flight configuration validator
├── feature_config.yaml       # the single feature definition (backtest = WFOV = live)
├── data_config.yaml          # data-source routing (yfinance / hybrid / IBKR)
├── requirements.txt
├── DISCLAIMER.md             # read this
└── TODO.md                   # honest roadmap
```

## Testing

```bash
pytest tests/ -q
```

The suite covers the parts of the system where silent failure is expensive:
kill-switch threshold decisions, per-position circuit-breaker accounting,
trials-ledger scoped counting, the portfolio deployer's config rewriting,
IBKR symbol → config-key resolution (a real incident: a European position
booked under the wrong exchange key), signal-history integrity, and
feature-engine edge cases (all-NaN columns, zero-volume forex series).

Two engineering details worth reading in `tests/conftest.py`:

- **Credential scrubbing at import time.** An earlier version of the suite
  once fired *real* Telegram alerts from synthetic kill-switch tests because
  a `.env` with production credentials was present. The conftest now scrubs
  alert credentials before any test module imports the alerting path, and
  the `.env` loader refuses to load inside pytest. Defense in depth against
  your own test suite.
- **The IBKR `error()` signature rule** (`execution/test_ibkr_error_signature.py`):
  IBAPI ≥ 10.37 requires a 6-argument `error()` callback; a wrong signature
  crashes *inside the API reader thread* and masquerades as a connection
  timeout. The test mechanically asserts the signature on every `EWrapper`
  subclass so it can never regress silently.

## Status — what this repo is *not*

- **Not a turnkey trading bot.** It's a research system with an execution
  layer attached, published to show methodology and engineering. Read
  [DISCLAIMER.md](DISCLAIMER.md).
- **Not complete.** No market data, no trained weights, no crypto module
  (the private system has one), no monitoring dashboard, and a few ops
  scripts referenced by tests are absent (noted in [TODO.md](TODO.md)).
- **Not a track record.** No performance figures are claimed; any numbers in
  example output are synthetic/from published demo runs.
- **Example config only.** `execution/config.py` ships a placeholder ETF
  portfolio, not any real allocation.

## Roadmap

See [TODO.md](TODO.md) — CI, packaging, CPCV, demo notebook, and the honest
list of what's still missing in this public subset.

## License

MIT — see [LICENSE](LICENSE). Educational use; no warranty; not investment
advice.
