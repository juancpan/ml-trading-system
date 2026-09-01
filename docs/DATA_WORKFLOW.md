# Data Workflow Guide

Local parquet-based market data store with IBKR as primary data source and yfinance as fallback. All backtesting, portfolio optimization, and WFOV validation read from local parquet files -- no network needed at read time.

## Architecture

```
                     update_market_data.py
                        (DB Updater CLI)
                         |           |
                    IBKR TWS API   yfinance
                    (primary)      (fallback)
                         |           |
                         v           v
                   data/market_data/
                   ├── SPY.parquet          <-- one file per ticker
                   ├── NVDA.parquet
                   ├── EURUSD.parquet       <-- forex (=X stripped)
                   ├── 8058.T.parquet
                   ├── _metadata.json       <-- last-update timestamps
                   ├── ticker_universe.json  <-- what tickers to track
                   └── updater.log
                              |
                     MarketDataStore
                     (read-only class)
                              |
            +---------+-------+--------+-----------+
            |         |                |           |
       data_cache  data_loader   portimization  data_manager
       (backtest)  (legacy)      (portfolio)    (live trading)
            |         |                |           |
     run_backtest  run_backtest  portfolio_    ibkr main.py
     _optimized    (legacy)      exploration   (warmup only)
            |
       wfov_runner
       (WFOV validation)
```

## Quick Start

```bash
conda activate <your-env>

# 1. Initial population (one-time, takes hours for ~1300 tickers)
#    If IB Gateway is running on port 4001, IBKR is used as primary.
#    If not, falls back to yfinance automatically.
python -m algos.common.update_market_data --init

# 2. Verify the store was populated
python -m algos.common.update_market_data --status

# 3. Now backtesting reads from local parquet (instant, no network)
python algos/backtest_code/run_backtest_optimized.py --model_name svm_optimized --ticker SPY
```

## Updater Commands

### Daily incremental update

Only fetches missing days since the last update. Skips tickers that are already current.

```bash
python -m algos.common.update_market_data
```

### Initial population

Downloads full history (default 5 years / 1825 days) for every ticker in `ticker_universe.json`.

```bash
python -m algos.common.update_market_data --init
python -m algos.common.update_market_data --init --lookback-days 2520   # 10 years
```

### Update specific tickers

```bash
python -m algos.common.update_market_data --tickers SPY NVDA EURUSD=X 8058.T
```

### Weekly full-refresh

Re-downloads the last N trading days to catch retroactive split/dividend adjustments to `adj_close`.

```bash
python -m algos.common.update_market_data --full-refresh 5
```

### Force a specific data source

```bash
python -m algos.common.update_market_data --source yfinance   # skip IBKR even if available
python -m algos.common.update_market_data --source ibkr        # skip yfinance (IBKR only)
```

### Export CSV for portimization.py

Replaces `yfinance_downloader_v5.py`'s CSV generation.

```bash
python -m algos.common.update_market_data \
    --export-csv data/financial_data_combined_prices.csv \
    --start 2021-01-01 --end 2026-02-25
```

### Seed from existing CSV files

Migrate data from existing `data/financial_data_combined_prices_*.csv` into parquet without re-downloading. Only prices are preserved (OHLC are set equal to the single price column).

```bash
python -m algos.common.update_market_data --seed-from-csv
```

### Show store status

```bash
python -m algos.common.update_market_data --status
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--init` | off | Full population mode (downloads all history) |
| `--tickers TICK ...` | all | Specific tickers to update |
| `--lookback-days N` | 1825 | Days of history for `--init` |
| `--start DATE` | auto | Explicit start date (YYYY-MM-DD) |
| `--end DATE` | today | Explicit end date (YYYY-MM-DD) |
| `--full-refresh N` | 0 | Re-download last N days for corrections |
| `--seed-from-csv` | off | Migrate existing CSV data to parquet |
| `--export-csv PATH` | off | Export multi-ticker prices to CSV |
| `--source` | auto | Force `yfinance`, `ibkr`, or `auto` |
| `--workers N` | 4 | Download threads |
| `--max-retries N` | 50 | Max retries per ticker before giving up |
| `--status` | off | Print store summary and exit |
| `--data-dir PATH` | `data/market_data/` | Override store directory |
| `-v` / `--verbose` | off | DEBUG-level logging |

## Data Source Priority

Source priority is **asset-type-based**, not one-size-fits-all:

```
STOCKS/ETFs: yfinance primary
   - Provides Adj Close (split + dividend adjusted prices)
   - IBKR does NOT provide adjusted prices, so it cannot be primary for stocks
   - IBKR is used as fallback only if yfinance fails

FOREX: IBKR primary (if IB Gateway running)
   - IDEALPRO MIDPOINT data, free, institutional quality
   - yfinance forex has gaps and stale quotes
   - Falls back to yfinance if IB Gateway is not running

If both fail: wait 60-120s (randomized), retry up to 50 times
```

### Why yfinance for stocks?

yfinance returns two price columns with `auto_adjust=False`:
- `Close` -- unadjusted close (accounts for splits but NOT dividends)
- `Adj Close` -- fully adjusted close (accounts for both splits AND dividends)

IBKR only provides unadjusted prices. Since all backtesting models are trained on `Adj Close`, using IBKR for stocks would produce incorrect historical prices for any stock with past dividends. yfinance is the correct primary source for stock OHLCV.

### Why IBKR for forex?

yfinance forex data comes from Yahoo Finance's internal feed (sourced from ICE/Refinitiv) and has known gaps, stale quotes, and missing weekend data. IBKR sources forex from IDEALPRO (their ECN aggregating 17+ interbank dealers) -- institutional quality, and free (no subscription needed).

### IBKR market data subscriptions

IBKR subscriptions are only needed if you use `--source ibkr` to force IBKR for stocks. For the default `--source auto` workflow, **no subscriptions are needed for stocks** (yfinance handles them). Forex is always free on IBKR.

If you want IBKR as a fallback for stocks when yfinance fails, subscribe in **IBKR Account Management > Settings > Market Data Subscriptions**:

| Bundle | Covers | Cost |
|---|---|---|
| US Securities Snapshot & Futures Value Bundle | US stocks/ETFs (SPY, NVDA, etc.) | ~$10/mo |
| Tokyo Stock Exchange (Non-Pro) | Japanese stocks (.T) | ~$1-6/mo |
| Hong Kong SEHK (L1) | HK stocks (.HK) | ~$4.50/mo |
| London Stock Exchange (L1) | UK stocks (.L) | ~$1-6/mo |
| Euronext Basic | French/Dutch stocks (.PA, .AS) | ~$1-6/mo |
| Deutsche Boerse Xetra (Non-Pro) | German stocks (.DE) | ~$1-5/mo |
| Forex | All forex pairs | **Free** |

Without stock subscriptions, IBKR fallback for stocks will fail silently and the updater will retry with yfinance. The system works fine without any subscriptions.

## Ticker Universe

The list of tickers the store tracks is defined in:

```
data/market_data/ticker_universe.json
```

This is a JSON file mapping yfinance ticker to output column name:

```json
{
  "_description": "...",
  "_updated": "2026-02-25",
  "tickers": {
    "SPY": "SPY",
    "RACE": "Ferrari",
    "9988.HK": "ALIBABA",
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY"
  }
}
```

- **Key** (left of `:`): yfinance ticker used for downloading
- **Value** (right of `:`): output column name in portfolio CSVs

### Adding/removing tickers

Edit `ticker_universe.json` directly. Then run the updater to fetch data for new tickers:

```bash
# After editing the JSON:
python -m algos.common.update_market_data --tickers NEW_TICKER_1 NEW_TICKER_2
```

If `ticker_universe.json` doesn't exist (fresh clone), the updater falls back to parsing `yfinance_downloader_v5.py`'s `current_tickers_map`.

## Parquet Store Layout

```
data/market_data/
├── SPY.parquet             # ~1260 rows x 8 columns (~80KB)
├── NVDA.parquet
├── 8058.T.parquet
├── EURUSD.parquet          # forex: =X suffix stripped from filename
├── ...                     # ~1304 files total
├── _metadata.json          # per-ticker: first_date, last_date, rows, last_updated, source
├── ticker_universe.json    # what tickers to track
├── updater.log             # updater log (append mode)
└── .updater.lock           # PID lock (prevents concurrent updater instances)
```

### Parquet schema per file

Each `.parquet` file has a DatetimeIndex (`date`) and these columns:

| Column | Type | Description |
|---|---|---|
| `open` | float64 | Unadjusted open |
| `high` | float64 | Unadjusted high |
| `low` | float64 | Unadjusted low |
| `close` | float64 | Unadjusted close |
| `volume` | float64 | Volume |
| `adj_close` | float64 | Split/dividend adjusted close |
| `source` | string | `"ibkr"` or `"yfinance"` |

Both `close` (unadjusted) and `adj_close` are stored. Consumers pick the one they need:
- `data_cache.py` (backtesting, `auto_adjust=True`): reads `adj_close` as `Close`
- `portimization.py` (portfolio optimization): reads `adj_close` for returns
- `data_manager.py` (live trading warmup): reads `adj_close` to match model training

## How Consumers Read Data

All consumers try the parquet store first, then fall back to their original data source (yfinance or CSV). If the parquet store doesn't exist or a ticker isn't in it, existing behavior is preserved.

### Backtesting (`run_backtest_optimized.py`)

```
OptimizedBacktester.load_and_preprocess_data()
  -> OptimizedDataLoader.load_data()    [algos/common/data_cache.py]
       -> MarketDataStore.get_ohlcv()   [tries parquet first]
       -> yfinance download             [fallback if ticker not in store]
```

No changes needed to backtest commands:

```bash
python algos/backtest_code/run_backtest_optimized.py \
    --model_name svm_optimized --ticker SPY --lookback_days 1260
```

### WFOV validation (`wfov_runner.py`)

Inherits from backtest path above. No changes needed:

```bash
python -m algos.wfov.wfov_runner \
    --mode monte_carlo --model_name svm_optimized --ticker SPY \
    --iterations 100 --seed 42
```

### Portfolio optimization (`portimization.py`)

Accepts both CSV and parquet file paths:

```bash
# Using parquet export from the store:
python -m algos.common.update_market_data --export-csv data/portfolio_prices.csv
python algos/backtest_code/portimization.py --mode miqp --budget 50000

# Or pass a parquet file directly (if you build one):
python algos/backtest_code/portimization.py --data_path data/prices.parquet
```

Scripts that import from `portimization.py` inherit parquet support automatically:
- `portfolio_exploration_global.py`
- `validate_portfolio_oos.py`
- `validate_fixed_portfolio.py`

### Live trading warmup (`data_manager.py`)

`fetch_and_store_historical_data()` tries the parquet store for historical bars before falling back to yfinance. If parquet data exists and is fresh (<48 hours old), it's used for warmup without any network call. The current day's bar is still fetched live.

### Legacy backtesting (`data_loader.py`)

`load_and_preprocess_data()` checks the parquet store as Priority 2 (after user-provided file, before CSV cache and yfinance download).

## Suggested Cron Schedule

```bash
# Daily incremental update (weekday mornings, after markets close)
0 6 * * 1-5  cd /path/to/project && conda run -n <your-env> python -m algos.common.update_market_data

# Weekly full-refresh on Saturday (catch adj_close corrections)
0 10 * * 6   cd /path/to/project && conda run -n <your-env> python -m algos.common.update_market_data --full-refresh 5
```

## Troubleshooting

### "No tickers to update"

The `ticker_universe.json` file is missing or empty. Either:
- Create it: `python -m algos.common.update_market_data --init` (triggers fallback parser)
- Or regenerate it from `yfinance_downloader_v5.py` (see source code for `_extract_tickers_from_downloader`)

### "Another updater instance is running"

A previous run crashed or was killed without releasing the PID lock. Delete the stale lockfile:

```bash
rm data/market_data/.updater.lock
```

### "IB Gateway not detected"

IB Gateway / TWS is not running or not on port 4001. The updater degrades to yfinance automatically. To use IBKR:
1. Start IB Gateway or TWS
2. Enable API connections (Configure > API > Settings > Enable ActiveX and Socket Clients)
3. Set socket port to 4001
4. Re-run the updater

### Stale data warnings

Consumers log a warning when parquet data is older than 48 hours:

```
WARNING: SPY data is 72.3h old (limit: 48h). Consider running update_market_data.py
```

Run the daily updater to fix.

### Checking data provenance

Each parquet file has a `source` column tracking where each row came from:

```python
from algos.common.market_data_store import MarketDataStore
store = MarketDataStore()
raw = store.get_ohlcv_raw("EURUSD")
print(raw["source"].value_counts())
# ibkr        297
```

## File Reference

| File | Purpose |
|---|---|
| `algos/common/market_data_store.py` | Parquet read/write accessor (MarketDataStore class) |
| `algos/common/ibkr_downloader.py` | IBKR historical data client via ibapi (clientId=10) |
| `algos/common/update_market_data.py` | DB updater CLI (IBKR primary, yfinance fallback) |
| `algos/common/yf_downloader.py` | Resilient yfinance wrapper (retry, backoff, cache clearing) |
| `data/market_data/ticker_universe.json` | Ticker list config (single source of truth) |
| `data/market_data/_metadata.json` | Per-ticker update timestamps (auto-managed) |
| `data/market_data/*.parquet` | Per-ticker OHLCV data files |
| `data/market_data/updater.log` | Updater log (append mode) |
