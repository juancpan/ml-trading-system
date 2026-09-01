# Model Selection Workflow - Quick Reference

**Complete control over model training, validation, and selection**

---

## 🚀 **Quick Start**

```bash
# Simplest: One ticker, uses config defaults
python scripts/model_selection_workflow.py --ticker NVDA

# Batch: All tickers from config
python scripts/model_selection_workflow.py --batch

# Quick mode: Fast screening (20 iterations)
python scripts/model_selection_workflow.py --ticker NVDA --quick
```

---

## 📋 **All Command Line Arguments**

### **Ticker Selection**

| Argument | Description | Default |
|----------|-------------|---------|
| `--ticker NVDA` | Single ticker | None (required) |
| `--batch` | All tickers from config | Uses `batch_tickers` |
| `--tickers NVDA SPY` | Specific tickers | Overrides `batch_tickers` |

### **Model Selection**

| Argument | Description | Default |
|----------|-------------|---------|
| (none) | Uses config | ticker_models → default_models |
| `--models lstm arima` | Custom models | Overrides config |
| `--preset quick` | 2 models (fast) | Uses `quick_models` |
| `--preset comprehensive` | 12+ models (thorough) | Uses `comprehensive_models` |

### **Iteration Control**

| Argument | Description | Iterations |
|----------|-------------|-----------|
| (default) | Standard | 50 |
| `--quick` | Fast screening | 20 |
| `--iterations N` | Custom | N |

### **Validation Mode**

| Argument | Description | Default |
|----------|-------------|---------|
| `--validation-mode monte_carlo` | Random sampling | monte_carlo |
| `--validation-mode walk_forward_expanding` | Growing train set | - |
| `--validation-mode walk_forward_rolling` | Sliding window | - |

### **WFOV Date Parameters**

| Argument | Description | Default |
|----------|-------------|---------|
| `--start-date 2020-01-01` | WFOV start date | 2020-01-01 |
| `--end-date 2025-12-01` | WFOV end date | Today |
| `--min-lookback 365` | Min lookback (Monte Carlo) | 365 |
| `--max-lookback 1825` | Max lookback (Monte Carlo) | 1825 |
| `--initial-train-days 1260` | Initial train (WF expanding) | None |
| `--window-size 1260` | Window size (WF rolling) | None |
| `--test-days 252` | Test period (WF modes) | None |
| `--step-days 126` | Step size (WF modes) | None |

### **Ranking & Config**

| Argument | Description | Default |
|----------|-------------|---------|
| `--profile A` | Max returns (aggressive) | A |
| `--profile B` | Risk-adjusted (balanced) | - |
| `--profile C` | Institutional (conservative) | - |
| `--seed 42` | Random seed | 42 |
| `--no-config` | Ignore config file | Uses config |
| `--config-path custom.yaml` | Custom config file | model_selection_config.yaml |

---

## 📝 **Config File: model_selection_config.yaml**

**Edit this file to set permanent defaults:**

```yaml
# Which tickers to test in --batch mode
batch_tickers:
  - NVDA
  - AVGO
  - SPY
  # Add/remove tickers here

# Default models (fallback for tickers not in ticker_models)
default_models:
  - lstm
  - svm_optimized
  - xgb_optimized
  # Add/remove default models

# Ticker-specific models (optional)
ticker_models:
  NVDA:
    - lstm
    - svm_optimized
    - ensemble_optimized

  SPY:
    - li_reg
    - arima
    - svm_optimized

# Presets
quick_models:
  - lstm
  - svm_optimized

comprehensive_models:
  - lstm
  - svm_optimized
  - xgb_optimized
  - linear_optimized
  - rf_optimized
  - arima
  # ... (12+ models)
```

---

## 💡 **Common Usage Patterns**

### **Daily Quick Check**
```bash
python scripts/model_selection_workflow.py --ticker NVDA --quick
```

### **Monthly Update (Monte Carlo)**
```bash
python scripts/model_selection_workflow.py --ticker NVDA --iterations 50
```

### **Pre-Deployment Validation (Walk-Forward)**
```bash
python scripts/model_selection_workflow.py \
    --ticker NVDA \
    --validation-mode walk_forward_expanding \
    --initial-train-days 1260 --test-days 252 --step-days 252
```

### **Batch Quick Screening**
```bash
python scripts/model_selection_workflow.py --batch --quick
```

### **Batch with Risk-Adjusted Ranking**
```bash
python scripts/model_selection_workflow.py --batch --profile B --iterations 50
```

### **Comprehensive Deep Dive**
```bash
python scripts/model_selection_workflow.py \
    --ticker NVDA \
    --preset comprehensive \
    --iterations 100 \
    --start-date 2018-01-01
```

### **Custom Dates and Models**
```bash
python scripts/model_selection_workflow.py \
    --tickers NVDA SPY \
    --models lstm svm_optimized ensemble_optimized \
    --start-date 2022-01-01 --end-date 2025-12-01 \
    --min-lookback 180 --max-lookback 730 \
    --iterations 100
```

---

## 🎯 **Priority Rules**

**Tickers:** `--tickers` → `--batch` (uses batch_tickers) → error
**Models:** `--models` → `--preset` → ticker_models → default_models → hardcoded
**Iterations:** `--iterations` → `--quick` (20) → default (50)

**Config file:** Always loaded unless `--no-config` specified

---

## 📊 **Time Estimates**

**With your current config (15 tickers):**

| Command | Tickers | Models | Iterations | Total Backtests | Time |
|---------|---------|--------|-----------|-----------------|------|
| `--batch --quick` | 15 | 4 | 20 | 1,200 | ~1 hour |
| `--batch` | 15 | 4 | 50 | 3,000 | ~2.5 hours |
| `--batch --preset comprehensive` | 15 | 12 | 50 | 9,000 | ~8-10 hours |
| `--tickers NVDA AVGO --quick` | 2 | 4 | 20 | 160 | ~10 min |
| `--ticker NVDA --preset comprehensive` | 1 | 12 | 50 | 600 | ~45 min |

---

## 📁 **Output Locations & Retrieval**

### **Where to Find Results After Workflow Completes**

**Primary output (START HERE):**
```bash
cat algos/wfov/deployment_recommendations/deployment_recommendation_NVDA_$(date +%Y%m%d).txt
```
Contains: All models ranked, best model selected, deployment commands

**Individual model details:**
```bash
algos/wfov/results/summaries/montec_{model}_{ticker}_{N}iter_*_summary.json
```

**Iteration-by-iteration data:**
```bash
algos/wfov/results/iterations/montec_{model}_{ticker}_{N}iter_*_iterations.csv
```

---

### **Retrieve Console Output After It's Gone**

**Re-generate ranking from existing results (no recomputation):**
```bash
python -m algos.wfov.model_ranker --ticker NVDA --profile A
# Takes <5 seconds, shows same ranking you saw in console
```

**Quick model comparison:**
```bash
for file in algos/wfov/results/summaries/montec_*_NVDA_*$(date +%Y%m%d)*.json; do
    model=$(basename $file | cut -d_ -f2)
    sharpe=$(jq -r '.performance_metrics.sharpe_ratio.mean' "$file")
    echo "$model: Sharpe=$sharpe"
done | sort -t: -k2 -nr
```

---

### **Automatic Output Saving (Built-In)**

**Output is automatically saved to file (no tee needed):**
```bash
# Just run normally - logging is automatic
python scripts/model_selection_workflow.py --ticker NVDA

# Output shows at start:
# 📝 Logging to: workflow_results/NVDA_20251204_153022.txt
# ... (workflow runs, output to both console and file)

# Output shows at end:
# 📝 Full output saved to: workflow_results/NVDA_20251204_153022.txt
#    View anytime: cat workflow_results/NVDA_20251204_153022.txt
```

**Benefits:**
- ✅ Automatic (no manual redirection needed)
- ✅ No subprocess buffering issues (unlike `tee`)
- ✅ Real-time updates to both console and file
- ✅ Can review later after console cleared
- ✅ Track history over time

**File naming conventions:**
- Single ticker: `workflow_results/NVDA_20251204_153022.txt`
- Batch mode: `workflow_results/batch_20251204_153022.txt`
- Custom tickers: `workflow_results/batch_NVDA_AVGO_SPY_20251204_153022.txt`

**View saved outputs:**
```bash
# List all saved workflows
ls -lt workflow_results/*.txt | head -10

# View specific run
cat workflow_results/NVDA_20251204_153022.txt

# Search for best models across all runs
grep "DEPLOY:" workflow_results/*.txt
```

---

## ✅ **TL;DR**

**Control tickers:** Edit `batch_tickers` in config or use `--tickers`
**Control models:** Edit `ticker_models`/`default_models` in config or use `--models`/`--preset`
**Control iterations:** Use `--iterations N` or `--quick`
**Retrieve results:** `cat workflow_results/NVDA_*.txt` (auto-saved) or `python -m algos.wfov.model_ranker --ticker NVDA`
**Output logging:** Automatic to `workflow_results/` directory

**Most common:**
```bash
# Edit config once
vim model_selection_config.yaml

# Run (output auto-saved to workflow_results/)
python scripts/model_selection_workflow.py --batch --quick

# View saved output later
cat workflow_results/batch_*.txt
```

**Everything is controlled via config file or command line flags. Output is automatically saved. Full flexibility.**

