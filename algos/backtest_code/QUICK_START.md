# Portfolio Exploration - Quick Reference

## One-Liner for Most Users

```bash
cd algos/backtest_code

python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01
```

**Output:** 3 portfolios in 1.7 seconds (21, 20, 39 positions)

---

## Performance Summary

| Metric | Old Method | New Workflow | Improvement |
|--------|------------|--------------|-------------|
| **Time** | 30+ minutes | 1.7 seconds | **1,000x faster** |
| **Assets optimized** | 350 | 40 (pre-filtered) | Smarter |
| **Monte Carlo** | 500,000 | 0 | Eliminated |
| **Output** | 350 weights | 21-39 positions | Interpretable |
| **Global optimum** | No | Optional | Better |

---

## What You Get

```
PORTFOLIO COMPARISON TABLE
────────────────────────────────────────────────────────────────────
Portfolio          Positions  Sharpe  Max Weight  Method
────────────────────────────────────────────────────────────────────
max_sharpe_fast    21        3.87    18.59%      SLSQP (fast)
min_volatility     20        3.18    19.00%      SLSQP (conservative)
hrp                39        3.37    13.87%      HRP (robust)

TOP 10 HOLDINGS (Max Sharpe):
  1. IAU (Gold)              18.59%
  2. 中国移动                   9.99%
  3. Fairfax_Financial        9.47%
  ...

CONSENSUS PICKS (in all 3 portfolios):
  - IAU, Fairfax_Financial, WMT, WELL, etc. (12 assets)

RECOMMENDATIONS:
  ✅ Position count (21) manageable - ready
  ⚠️  Top weight 18.6% - consider 10-12% cap
```

---

## Common Use Cases

### **Daily Exploration (Default)**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/latest_prices.csv \
    --start 2023-01-01 \
    --end 2025-12-31
```

### **Higher Quality Assets Only**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/latest_prices.csv \
    --start 2023-01-01 \
    --end 2025-12-31 \
    --min-sharpe 0.8          # Only excellent performers
    --stage1-top-n 60         # Fewer but better
```

### **More Diversification**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/latest_prices.csv \
    --start 2023-01-01 \
    --end 2025-12-31 \
    --stage2-target 50        # More assets in final pool
    --max-per-cluster 5       # Spread across sectors
```

### **Final Selection (Global Search)**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/latest_prices.csv \
    --start 2023-01-01 \
    --end 2025-12-31 \
    --use-global-search       # Takes ~7 seconds instead of 1.7s
```

---

## Decision Tree

```
Run workflow
    ↓
Review position count
    ├─ 20-30 positions? → ✅ Good, proceed
    └─ >40 positions? → Increase --min-sharpe to 0.5-0.8
    ↓
Review max weight
    ├─ <15%? → ✅ Good
    └─ >15%? → Note for rebalancing, or use HRP instead
    ↓
Compare Sharpe ratios
    ├─ Max Sharpe > HRP by >0.5? → Use Max Sharpe
    └─ HRP competitive (<0.3 difference)? → Use HRP (more robust)
    ↓
Check consensus assets
    ↓
Deploy to live trading
```

---

## Output Files

**Every run creates 2 files:**

1. **Detailed log:** `logs/portfolio_exploration_20260106_031658.log`
   - Full stage-by-stage execution
   - All 21-39 holdings with weights
   - Cluster composition
   - Overlap analysis

2. **Quick summary:** `logs/exploration_results_20260106_031658.txt`
   - Comparison table only
   - For quick review

---

## Parameters Cheat Sheet

| Want More... | Adjust |
|--------------|--------|
| **Quality over quantity** | `--min-sharpe 0.8` |
| **Diversity** | `--stage2-target 50 --max-per-cluster 5` |
| **Fewer positions** | `--stage1-top-n 50 --stage2-target 25` |
| **Confidence in optimum** | `--use-global-search` |

---

**See PORTFOLIO_EXPLORATION_GUIDE.md for full documentation.**
