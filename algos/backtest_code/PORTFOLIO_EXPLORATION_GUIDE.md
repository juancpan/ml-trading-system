# Portfolio Exploration Workflow - Usage Guide

## Overview

`portfolio_exploration_global.py` is a **4-stage intelligent asset selection pipeline** that optimizes 300+ asset universes down to 20-30 final portfolio holdings in **~2 seconds** (vs 30+ minutes with old approach).

**Speed improvement: 1,000x faster** than traditional Monte Carlo optimization.

---

## Quick Start

### **Example 1: Basic Usage (Fast Mode)**

```bash
cd algos/backtest_code

python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01
```

**What happens:**
```
350 assets → Stage 1 (screening) → 100 assets
                ↓
           Stage 2 (clustering) → 40 assets
                ↓
           Stage 3 (optimization) → 3 portfolios
                ↓
           Stage 4 (analysis) → Recommendations
```

**Execution time:** ~1.6 seconds

**Output:**
```
Portfolio          Positions  Sharpe  Max Weight  Method
─────────────────────────────────────────────────────────
max_sharpe_fast    21        3.87    18.59%      SLSQP
min_volatility     20        3.18    19.00%      SLSQP
hrp                39        3.37    13.87%      HRP

✅ Position count (21) is manageable - portfolio ready
⚠️  High concentration (18.6% max) - consider max weight cap
```

**Files created:**
- `logs/portfolio_exploration_20260106_031658.log` (detailed execution log)
- `logs/exploration_results_20260106_031658.txt` (summary table)

---

## Example Scenarios

### **Scenario 1: Quick Exploration (Default)**

**Goal:** Understand what unconstrained optimization suggests for your 350-asset universe.

```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01
```

**Recommended next steps:**

**If output shows:**
```
max_sharpe_fast: 21 positions, max weight 18.6%
```

**Then:**
- **21 positions OK?** → Use max_sharpe_fast as-is
- **18.6% too concentrated?** → See Scenario 2 (add constraints)
- **Want exactly 25 positions?** → See Scenario 3 (cardinality)

---

### **Scenario 2: More Aggressive Screening**

**Goal:** Reduce asset pool to only strong performers (Sharpe > 0.5).

```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --min-sharpe 0.5          # Only keep assets with Sharpe > 0.5
    --stage1-top-n 60         # Less assets in Stage 1
    --stage2-target 30        # Less assets in Stage 2
```

**Use when:**
- You only want proven winners
- Your 350-asset universe has many mediocre performers
- You prefer concentrated, high-conviction portfolio

**Expected output:**
- Fewer but higher-quality positions
- Higher average Sharpe per position
- More concentrated (higher max weight)

---

### **Scenario 3: More Diversification**

**Goal:** Force broader diversification across sectors/strategies.

```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --stage1-top-n 120        # Keep more candidates
    --stage2-target 50        # More diverse assets
    --max-per-cluster 5       # Limit concentration per cluster
```

**Use when:**
- You want broad sector exposure
- Risk management prefers diversification
- You suspect correlation regime changes

**Expected output:**
- More positions (30-40)
- Lower concentration (max weight ~8-12%)
- HRP portfolio performs better relative to Max Sharpe

---

### **Scenario 4: Global Search Mode (Slower, More Confident)**

**Goal:** Find true global optimum with differential evolution.

```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --use-global-search       # Enable differential evolution
```

**Execution time:** ~7-10 seconds (vs ~1.6 seconds fast mode)

**What you get:**
- `max_sharpe_fast` (SLSQP result, 1.6s)
- `max_sharpe_global` (DE result, additional 6s)
- Comparison showing if global search found better solution

**Expected log output:**
```
  2. Maximum Sharpe Ratio (Differential Evolution - global search)
     ✅ Sharpe=3.89, Positions=20, Iters=87
     🎯 Global search found better solution (+0.02 Sharpe)
```

**Use when:**
- You suspect SLSQP trapped in local optimum
- Final portfolio selection (worth extra 6 seconds)
- You want confidence in optimality

---

### **Scenario 5: Different Time Windows**

**Goal:** Test portfolio stability across different market regimes.

```bash
# Bull market (2023-2024)
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2023-01-01_2025-06-01_1d.csv \
    --start 2023-01-01 \
    --end 2024-12-31

# Full cycle (2020-2025, includes COVID crash)
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-11-01_2025-11-01_1d.csv \
    --start 2020-11-01 \
    --end 2025-11-01

# Recent only (2024-2025)
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2024-01-01_2025-06-01_1d.csv \
    --start 2024-01-01 \
    --end 2025-06-01
```

**Compare results:**
- Do same assets appear across regimes? (robust candidates)
- Does max weight change significantly? (concentration risk)
- Does HRP beat Max Sharpe in volatile periods? (robustness indicator)

---

## Understanding the Output

### **Comparison Table**

```
Portfolio          Method              Positions  Sharpe  Max Weight  HHI
─────────────────────────────────────────────────────────────────────────
max_sharpe_fast    SLSQP (local)       21        3.87    18.59%      0.0875
min_volatility     SLSQP (convex)      20        3.18    19.00%      0.1055
hrp                HRP (deterministic)  39        3.37    13.87%      0.0547
```

**Metrics explained:**

| Metric | What It Means | Decision Guide |
|--------|---------------|----------------|
| **Positions** | # of assets in portfolio | 20-30 = ideal, >40 = too many, <15 = concentrated |
| **Sharpe** | Risk-adjusted return | >2.0 = excellent, 1.0-2.0 = good, <1.0 = review |
| **Max Weight** | Largest single position | <10% = conservative, 10-15% = moderate, >15% = aggressive |
| **HHI** | Concentration index | <0.05 = diversified, 0.05-0.10 = moderate, >0.10 = concentrated |

### **Top 10 Holdings**

```
MAX SHARPE FAST:
  1. IAU                  18.59%  (Gold ETF)
  2. 中国移动                9.99%  (China Mobile)
  3. Fairfax_Financial     9.47%  (Insurance)
  ...
```

**How to interpret:**
- **Sector concentration:** Top 3 = 37.5% → moderate risk
- **Asset types:** Gold + Defensive stocks → conservative tilt
- **Geographic:** US + China + Canada → global diversification

### **Asset Overlap (Consensus Picks)**

```
Assets in ALL portfolios (12):
  IAU                 max: 18.59% | min_vol: 19.00% | hrp: 13.87%
  Fairfax_Financial   max:  9.47% | min_vol:  7.12% | hrp:  8.98%
  ...
```

**What this tells you:**
- These 12 assets are **robust** (selected by all methods)
- Weight variation shows how each method values them
- **High confidence picks** for core portfolio

---

## Decision Framework

### **Step 1: Review Position Count**

```
Max Sharpe wants: 21 positions
```

**Decision:**
- ✅ **If 21 is acceptable:** Use portfolio as-is → Proceed to Step 2
- ⚠️ **If want exactly 25:** Re-run with cardinality constraint (future feature)
- ⚠️ **If want fewer (<15):** Increase `--min-sharpe` to filter more aggressively

### **Step 2: Review Concentration**

```
Max Weight: 18.59% in IAU
```

**Decision:**
- ✅ **If <15% is OK:** Use as-is
- ⚠️ **If >15% is too risky:** Need to add max_weight constraint (future feature)
- 💡 **Alternative:** Use HRP instead (max weight 13.87%, more balanced)

### **Step 3: Choose Portfolio**

| If You Prioritize... | Choose Portfolio | Why |
|---------------------|------------------|-----|
| **Maximum returns** | max_sharpe_fast | Highest Sharpe (3.87) |
| **Lowest risk** | min_volatility | Lowest volatility (0.0829) |
| **Robustness** | hrp | No mean estimation error, proven out-of-sample |
| **Balance** | max_sharpe_fast | Best risk-adjusted return |

### **Step 4: Validate with Consensus**

Review the "Assets in ALL portfolios" section. These 12 assets appear in every portfolio:
- **High confidence:** IAU, Fairfax_Financial, WMT, etc.
- **Core positions:** Allocate at least 50% to these consensus picks
- **Satellite positions:** Remaining 50% from portfolio-specific picks

---

## Common Workflows

### **Workflow A: Weekly Portfolio Rebalancing**

```bash
# Monday morning: Run exploration
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01

# Review output
cat ../../logs/exploration_results_*.txt | tail -20

# Decision: Use max_sharpe_fast (21 positions, Sharpe 3.87)

# Extract weights for live trading
grep -A 21 "MAX SHARPE FAST:" ../../logs/portfolio_exploration_*.log | tail -21

# Update execution/config.py with the 21 tickers and weights
```

---

### **Workflow B: Regime Comparison**

```bash
# Test 1: Bull market period
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2023-01-01_2025-06-01_1d.csv \
    --start 2023-01-01 \
    --end 2024-12-31 \
    > results_bull.txt

# Test 2: Volatile period (COVID recovery)
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-11-01_2025-11-01_1d.csv \
    --start 2020-11-01 \
    --end 2022-12-31 \
    > results_volatile.txt

# Compare
diff results_bull.txt results_volatile.txt
```

**Look for:**
- Assets that appear in both → robust picks
- Assets only in bull market → regime-dependent (risky)
- HRP vs Max Sharpe performance difference → robustness indicator

---

### **Workflow C: Progressive Refinement**

**Round 1: Broad exploration**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --min-sharpe -0.5         # Very permissive

# Output: 21 positions, top weight 18.6%
# Decision: Too concentrated
```

**Round 2: Increase minimum quality**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --min-sharpe 0.8          # Only strong performers
    --stage1-top-n 50         # Fewer candidates

# Output: 15 positions, top weight 22%
# Decision: Still concentrated, but better quality
```

**Round 3: Use HRP for diversification**
```bash
# Review HRP result from Round 1
grep -A 40 "HRP:" ../../logs/portfolio_exploration_*.log | tail -40

# HRP gives: 39 positions, max weight 13.87%
# Decision: More balanced, use HRP for live trading
```

---

## Parameter Tuning Guide

### **Stage 1 Parameters**

| Parameter | Default | Effect | When to Change |
|-----------|---------|--------|----------------|
| `--stage1-top-n` | 100 | Assets after screening | Increase (120-150) for more diversity |
| `--min-sharpe` | -0.5 | Minimum Sharpe to keep | Increase (0.3-0.8) for quality over quantity |
| `--max-correlation` | 0.95 | Remove duplicates | Decrease (0.85-0.90) if many similar ETFs |
| `--min-trading-days` | 500 | Data quality threshold | Decrease (300-400) for shorter histories |

### **Stage 2 Parameters**

| Parameter | Default | Effect | When to Change |
|-----------|---------|--------|----------------|
| `--stage2-target` | 40 | Assets after clustering | Increase (50-60) for more sector exposure |
| `--n-clusters` | auto | # of clusters | Set to 8-12 if auto-selection is unstable |
| `--min-per-cluster` | 1 | Force diversity | Increase (2-3) to ensure balanced sectors |
| `--max-per-cluster` | 8 | Prevent overconcentration | Decrease (5-6) if one sector dominates |

### **Stage 3 Parameters**

| Parameter | Default | Effect | When to Change |
|-----------|---------|--------|----------------|
| `--risk-free-rate` | 0.01 | Sharpe calculation baseline | 0.04-0.05 in high-rate environments |
| `--use-global-search` | False | Enable differential evolution | True for final portfolio selection |

---

## Reading the Logs

### **Stage 1 Output**
```
Stage 1 Complete: 350 → 100 assets
  Removed: 0 (quality) + 21 (Sharpe) + 13 (correlation)
  Score range: [0.298, 2.996]
  Sharpe range: [0.659, 2.243]
```

**Interpretation:**
- 21 assets had Sharpe < -0.5 (extreme losers)
- 13 assets were duplicates (e.g., SPY vs VOO)
- Top asset scored 2.996 (composite), Sharpe 2.243 (excellent)

---

### **Stage 2 Output**
```
Auto-selected 6 clusters (silhouette score: 0.294)

Cluster composition:
  Cluster 0: 15 assets (e.g., Tech stocks)
  Cluster 1: 30 assets (e.g., Large cap value)
  Cluster 2: 15 assets (e.g., Commodities)
  ...

  Selected 7 from Cluster 0
  Selected 7 from Cluster 1
  ...

Stage 2 Complete: 100 → 40 assets
```

**Interpretation:**
- K-means found 6 natural groupings in correlation space
- Each cluster = sector/strategy/region proxy
- Balanced selection ensures you're not 100% tech

---

### **Stage 3 Output**
```
Ledoit-Wolf shrinkage intensity: 0.0522
Covariance condition number: 74.49

1. Max Sharpe (SLSQP): Sharpe=3.87, Positions=21, Max=18.59%
2. Min Vol (SLSQP): Sharpe=3.18, Positions=20, Max=19.00%
3. HRP: Sharpe=3.37, Positions=39, Max=13.87%
```

**Interpretation:**
- **Shrinkage 0.05:** Low shrinkage = high-quality data (closer to sample cov)
- **Condition number 74:** Moderate ill-conditioning (Ledoit-Wolf stabilized it)
- **Max Sharpe > Min Vol Sharpe:** Rare! Usually Min Vol has lower Sharpe
- **HRP 39 positions:** More diversified than optimization methods

---

### **Stage 4 Recommendations**
```
1. Position count (21) is manageable - portfolio ready
2. High concentration (18.6% max) - consider max weight cap
3. HRP is competitive (3.37 vs 3.87 Sharpe) - consider for robustness
```

**Action items:**
- ✅ Accept 21 positions as-is
- ⚠️ Note: Top position (IAU) is 18.6% → watch for rebalancing needs
- 💡 Consider: HRP as alternative for out-of-sample robustness

---

## Advanced: Extracting Weights for Live Trading

### **Step 1: Find the log file**
```bash
ls -t ../../logs/portfolio_exploration_*.log | head -1
```

### **Step 2: Extract weights**
```bash
# For Max Sharpe portfolio
grep -A 25 "MAX SHARPE FAST:" ../../logs/portfolio_exploration_20260106_031658.log

# Output:
#   1. IAU                  18.59%
#   2. 中国移动                9.99%
#   3. Fairfax_Financial     9.47%
#   ...
```

### **Step 3: Convert to config format**

**For IBKR (execution/config.py):**
```python
TARGET_ALLOCATION = {
    'IAU': 0.1859,            # Gold ETF (US)
    '0941.HK': 0.0999,        # China Mobile (Hong Kong)
    # Note: Fairfax_Financial ticker lookup needed
    'WELL': 0.0730,
    'WMT': 0.0479,
    # ... continue for all 21 positions
}
```

**For old portimization.py format:**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01 \
    | grep -A 25 "MAX SHARPE" \
    | awk '{print "--fixed-weights", $2"="$3}' \
    | sed 's/%//' | sed 's/,//'
```

---

## Troubleshooting

### **Issue: "Too few assets after quality filter"**
```
WARNING - Too few assets after quality filter (9 < 50)
WARNING - FALLBACK: Creating equal-weight portfolio
```

**Solution:**
```bash
# Lower the minimum trading days threshold
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2024-01-01_2025-06-01_1d.csv \
    --start 2024-01-01 \
    --end 2025-06-01 \
    --min-trading-days 200    # Lower from 500
```

---

### **Issue: "Auto-selected X clusters" seems wrong**
```
INFO - Auto-selected 12 clusters (silhouette score: 0.15)
```

Silhouette < 0.2 = poor clustering quality.

**Solution:**
```bash
# Manually specify cluster count
python portfolio_exploration_global.py \
    --csv ../../data/your_data.csv \
    --start 2022-01-01 \
    --end 2025-01-01 \
    --n-clusters 8           # Fixed at 8 clusters
```

---

### **Issue: HRP has too many positions**
```
hrp: 39 positions (vs max_sharpe: 21)
```

**Explanation:** HRP naturally diversifies more broadly than mean-variance.

**Options:**
1. **Use max_sharpe instead** (fewer positions)
2. **Filter HRP weights:** Take top 25 HRP positions, ignore rest
3. **Accept 39 positions** (if your trading platform supports it)

---

### **Issue: All portfolios have low Sharpe (<1.0)**
```
max_sharpe_fast: Sharpe=0.65
```

**Possible causes:**
1. **Bear market period:** All assets underperformed
2. **High risk-free rate:** Effective Sharpe = (Return - 0.01) / Vol
3. **Poor asset universe:** All 350 assets are low-quality

**Solutions:**
```bash
# Check if risk-free rate is correct
python portfolio_exploration_global.py \
    --csv ../../data/your_data.csv \
    --start 2022-01-01 \
    --end 2025-01-01 \
    --risk-free-rate 0.04    # Update if rates changed

# Try stricter screening
python portfolio_exploration_global.py \
    --csv ../../data/your_data.csv \
    --start 2022-01-01 \
    --end 2025-01-01 \
    --min-sharpe 0.5         # Only keep good performers
```

---

## Performance Benchmarks

| Dataset | Assets | Mode | Time | Result |
|---------|--------|------|------|--------|
| Synthetic (9 assets) | 9 | Fast | 0.1s | 6 positions |
| Small real (8 assets) | 8 | Fast | 0.2s | 4 positions |
| **Medium (350 assets)** | **350** | **Fast** | **1.6s** | **21 positions** |
| Medium (350 assets) | 350 | Global | 7.6s | 21 positions (+0.02 Sharpe improvement) |

**Comparison with old method:**
- Old portimization.py with 500k MC: **30+ minutes**
- New workflow (fast mode): **1.6 seconds**
- **Speedup: 1,125x**

---

## Next Steps After Exploration

### **1. Deploy to Live Trading**

```bash
# Extract top 21 positions
LOG_FILE=$(ls -t ../../logs/portfolio_exploration_*.log | head -1)

# Get the tickers
grep -A 25 "MAX SHARPE FAST:" $LOG_FILE | awk 'NR>1 {print $2, $3}' | head -21

# Manually update execution/config.py with these tickers
```

### **2. Backtest Selected Portfolio**

```bash
# For each selected asset, run individual backtest
for ticker in IAU 0941.HK WELL WMT NVDA AVGO; do
    python run_backtest_optimized.py \
        --model_name svm_optimized \
        --ticker $ticker \
        --start 2022-12-01 \
        --end 2025-12-01
done
```

### **3. Compare with Old Method**

```bash
# Run old portimization.py for comparison
python portimization.py \
    --start 2022-12-01 \
    --end 2025-12-01 \
    --target_volatility 0.10

# Compare results: Do the weights agree?
```

---

## Best Practices

1. **Run weekly:** Portfolio composition can change with market conditions
2. **Use fast mode for exploration:** Save global search for final selection
3. **Trust the consensus:** Assets in ALL portfolios are highest confidence
4. **Monitor HHI:** If >0.10, portfolio is concentrated (add more diversity)
5. **Validate Sharpe:** Compare with SPY benchmark (Sharpe ~1.0-1.5 typical)
6. **Check logs:** Review which assets were removed and why

---

## Summary

**For most users:**
```bash
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \
    --start 2022-12-01 \
    --end 2025-12-01
```

**Review output → Pick portfolio (usually max_sharpe_fast) → Deploy to live trading**

**Execution time:** 1.6 seconds
**Result:** 20-30 positions, ready for production
