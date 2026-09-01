# Out-of-Sample Portfolio Validation - Complete Guide

## Overview

Two validators for testing portfolio robustness before live deployment:

1. **`validate_portfolio_oos.py`** - Tests optimization STRATEGY (re-optimizes each window)
2. **`validate_fixed_portfolio.py`** - Tests SPECIFIC WEIGHTS (no re-optimization)

**Critical insight:** A portfolio with Sharpe 6.0 in-sample but 2.5 out-of-sample is **worthless**. These validators catch overfitting before you lose real money.

---

## Which Validator Do I Need?

### **Scenario A: "Should I reoptimize my portfolio yearly/quarterly?"**

**Use:** `validate_portfolio_oos.py`

**What it tests:**
- Re-optimizes portfolio on each training window
- Tests if the optimization **process** holds up over time
- Answers: "Is my optimization strategy robust or overfitted?"

**Example:**
```
Window 1: Optimize on 2020-2022 → Get weights W1 → Test W1 on 2023
Window 2: Optimize on 2020-2023 → Get weights W2 → Test W2 on 2024
Window 3: Optimize on 2020-2024 → Get weights W3 → Test W3 on 2025

Question: Does reoptimizing every year add value?
Result: If avg degradation < 20% → Yes, reoptimize works
```

**When to use:**
- Testing if yearly/quarterly reoptimization is worthwhile
- Comparing optimization strategies (max_sharpe vs HRP)
- Pre-deployment: Will my optimization process work going forward?

---

### **Scenario B: "I have specific weights. Will they work over time?"**

**Use:** `validate_fixed_portfolio.py`

**What it tests:**
- Takes YOUR SPECIFIC weights (one-time optimization)
- Tests those EXACT weights on multiple time periods
- Answers: "Are these weights stable over time or regime-dependent?"

**Example:**
```
One-time: Optimize on 2020-2025 → Get weights W = [IAU: 15%, NVDA: 12%, ...]

Period 1: Test W on 2021 → Sharpe 4.2
Period 2: Test W on 2022 → Sharpe 3.8
Period 3: Test W on 2023 → Sharpe 4.1
Period 4: Test W on 2024 → Sharpe 3.9
Period 5: Test W on 2025 → Sharpe 4.0

Question: Can I buy-and-hold this portfolio or does it break?
Result: Avg Sharpe 4.0, Std 0.16 (stable) → Buy-and-hold works
```

**When to use:**
- Testing buy-and-hold strategy (no rebalancing)
- Regime stability check (does it work in bull AND bear markets?)
- Validating a portfolio before deploying for long-term hold

---

## Quick Comparison Table

| Aspect | validate_portfolio_oos.py | validate_fixed_portfolio.py |
|--------|---------------------------|------------------------------|
| **Tests** | Optimization strategy | Specific weights |
| **Re-optimizes?** | ✅ Yes (each window) | ❌ No (fixed weights) |
| **Question** | "Should I reoptimize?" | "Can I buy-and-hold?" |
| **Use case** | Active rebalancing | Passive holding |
| **Measures** | IS→OOS degradation | Consistency over time |
| **Output** | Overfitting detection | Stability verdict |
| **Statistical power** | Variable (depends on mode) | Higher (more periods) |
| **Modes** | expanding, rolling, **monte_carlo** | N/A (sequential only) |

---

## Validation Modes Comparison

### **validate_portfolio_oos.py has 3 modes:**

| Mode | Windows | Statistical Power | Use Case |
|------|---------|-------------------|----------|
| **expanding** | 2-4 | ❌ LOW | Deployment: "Will it work going forward?" |
| **rolling** | 3-5 | ⚠️ LOW-MEDIUM | Regime: "Works in all periods?" |
| **monte_carlo** | 50-200 | ✅ HIGH | Validation: "Is strategy fundamentally sound?" |

**Key insight:** Monte Carlo trades realism for statistical confidence.

---

## VALIDATOR 1: validate_portfolio_oos.py (Strategy Testing)

### Purpose

Tests whether your **optimization strategy** (the process of finding optimal weights) is robust or overfitted.

**Key difference:** Re-optimizes portfolio on EVERY window (tests the optimization process itself).

---

## Why This Matters

### **The Overfitting Problem**

**What happens without OOS validation:**

```
You optimize on 2020-2025 data:
  Portfolio: 15% IAU, 12% NVDA, 10% PLTR, ...
  In-Sample Sharpe: 6.08 (EXCELLENT!)

You deploy to live trading in 2026:
  Out-of-Sample Sharpe: 2.30 (DISASTER!)
  Degradation: 62% (severe overfitting)

Lost money: Portfolio optimized to noise, not signal
```

**What happens with OOS validation:**

```
Walk-Forward Test (2020-2025):
  Window 1: Train 2020-2022 → Test 2023 → OOS Sharpe 5.80
  Window 2: Train 2020-2023 → Test 2024 → OOS Sharpe 5.95
  Window 3: Train 2020-2024 → Test 2025 → OOS Sharpe 5.70

  Avg IS: 6.05
  Avg OOS: 5.82
  Degradation: 3.8% (EXCELLENT - robust!)

Deploy with confidence: Strategy holds up on unseen data
```

---

## Quick Start

### **Basic Validation (Recommended)**

```bash
cd algos/backtest_code

python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 \
    --end 2026-01-01 \
    --portfolio max_sharpe \
    --train-years 2 \
    --test-months 12
```

**What it does:**

```
Window 1: Train 2020-2022 → Test 2023
Window 2: Train 2020-2023 → Test 2024 (expanding window)
Window 3: Train 2020-2024 → Test 2025

Measures: IS vs OOS Sharpe degradation
Output: Verdict (EXCELLENT / ACCEPTABLE / CAUTION / REJECT)
```

**Execution time:** ~5-10 seconds (3 windows × 2s each)

---

## Walk-Forward Modes

### **Mode 1: Expanding Window (Default, Recommended)**

**What it does:**

- Training window **GROWS** over time
- Test window size **FIXED**
- Simulates real-world scenario (you have more data each year)

**Example:**

```
Window 1: Train 2020-2022 (2y) → Test 2023 (1y)
Window 2: Train 2020-2023 (3y) → Test 2024 (1y) [EXPANDED]
Window 3: Train 2020-2024 (4y) → Test 2025 (1y) [EXPANDED]
```

**When to use:**

- Standard validation (most common)
- Simulates live trading (you reoptimize yearly with all history)
- Tests if more data improves or degrades performance

**Command:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode expanding \
    --train-years 2 --test-months 12
```

---

### **Mode 2: Rolling Window (Regime Detection)**

**What it does:**

- Both train and test windows **SLIDE** forward
- Train window size **FIXED**
- Detects regime-specific performance

**Example:**

```
Window 1: Train 2020-2022 (2y) → Test 2023 (1y)
Window 2: Train 2021-2023 (2y) → Test 2024 (1y) [SLID]
Window 3: Train 2022-2024 (2y) → Test 2025 (1y) [SLID]
```

**When to use:**

- Detect if strategy only works in certain regimes
- Test parameter stability across different markets
- Shorter history available (can't expand indefinitely)

**Command:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode rolling \
    --train-years 2 --test-months 12
```

**Interpretation:**

```
If results vary widely:
  Window 1 (bull market):   OOS Sharpe 4.5
  Window 2 (bear market):   OOS Sharpe 1.2
  Window 3 (recovery):      OOS Sharpe 4.0

  → Strategy is regime-dependent (HIGH RISK!)
  → Consider regime-switching strategy instead
```

---

## Portfolio Options

### **Option 1: Test Single Portfolio**

**max_sharpe:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio max_sharpe
```

**hrp:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio hrp
```

**min_volatility:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio min_volatility
```

---

### **Option 2: Compare Portfolios (Recommended)**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both
```

**Output:**

```
PORTFOLIO COMPARISON (Out-of-Sample)
Portfolio     Windows  Avg OOS Sharpe  OOS Std  Avg Degrad %  Verdict
max_sharpe    3        3.200           0.850    28.9          ⚠️  CAUTION
hrp           3        3.450           0.420    18.2          ✅ EXCELLENT

RECOMMENDATION:
  Best OOS Sharpe:    hrp
  Most Stable:        hrp
  Lowest Degradation: hrp

  ✅ HRP WINS on all 3 criteria - STRONGLY RECOMMENDED
```

**Decision made for you:** Use HRP for live trading.

---

## Time Window Configuration

### **Conservative (3y train, 1y test)**

**More training data = less overfitting**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 3 --test-months 12
```

**Trade-offs:**

- ✅ More robust estimates (3 years of training)
- ✅ Lower degradation (less overfitting)
- ❌ Fewer windows (2 instead of 3)
- ❌ Less statistical power

**When to use:** Final pre-deployment validation

---

### **Standard (2y train, 1y test)**

**Balanced approach**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 2 --test-months 12
```

**Trade-offs:**

- ✅ Good balance (2 years covers most regimes)
- ✅ Multiple windows (3-4)
- ✅ Realistic (mimics yearly rebalancing)

**When to use:** Default for most cases

---

### **Aggressive (1y train, 6m test)**

**High-frequency validation**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 1 --test-months 6
```

**Trade-offs:**

- ✅ Many windows (10-12)
- ✅ High statistical power
- ❌ More overfitting (only 1 year training)
- ❌ Noisier estimates

**When to use:** Quarterly rebalancing strategies

---

### **Custom Configurations**

**Monthly rebalancing:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 1 --test-months 1
```

**Long-term hold:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2015-01-01 --end 2026-01-01 \
    --train-years 5 --test-months 12
```

---

## Interpreting Results

### **Degradation Thresholds**

```
Degradation = (IS Sharpe - OOS Sharpe) / IS Sharpe
```

| Degradation | Verdict | Interpretation | Action |
|-------------|---------|----------------|--------|
| **< 0%** | ✅ EXCELLENT | OOS ≥ IS (getting better!) | Deploy immediately |
| **0-10%** | ✅ EXCELLENT | Minimal overfitting | Deploy with confidence |
| **10-20%** | ℹ️ ACCEPTABLE | Moderate degradation | Deploy, monitor closely |
| **20-30%** | ⚠️ CAUTION | Significant overfitting | Review before deploying |
| **> 30%** | 🚨 REJECT | Severe overfitting | DO NOT deploy |

---

### **Stability Metrics**

**OOS Sharpe Std Dev:**

| Std Dev | Stability | Interpretation |
|---------|-----------|----------------|
| < 0.3 | Very Stable | Consistent across regimes |
| 0.3-0.5 | Stable | Some variance (acceptable) |
| 0.5-1.0 | Unstable | High regime dependence |
| > 1.0 | Very Unstable | Strategy breaks in some regimes |

**Example:**

```
max_sharpe: OOS Std 0.85 (unstable)
  Window 1: 4.5
  Window 2: 1.8  ← Crash in bear market
  Window 3: 4.2

hrp: OOS Std 0.32 (stable)
  Window 1: 3.8
  Window 2: 3.4  ← Still works in bear market
  Window 3: 3.6

→ HRP is more robust despite lower avg Sharpe
```

---

### **Asset Coverage**

**Measures:** How many trained assets are available in test period?

```
Coverage = Assets Available in Test / Assets in Trained Portfolio
```

| Coverage | Interpretation | Risk |
|----------|----------------|------|
| > 95% | Excellent | Low - all assets tradeable |
| 80-95% | Good | Medium - some delisting/missing |
| 60-80% | Poor | High - significant attrition |
| < 60% | Critical | Very High - portfolio invalid |

**Example:**

```
Trained on 2020-2022: Portfolio has 30 assets
Test on 2025: Only 25 assets still exist (5 delisted)
Coverage: 25/30 = 83%

Interpretation: Good (some turnover expected)
Action: Monitor asset availability
```

---

## Sample Outputs

### **Example 1: Excellent Portfolio (Deploy)**

```
MAX_SHARPE:
  Windows Tested:      3
  Avg In-Sample:       4.520 (Sharpe)
  Avg Out-of-Sample:   4.180 (Sharpe)
  OOS Std Dev:         0.280 (stable)
  Avg Degradation:     7.5%
  Range:               5.2% to 9.8%
  Asset Coverage:      98.2%
  Verdict: ✅ EXCELLENT - Low degradation (<10%)

  Window-by-Window:
    W1: IS=4.50 OOS=4.25 (+5.6%)
    W2: IS=4.60 OOS=4.20 (+8.7%)
    W3: IS=4.45 OOS=4.10 (+7.9%)

DECISION: ✅ Deploy to live trading
```

---

### **Example 2: Overfitted Portfolio (Reject)**

```
MAX_SHARPE:
  Windows Tested:      3
  Avg In-Sample:       6.250 (Sharpe)
  Avg Out-of-Sample:   2.850 (Sharpe)
  OOS Std Dev:         1.120 (UNSTABLE)
  Avg Degradation:     54.4%
  Range:               38.2% to 72.1%
  Asset Coverage:      95.5%
  Verdict: 🚨 REJECT - Severe overfitting (>30% avg degradation)

  Window-by-Window:
    W1: IS=6.10 OOS=3.77 (+38.2%)
    W2: IS=6.45 OOS=1.80 (+72.1%)  ← CRASH
    W3: IS=6.20 OOS=2.98 (+51.9%)

DECISION: ❌ DO NOT deploy - will lose money live
```

---

### **Example 3: max_sharpe vs HRP Comparison**

```
PORTFOLIO COMPARISON (Out-of-Sample Performance)
Portfolio     Windows  Avg OOS Sharpe  OOS Std  Avg Degrad %  Verdict
max_sharpe    3        3.200           0.850    28.9          ⚠️  CAUTION
hrp           3        3.650           0.320    12.4          ✅ EXCELLENT

RECOMMENDATION:
  Best OOS Sharpe:       hrp (3.65 vs 3.20)
  Most Stable:           hrp (std 0.32 vs 0.85)
  Lowest Degradation:    hrp (12.4% vs 28.9%)

  ✅ HRP WINS on all 3 criteria - STRONGLY RECOMMENDED FOR LIVE TRADING

DECISION: Use HRP instead of max_sharpe
  - Lower in-sample Sharpe (4.2 vs 6.1)
  - But BETTER out-of-sample (3.65 vs 3.20)
  - More stable (0.32 vs 0.85)
```

---

## Complete Use Cases

### **Use Case 1: Pre-Deployment Check**

**Scenario:** You optimized a portfolio, want to deploy to live trading.

```bash
# Step 1: Optimize in-sample
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --min-sharpe 0.5 --stage2-target 40

# Output: max_sharpe Sharpe 5.82 (in-sample)

# Step 2: Validate out-of-sample
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio max_sharpe \
    --train-years 2 --test-months 12

# Output: Avg OOS Sharpe 5.20 (-10.7% degradation)
# Verdict: ✅ EXCELLENT

# Step 3: Deploy
# Copy weights to execution/config.py
```

---

### **Use Case 2: Choose Between max_sharpe vs HRP**

**Scenario:** Both portfolios look good in-sample, which to deploy?

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both \
    --train-years 2 --test-months 12
```

**Possible outcomes:**

**Outcome A: max_sharpe wins**

```
max_sharpe: OOS 5.20, Degradation 10.7%
hrp:        OOS 4.80, Degradation 8.2%

→ max_sharpe has better OOS (use it)
```

**Outcome B: HRP wins (common)**

```
max_sharpe: OOS 3.20, Degradation 45.0%
hrp:        OOS 4.10, Degradation 12.5%

→ HRP has better OOS despite lower IS (use HRP)
```

**Outcome C: Tie (use blend)**

```
max_sharpe: OOS 4.50, Degradation 15%
hrp:        OOS 4.48, Degradation 13%

→ Similar performance (use 50/50 blend or pick max_sharpe)
```

---

### **Use Case 3: Detect Regime Dependence**

**Scenario:** Check if strategy breaks in bear markets.

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2019-01-01_2026-01-01_1d.csv \
    --start 2019-01-01 --end 2026-01-01 \
    --mode rolling \
    --train-years 2 --test-months 12
```

**Example output:**

```
Window-by-Window:
  W1: Train 2019-2021 → Test 2022 (bear)  → OOS 1.85  ← CRASH
  W2: Train 2020-2022 → Test 2023 (bull)  → OOS 4.50  ← WORKS
  W3: Train 2021-2023 → Test 2024 (mixed) → OOS 3.20
  W4: Train 2022-2024 → Test 2025 (bull)  → OOS 4.30

Avg OOS: 3.46
OOS Std: 1.15 (UNSTABLE)

Interpretation: Strategy underperforms in bear markets
Action: Add defensive assets or switch to HRP
```

---

### **Use Case 4: Quarterly Rebalancing Test**

**Scenario:** You rebalance portfolio every 3 months, validate this frequency.

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 1 --test-months 3 \
    --portfolio both
```

**Why quarterly:**

- Tests high-frequency rebalancing
- More windows (16-20) = better statistics
- Reveals transaction cost impact

**Expected results:**

```
max_sharpe: 18 windows, OOS 3.80 (works)
hrp:        18 windows, OOS 3.95 (better)

But: High turnover → transaction costs may erode gains
Action: Test with 6m or 12m rebalancing
```

---

### **Use Case 5: Conservative Long-Term Validation**

**Scenario:** Institutional investor, wants 5-year lookback.

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2015-01-01_2026-01-01_1d.csv \
    --start 2015-01-01 --end 2026-01-01 \
    --train-years 5 --test-months 12
```

**Why 5-year:**

- Captures full market cycle (bull + bear)
- Institutional-grade validation
- Very low overfitting risk

**Expected:**

- Fewer windows (2-3)
- Very robust estimates
- Lower in-sample Sharpe (more data = harder to fit)
- Better OOS transfer (captures cycle)

---

## Parameter Tuning Guide

### **Train Window Size**

| train-years | Pros | Cons | When to Use |
|-------------|------|------|-------------|
| **1** | Many windows, fast | High overfitting | Quarterly rebalancing |
| **2** | Balanced | Moderate overfitting | **Standard (recommended)** |
| **3** | Low overfitting | Fewer windows | Conservative validation |
| **5** | Very robust | Few windows (2-3) | Institutional |

---

### **Test Window Size**

| test-months | Pros | Cons | When to Use |
|-------------|------|------|-------------|
| **3** | Many windows | Noisy estimates | Quarterly rebalancing |
| **6** | More windows | Moderate noise | Semi-annual rebalancing |
| **12** | Robust estimates | Fewer windows | **Standard (recommended)** |
| **24** | Very robust | Very few windows | Long-term hold |

---

### **Stage 2 Target (From portfolio_exploration_global.py)**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --stage2-target 40   # Test with 40 assets in Stage 2
```

**Impact:**

- Smaller stage2-target (20): Concentrated portfolio, higher variance
- Larger stage2-target (60): Diversified, lower variance, potentially lower Sharpe

**Recommendation:** Use the stage2-target that gave best in-sample Sharpe

---

## Output Files

### **1. Log File**

**Path:** `logs/portfolio_oos_validation_YYYYMMDD_HHMMSS.log`

**Contains:**

- Window-by-window detailed results
- IS vs OOS metrics per window
- Verdict per window
- Summary statistics
- Recommendation

**Size:** 5-15 KB

---

### **2. CSV Results**

**Path:** `logs/oos_validation_YYYYMMDD_HHMMSS.csv`

**Columns:**

```
window, portfolio, train_start, train_end, test_start, test_end,
is_sharpe, is_return, is_vol,
oos_sharpe, oos_return, oos_vol,
degradation_sharpe_pct, degradation_return_pct,
oos_max_drawdown, oos_win_rate, asset_coverage
```

**Use:** Import to Excel/Python for custom analysis

---

## Decision Framework

### **Step 1: Check Avg Degradation**

```
Avg Degradation < 20%? → Proceed to Step 2
Avg Degradation ≥ 20%? → REJECT or use HRP instead
```

---

### **Step 2: Check Stability**

```
OOS Std Dev < 0.5? → Proceed to Step 3
OOS Std Dev ≥ 0.5? → Check window-by-window (regime issue?)
```

---

### **Step 3: Make Decision**

```
If max_sharpe passes both:
  → Deploy max_sharpe

If max_sharpe fails either:
  → Run validation with --portfolio both
  → Compare with HRP
  → Deploy winner
```

---

## Common Workflows

### **Workflow A: Weekly Portfolio Update**

```bash
# Monday: Optimize on latest data
python portfolio_exploration_global.py \
    --csv ../../data/latest_prices.csv \
    --start 2024-01-01 --end 2026-01-01 \
    --stage2-target 40

# Output: max_sharpe Sharpe 5.85

# Tuesday: Validate OOS
python validate_portfolio_oos.py \
    --csv ../../data/latest_prices.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio max_sharpe

# Output: OOS 4.95 (-15% degradation, ACCEPTABLE)

# Wednesday: Deploy
# Update execution/config.py with weights
```

---

### **Workflow B: Quarterly Rebalancing Test**

```bash
# Test if quarterly rebalancing adds value
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 1 --test-months 3 \
    --portfolio max_sharpe

# Compare with annual rebalancing
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 1 --test-months 12 \
    --portfolio max_sharpe

# If quarterly OOS ≈ annual OOS:
#   → Stick with annual (lower transaction costs)
# If quarterly OOS >> annual OOS:
#   → Quarterly rebalancing adds value
```

---

### **Workflow C: Find Optimal stage2-target**

```bash
# Test multiple stage2-target values with OOS validation
for N in 20 30 40 50 60; do
  echo "Testing stage2-target=$N..."
  python validate_portfolio_oos.py \
      --csv ../../data/your_data.csv \
      --start 2020-01-01 --end 2026-01-01 \
      --stage2-target $N \
      --train-years 2 --test-months 12 \
      | grep "Avg Out-of-Sample" | sed "s/^/  stage2=$N: /"
done

# Output:
#   stage2=20: Avg Out-of-Sample: 4.150
#   stage2=30: Avg Out-of-Sample: 4.520
#   stage2=40: Avg Out-of-Sample: 4.680 ← BEST OOS
#   stage2=50: Avg Out-of-Sample: 4.520
#   stage2=60: Avg Out-of-Sample: 4.200

# Decision: Use stage2-target=40 (best OOS, not best IS)
```

---

## Troubleshooting

### **Issue: "Too few assets after quality filter"**

```
ERROR: Too few assets after quality filter (0 < 50)
FALLBACK: Creating equal-weight portfolio
```

**Cause:** Short training window + strict Sharpe filter

**Solution:**

```bash
# Lower min-sharpe threshold
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2023-01-01 --end 2026-01-01 \
    --min-sharpe 0.3  # Down from 0.5
```

**Or:** Increase train-years

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 3  # More training data
```

---

### **Issue: High Degradation (>30%)**

```
Avg Degradation: 52.2%
Verdict: 🚨 REJECT
```

**Possible causes:**

1. **Overfitting:** Portfolio fit to noise, not signal
2. **Regime change:** Train period doesn't match test period
3. **Look-ahead bias:** Data leakage (shouldn't happen with this validator)

**Solutions:**

**A) Switch to HRP:**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio hrp
```

**B) Increase training period:**

```bash
# More data = less overfitting
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --train-years 3  # Up from 2
```

**C) Reduce stage2-target:**

```bash
# Fewer assets = less parameters = less overfitting
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --stage2-target 20  # Down from 40
```

---

### **Issue: Unstable OOS (High Std Dev)**

```
OOS Std Dev: 1.120 (UNSTABLE)
  W1: 4.5
  W2: 1.8  ← Crash
  W3: 4.2
```

**Cause:** Strategy is regime-dependent

**Investigation:**

```bash
# Use rolling window to see which regimes break
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2019-01-01 --end 2026-01-01 \
    --mode rolling \
    --train-years 2 --test-months 12

# Check logs: Which test period had Sharpe 1.8?
# Was it 2020 (COVID crash), 2022 (bear market)?
```

**Solution:**

- Use HRP (more stable across regimes)
- OR implement regime detection
- OR add defensive assets (gold, bonds)

---

### **Issue: Low Asset Coverage (<80%)**

```
Asset Coverage: 67.8%
```

**Cause:** Many assets delisted or missing between train and test

**Impact:** Portfolio weights don't sum to 1.0 in test period

**Solution:**

```
Acceptable: Coverage drops naturally over time
Workaround: Validator auto-renormalizes weights
Monitor: If < 60%, consider reoptimizing
```

---

## Advanced: Custom Analysis

### **Extract Results for Plotting**

```python
import pandas as pd

# Load CSV
df = pd.read_csv('logs/oos_validation_20260108_010505.csv')

# Plot IS vs OOS over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['window'], df['is_sharpe'], label='In-Sample', marker='o')
plt.plot(df['window'], df['oos_sharpe'], label='Out-of-Sample', marker='s')
plt.xlabel('Window')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.title('In-Sample vs Out-of-Sample Performance')
plt.grid(True)
plt.savefig('oos_performance.png')
```

---

### **Statistical Significance Test**

```python
from scipy import stats

# Load results
df = pd.read_csv('logs/oos_validation_20260108_010505.csv')

# T-test: Is OOS Sharpe significantly > 0?
oos_sharpes = df['oos_sharpe'].values
t_stat, p_value = stats.ttest_1samp(oos_sharpes, 0)

print(f"OOS Sharpe t-test:")
print(f"  Mean: {oos_sharpes.mean():.3f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print("  ✅ Significantly positive (p < 0.05)")
else:
    print("  ❌ Not significant (might be luck)")
```

---

## Best Practices

### **1. Always Validate Before Live Trading**

```
NEVER deploy based on in-sample Sharpe alone.
ALWAYS run OOS validation first.
```

**Example disaster avoided:**

```
In-Sample:  Sharpe 7.2 (looks amazing!)
Out-of-Sample: Sharpe 1.8 (would lose money)
Degradation: 75% (extreme overfitting)

Without OOS validation: Lost $50k in 3 months
With OOS validation: Caught overfitting, used HRP instead
```

---

### **2. Compare Multiple Portfolios**

```bash
# Always test max_sharpe vs HRP
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both
```

**Why:** HRP often wins OOS despite lower IS

---

### **3. Use Expanding Mode for Deployment**

```
Expanding window mimics real-world:
  Year 1: Optimize on 2y data
  Year 2: Optimize on 3y data (more history)
  Year 3: Optimize on 4y data

Rolling window is for regime analysis, not deployment validation.
```

---

### **4. Minimum 3 Windows**

```
2 windows: Insufficient (could be luck)
3 windows: Minimum (start to see pattern)
5+ windows: Good statistics
10+ windows: Excellent (quarterly testing)
```

---

### **5. Accept Some Degradation**

```
Realistic expectations:
  0-10%:  Excellent (rare)
  10-20%: Good (common for well-designed strategies)
  20-30%: Marginal (deploy with caution)
  >30%:   Reject (overfitted)

Perfect (0% degradation) is suspicious (might indicate data leakage)
```

---

## Integration with Live Trading

### **After Validation Passes:**

```bash
# 1. Get recommended portfolio from validation
grep "RECOMMENDED" logs/portfolio_oos_validation_*.log

# Output: "✅ HRP WINS - STRONGLY RECOMMENDED"

# 2. Extract HRP weights from latest portfolio_exploration log
grep -A 40 "HRP:" logs/portfolio_exploration_*.log | grep -E "^\s+\d+\."

# 3. Update live trading config
# Edit execution/config.py:

TARGET_ALLOCATION = {
    'IAU': 0.0850,
    'NVDA': 0.0720,
    'WELL': 0.0680,
    # ... (HRP weights from step 2)
}

ASSET_SPECIFIC_CONFIGS = {
    'IAU': {'strategy_type': 'buy_and_hold', 'kelly_fraction': 1.0},
    'NVDA': {'strategy_type': 'ml_signal', 'model_type': 'lstm', ...},
    # ...
}

# 4. Validate config
python validate_config.py

# 5. Deploy
python execution/main.py
```

---

## Interpreting Edge Cases

### **Case 1: OOS Better Than IS**

```
Avg Degradation: -5.2% (negative = OOS > IS)
```

**Possible causes:**

1. **Lucky test period:** Favorable market conditions
2. **Conservative training:** Underfit in-sample, generalizes better
3. **Small sample:** Statistical noise (need more windows)

**Action:** Good sign but verify with more windows

---

### **Case 2: Inconsistent Windows**

```
W1: Degradation +8%  (good)
W2: Degradation +45% (terrible)
W3: Degradation +12% (good)
```

**Interpretation:** Window 2 regime mismatch

**Investigation:**

- Check dates: Was W2 test period 2022 (bear market)?
- Check assets: Did key holdings crash in W2?
- Check correlation: Did correlation structure break?

**Action:**

- Use rolling mode to identify problematic regime
- Add regime detection to live trading

---

### **Case 3: Both Portfolios Fail**

```
max_sharpe: Degradation 45% (REJECT)
hrp:        Degradation 38% (REJECT)
```

**Possible causes:**

1. **Data quality:** Bad price data, missing values
2. **Universe selection:** All 391 assets are poor quality
3. **Regime shift:** 2020-2022 doesn't predict 2023-2026

**Actions:**

1. Check data: Run `scripts/validate_data_csv.py`
2. Increase train period: Try 3-4 years
3. Lower min-sharpe: Try 0.3 instead of 0.5
4. Consider equal-weight: Simple 1/N portfolio as baseline

---

## Performance Benchmarks

### **Expected Execution Times:**

| Configuration | Windows | Time per Window | Total Time |
|---------------|---------|-----------------|------------|
| 2y train, 1y test, expanding | 3 | 2s | ~6 seconds |
| 2y train, 1y test, rolling | 4 | 2s | ~8 seconds |
| 1y train, 3m test, expanding | 10 | 1.5s | ~15 seconds |
| 3y train, 1y test, expanding | 2 | 2.5s | ~5 seconds |
| both portfolios (×2) | × 2 | × 2 | Double |

---

## Real-World Example (Your Dataset)

**From your test:**

```
Dataset: 478 assets, 2023-2026
Config: train-years=1, test-months=6, stage2-target=40

Result:
  Window 1: IS=5.07 OOS=3.35 (+33.8% degradation)
  Window 2: IS=5.71 OOS=1.68 (+70.6% degradation)

  Avg Degradation: 52.2%
  Verdict: 🚨 REJECT - SEVERE OVERFITTING

Interpretation:
  - Only 1 year training = insufficient data
  - 391→40 asset compression = high overfitting risk
  - Need longer training or simpler portfolio (HRP)

Recommendation:
  1. Try train-years=2 (more data)
  2. Test --portfolio both (compare vs HRP)
  3. Likely: HRP will win OOS
```

---

## Summary

**To validate any portfolio:**

```bash
# Basic command
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both

# Decision tree:
#   Degradation < 20% → Deploy
#   Degradation ≥ 20% → Use HRP or reject
#   HRP wins → Always use HRP for robustness
```

**Key metrics:**

- Avg Degradation < 20% (acceptable)
- OOS Std Dev < 0.5 (stable)
- Asset Coverage > 80% (tradeable)

**Output:** Clear recommendation (deploy max_sharpe, deploy HRP, or reject)

---

**The validator saved you from deploying overfitted portfolios. Use it before every deployment.**

---

## VALIDATOR 2: validate_fixed_portfolio.py (Fixed Weights Testing)

### Purpose

Tests whether **specific portfolio weights** perform consistently across multiple time periods.

**Key difference:** NO re-optimization - tests the SAME weights on different time periods.

---

### Quick Start

```bash
cd algos/backtest_code

# Step 1: Optimize portfolio once
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --stage2-target 40

# Output saved to: logs/portfolio_exploration_20260108_HHMMSS.log

# Step 2: Test those specific weights on yearly periods
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_20260108_145404.log \
    --portfolio max_sharpe \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 12
```

**What it does:**
```
Extract weights from log: IAU 15%, NVDA 12%, WELL 10%, ...

Period 1: 2020 → Test weights on 2020 data → Sharpe 4.2
Period 2: 2021 → Test weights on 2021 data → Sharpe 3.8
Period 3: 2022 → Test weights on 2022 data → Sharpe 4.1
Period 4: 2023 → Test weights on 2023 data → Sharpe 3.9
Period 5: 2024 → Test weights on 2024 data → Sharpe 4.0
Period 6: 2025 → Test weights on 2025 data → Sharpe 4.1

Output:
  Avg Sharpe: 4.02
  Std Dev: 0.15 (stable)
  Coefficient of Variation: 0.037 (very stable)
  Verdict: ✅ VERY STABLE - Deploy for buy-and-hold
```

**Execution time:** ~2-3 seconds

---

### Input Methods

#### **Method 1: From Log File (Recommended)**

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_20260108_145404.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01
```

**Automatically extracts weights** from the "All Holdings" section of the log.

**Portfolios available:** max_sharpe, hrp, min_volatility

---

#### **Method 2: Manual Weights**

```bash
python validate_fixed_portfolio.py \
    --weights "IAU:0.15,NVDA:0.12,WELL:0.10,WMT:0.08,AVGO:0.07" \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01
```

**Use when:**
- Testing custom portfolio
- Blending portfolios (50% max_sharpe + 50% hrp)
- External portfolio source

---

### Test Period Configurations

#### **Yearly Testing (Standard)**

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 12
```

**Result:** 5-6 yearly periods
**Use:** Standard consistency check

---

#### **Quarterly Testing (High Statistical Power)**

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3
```

**Result:** 20-24 quarterly periods
**Advantage:** 
- High statistical power (n ≥ 20)
- T-test for significance
- 95% confidence intervals

---

#### **Monthly Testing (Maximum Granularity)**

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio hrp \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 1
```

**Result:** 60-72 monthly periods
**Use:** Detect seasonality, high statistical power

---

### Output Interpretation

#### **Coefficient of Variation (CV)**

**CV = Std Dev / Mean** (measures relative stability)

| CV | Stability | Interpretation | Action |
|----|-----------|----------------|--------|
| **< 0.15** | ✅ Very Stable | Consistent across all periods | Deploy with high confidence |
| **0.15-0.30** | ✅ Stable | Minor variation acceptable | Deploy |
| **0.30-0.50** | ⚠️ Moderate | Some regime dependence | Deploy with monitoring |
| **> 0.50** | 🚨 Unstable | Breaks in some regimes | Reject or add regime detection |

**Example:**
```
Portfolio A: Mean Sharpe 4.0, Std 0.6, CV = 0.15 (stable)
Portfolio B: Mean Sharpe 5.0, Std 2.5, CV = 0.50 (unstable)

→ Portfolio A is better for buy-and-hold (lower but more consistent)
```

---

#### **Statistical Significance (if ≥10 periods)**

**Automatically calculated when n ≥ 10:**

```
T-test (H0: Sharpe = 0):
  t-statistic: 8.234
  p-value:     0.0002
  ✅ HIGHLY SIGNIFICANT (p < 0.01) - Portfolio has positive Sharpe

  95% Confidence Interval: [3.45, 4.55]
  ✅ Lower bound > 0 - Positive Sharpe with high confidence
```

**Interpretation:**
- **p < 0.01:** Very confident portfolio is profitable
- **p < 0.05:** Confident portfolio is profitable
- **p ≥ 0.05:** Not confident (could be luck)
- **CI lower > 0:** Guarantees positive Sharpe with 95% confidence

---

### Use Cases for Fixed Portfolio Validator

#### **Use Case 1: Buy-and-Hold Validation**

**Scenario:** You want to hold portfolio for 5 years without rebalancing.

```bash
# Step 1: Optimize once
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2025-01-01_1d.csv \
    --start 2020-01-01 --end 2025-01-01 \
    --stage2-target 40

# Step 2: Test if weights are stable 2020-2025
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 12

# Output:
#   CV = 0.12 (very stable)
#   Avg Sharpe 4.2
#   Verdict: ✅ VERY STABLE - Deploy for buy-and-hold
```

**Decision:** Hold portfolio for 5 years, check annually but don't reoptimize.

---

#### **Use Case 2: Regime Stability Check**

**Scenario:** Check if portfolio works in bull AND bear markets.

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/financial_data_combined_prices_2019-01-01_2026-01-01_1d.csv \
    --start 2019-01-01 --end 2026-01-01 \
    --test-period-months 6  # Semi-annual (detect regime changes)
```

**Example output:**
```
Period 1 (2019 H1): Sharpe 4.5 (bull)
Period 2 (2019 H2): Sharpe 4.2 (bull)
Period 3 (2020 H1): Sharpe 1.8 (COVID crash) ← BREAKS
Period 4 (2020 H2): Sharpe 5.2 (recovery)
Period 5 (2021 H1): Sharpe 4.8 (bull)
...

CV = 0.45 (moderate - regime dependent)
Verdict: ⚠️  MODERATE - Breaks during crashes
```

**Decision:** 
- Add defensive assets (gold, bonds)
- OR accept that it crashes in bear markets
- OR use HRP instead (more stable)

---

#### **Use Case 3: Compare max_sharpe vs HRP Stability**

```bash
# Test max_sharpe
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3  # Quarterly

# Test HRP
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio hrp \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3
```

**Compare results:**
```
max_sharpe: Avg Sharpe 4.8, CV 0.35 (moderate stability)
hrp:        Avg Sharpe 4.2, CV 0.18 (very stable)

→ max_sharpe has higher return but more volatile
→ hrp has lower return but more consistent
→ Choose based on risk tolerance
```

---

#### **Use Case 4: Test Portfolio Before Long-Term Hold**

**Scenario:** You optimized a portfolio, want to hold 10 years. Will it work?

```bash
# Optimize on all available data
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2010-01-01_2025-01-01_1d.csv \
    --start 2010-01-01 --end 2025-01-01

# Test stability on historical 15-year period
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio hrp \
    --csv ../../data/financial_data_combined_prices_2010-01-01_2025-01-01_1d.csv \
    --start 2010-01-01 --end 2025-01-01 \
    --test-period-months 12  # 15 yearly periods

# Output:
#   15 periods tested
#   CV = 0.22 (stable)
#   t-test p-value: 0.0001 (highly significant)
#   Verdict: ✅ STABLE across 15 years - good for long-term hold
```

---

### Sample Output

```
================================================================================
 FIXED PORTFOLIO CONSISTENCY TEST
================================================================================

Portfolio Weights (39 assets):
   1. PLTR                  6.80%
   2. Rolls_Royce           5.82%
   3. NVDA                  5.38%
   ...

Test Configuration:
  Test Period Size:   12 months
  Date Range:         2020-01-03 to 2025-12-31
  Total Days:         1510

================================================================================
PERIOD 1
================================================================================
  Date Range: 2020-01-03 to 2021-01-03
  Trading Days: 252

  Performance:
    Sharpe Ratio:      4.230
    Annual Return:     0.452
    Annual Volatility: 0.105
    Total Return:      45.20%
    Max Drawdown:      -8.30%
    Win Rate:          58.3%
    Asset Coverage:    39/39 (100.0%)

================================================================================
PERIOD 2
================================================================================
  Date Range: 2021-01-03 to 2022-01-03
  Trading Days: 252

  Performance:
    Sharpe Ratio:      3.850
    Annual Return:     0.398
    Annual Volatility: 0.101
    Total Return:      39.80%
    Max Drawdown:      -12.10%
    Win Rate:          55.6%
    Asset Coverage:    38/39 (97.4%)

[... more periods ...]

================================================================================
 CONSISTENCY SUMMARY
================================================================================

Performance Across 5 Periods:
  Sharpe Ratio:
    Mean:      4.020
    Std Dev:   0.180 (stable)
    Range:     3.750 to 4.230

  Returns & Risk:
    Avg Annual Return: 0.425
    Avg Volatility:    0.103
    Worst Drawdown:    -15.20%

  Asset Coverage:
    Average: 98.2%

================================================================================
 STABILITY VERDICT
================================================================================

✅ VERY STABLE
  CV=0.045 - Consistent across all periods

Performance: ✅ EXCELLENT (avg Sharpe 4.02)

================================================================================
 STATISTICAL SIGNIFICANCE
================================================================================

T-test (H0: Sharpe = 0):
  t-statistic: 12.345
  p-value:     0.0001
  ✅ HIGHLY SIGNIFICANT (p < 0.01) - Portfolio has positive Sharpe

  95% Confidence Interval: [3.720, 4.320]
  ✅ Lower bound > 0 - Positive Sharpe with high confidence

================================================================================
 PERIOD-BY-PERIOD BREAKDOWN
================================================================================

Period   Date Range                     Sharpe   Return   Vol      Max DD    
-------- ------------------------------ -------- -------- -------- ----------
P1       2020-01-03 to 2021-01-03       4.230    0.452    0.105    -8.30%    
P2       2021-01-03 to 2022-01-03       3.850    0.398    0.101    -12.10%   
P3       2022-01-03 to 2023-01-03       3.980    0.415    0.102    -10.50%   
P4       2023-01-03 to 2024-01-03       4.150    0.438    0.104    -7.80%    
P5       2024-01-03 to 2025-01-03       3.900    0.422    0.106    -15.20%   

================================================================================
 RECOMMENDATION
================================================================================

✅ PORTFOLIO IS ROBUST
   Stable across 5 periods (CV=0.045)
   Average Sharpe 4.02 (good performance)

   → DEPLOY with confidence for buy-and-hold strategy
```

---

### When to Use Which Validator

| Your Question | Use This Validator | Why |
|---------------|-------------------|-----|
| "Should I reoptimize yearly?" | `validate_portfolio_oos.py` | Tests if re-optimization adds value |
| "Can I hold this portfolio for 5 years?" | `validate_fixed_portfolio.py` | Tests stability over time |
| "Is max_sharpe better than HRP?" | `validate_portfolio_oos.py --portfolio both` | Compares strategies with re-opt |
| "Are these weights regime-stable?" | `validate_fixed_portfolio.py` | Tests across bull/bear/sideways |
| "Does quarterly rebalancing help?" | `validate_portfolio_oos.py` with test-months=3 | Tests rebalancing frequency |
| "Will my portfolio break in a crash?" | `validate_fixed_portfolio.py` with monthly periods | High granularity regime check |

---

### Statistical Power Comparison

**validate_portfolio_oos.py (Re-optimization):**
```
2y train, 1y test, 2020-2026:
  Windows: 3-4
  Paired samples: 3-4
  Statistical power: LOW (insufficient for t-test)
  
To get significance: Use 1y train, 3m test (15-18 windows)
```

**validate_fixed_portfolio.py (Fixed Weights):**
```
12-month periods, 2020-2026:
  Periods: 5-6
  Statistical power: LOW-MEDIUM
  
3-month periods, 2020-2026:
  Periods: 20-24
  Statistical power: HIGH (t-test significant)
  ✅ Confidence intervals meaningful
```

**Key insight:** Fixed portfolio validator gives **more statistical power** (more test periods, no training overhead).

---

### Complete Workflow Example

```bash
# ============================================
# COMPLETE VALIDATION WORKFLOW
# ============================================

# Step 1: Optimize portfolio in-sample
python portfolio_exploration_global.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2025-01-01_1d.csv \
    --start 2020-01-01 --end 2025-01-01 \
    --min-sharpe 0.5 --stage2-target 40

# Output:
#   max_sharpe: Sharpe 5.82 (in-sample)
#   hrp:        Sharpe 4.25 (in-sample)
#   Log: logs/portfolio_exploration_20260108_145404.log

# Step 2: Test optimization strategy (will it work if I reoptimize yearly?)
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both \
    --train-years 2 --test-months 12

# Output:
#   max_sharpe: Avg OOS 3.20, Degradation 45% (REJECT)
#   hrp:        Avg OOS 3.85, Degradation 9% (EXCELLENT)
#   Winner: HRP

# Step 3: Test HRP fixed weights (can I hold for 5 years?)
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_20260108_145404.log \
    --portfolio hrp \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3  # Quarterly for high stat power

# Output:
#   22 periods tested
#   Avg Sharpe 3.92, CV 0.18 (stable)
#   t-test p-value: 0.0001 (highly significant)
#   Verdict: ✅ VERY STABLE

# ============================================
# DECISION TREE
# ============================================

# IF validate_portfolio_oos.py says HRP wins:
#   → Use HRP
#
# IF validate_fixed_portfolio.py says HRP is stable (CV < 0.30):
#   → Deploy HRP for buy-and-hold (no rebalancing)
#
# ELSE IF HRP is unstable (CV > 0.50):
#   → Deploy with quarterly rebalancing
#   → Monitor regime changes

# Final Decision: Deploy HRP, hold for 1 year, recheck annually
```

---

### Troubleshooting

#### **Issue: No weights extracted from log**

```
ERROR: Failed to extract max_sharpe from log file
```

**Cause:** Log format mismatch or portfolio not in log

**Solution:**
```bash
# Check log file has the portfolio
grep "MAX SHARPE:" logs/portfolio_exploration_*.log

# If found, try specifying exact log file
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_20260108_145404.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01

# If still fails, use manual weights
grep -A 40 "MAX SHARPE:" logs/portfolio_exploration_*.log
# Copy weights, format as: IAU:0.15,NVDA:0.12,...
```

---

#### **Issue: Low asset coverage**

```
Asset Coverage: 38/39 (97.4%)
```

**Interpretation:** 1 asset delisted or missing in test period

**Impact:**
- Validator auto-renormalizes remaining 38 assets
- Minor impact if coverage > 95%
- Significant impact if coverage < 80%

**Action:**
```
Coverage > 95%: Acceptable (normal attrition)
Coverage 80-95%: Monitor (some portfolio drift)
Coverage < 80%: Reoptimize (significant change)
```

---

#### **Issue: Unstable results (CV > 0.5)**

```
CV = 0.68 (UNSTABLE)
  Period 1: Sharpe 5.2
  Period 2: Sharpe 1.5  ← CRASH
  Period 3: Sharpe 4.8
```

**Cause:** Portfolio is regime-dependent

**Investigation:**
```bash
# Check which period crashed
grep "Period 2" logs/fixed_portfolio_validation_*.log -A 10

# Was it 2020 (COVID), 2022 (bear market)?
# Which assets crashed?
```

**Solutions:**
1. **Add defensive assets** (gold, bonds, utilities)
2. **Use HRP** (more stable across regimes)
3. **Implement regime detection** (switch strategies based on VIX, market state)
4. **Accept rebalancing** (use validate_portfolio_oos.py instead)

---

### Advanced: Custom Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('logs/fixed_portfolio_validation_20260108_HHMMSS.csv')

# Plot Sharpe over time
plt.figure(figsize=(12, 6))
plt.plot(df['start_date'], df['sharpe'], marker='o', linewidth=2)
plt.axhline(y=df['sharpe'].mean(), color='r', linestyle='--', label='Average')
plt.xlabel('Period Start Date')
plt.ylabel('Sharpe Ratio')
plt.title('Fixed Portfolio Sharpe Ratio Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fixed_portfolio_stability.png')

# Identify worst period
worst = df.loc[df['sharpe'].idxmin()]
print(f"Worst period: {worst['start_date']} to {worst['end_date']}")
print(f"  Sharpe: {worst['sharpe']:.2f}")
print(f"  Investigate what happened in this period")
```

---

## Summary: When to Use Which

### **Use validate_portfolio_oos.py if:**

✅ You plan to reoptimize regularly (yearly/quarterly)
✅ Comparing optimization strategies (max_sharpe vs HRP)
✅ Want to know if optimization adds value vs buy-and-hold

### **Use validate_fixed_portfolio.py if:**

✅ You have specific weights to test
✅ Testing buy-and-hold strategy (no reoptimization)
✅ Checking regime stability
✅ Want statistical significance (easier with more periods)

### **Use BOTH if:**

✅ Comprehensive validation before deployment
✅ Decision: Reoptimize yearly OR buy-and-hold?

**Workflow:**
```bash
# 1. Test optimization strategy
python validate_portfolio_oos.py --portfolio both

# If max_sharpe wins: Test if you can hold it long-term
# 2. Test fixed weights stability
python validate_fixed_portfolio.py --portfolio max_sharpe

# If stable (CV < 0.30):
#   → Buy-and-hold (no rebalancing)
# If unstable (CV > 0.30):
#   → Reoptimize yearly (active management)
```

---

**Both validators are production-ready. Use them before every deployment.**

---

### **Mode 3: Monte Carlo (High Statistical Power)**

**What it does:**
- Generates N **random train/test splits** (50-200 iterations)
- Uses **quartile stratification** (ensures regime coverage)
- Includes **embargo** (2% gap prevents temporal leakage, López de Prado methodology)
- **High statistical power** (n=50-200 → meaningful t-tests and confidence intervals)

**Example:**
```
Iteration 1:  Random start 2020-03-15 → Train 2y → Embargo 10d → Test 2023-04-05 to 2024-04-05
Iteration 2:  Random start 2020-08-22 → Train 2y → Embargo 10d → Test 2023-09-11 to 2024-09-11
Iteration 3:  Random start 2021-02-10 → Train 2y → Embargo 10d → Test 2024-03-01 to 2025-03-01
...
Iteration 100: Random start 2021-11-30 → Train 2y → Embargo 10d → Test 2024-12-20 to 2025-12-20

Result: n=100 samples → robust statistics
```

**When to use:**
- **Before deployment:** Prove strategy is fundamentally sound
- **Comparing strategies:** max_sharpe vs HRP with statistical rigor
- **Long training windows:** 3y train + 1y test = only 2 sequential windows (useless), but 100 Monte Carlo windows (excellent)
- **Need confidence intervals:** "OOS Sharpe is 3.2 ± 0.4 with 95% confidence"

**Command:**
```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --iterations 100 \
    --train-years 2 --test-years 1 \
    --min-sharpe 0.4
```

**Advantages over Sequential:**
- ✅ **High statistical power** (50-200 samples vs 2-4)
- ✅ **Robust confidence intervals** (meaningful with n ≥ 30)
- ✅ **T-tests work** (p-values significant)
- ✅ **Better regime coverage** (samples all market conditions)
- ✅ **Solves 3y train problem** (only 2 sequential windows → 100 Monte Carlo windows)

**Disadvantages:**
- ⚠️ **Test windows overlap** (50-60% overlap typical)
  - Results partially correlated (not fully independent)
  - CIs slightly optimistic (validator warns about this)
- ⚠️ **Not deployment-realistic** (doesn't simulate live trading)
  - Random starts "look into future" for training
  - Good for **strategy validation**, not **deployment simulation**
- ⚠️ **Slower** (100 iterations × 2s = 3-4 minutes vs 6 seconds sequential)

**Interpretation:**
```
Monte Carlo says: "This strategy works 95% of the time across different market conditions"
Sequential says: "This strategy will work starting tomorrow"

Use both:
  - Monte Carlo: Prove strategy is sound (before building system)
  - Sequential (expanding): Validate deployment (before going live)
```

---

### **Mode Comparison: When to Use Each**

| Your Question | Use This Mode | Why |
|---------------|---------------|-----|
| "Is max_sharpe better than HRP?" | **monte_carlo** | Need stats (n=100, p-value) |
| "Will it work when I deploy tomorrow?" | **expanding** | Realistic (train on all past) |
| "Does it break in bear markets?" | **rolling** | Regime detection |
| "3y train only gives 2 windows!" | **monte_carlo** | Generate 100 windows |
| "I need 95% confidence intervals" | **monte_carlo** | High sample size (n ≥ 30) |
| "Final pre-deployment check" | **expanding** then **monte_carlo** | Both perspectives |

---

## Monte Carlo Examples

### **Example 1: Standard Monte Carlo Validation**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --portfolio both \
    --iterations 100 \
    --train-years 2 --test-years 1 \
    --min-sharpe 0.4
```

**Output:**
```
MAX_SHARPE:
  Iterations: 100
  Avg IS: 4.52, Avg OOS: 3.18 ± 0.42 (95% CI: [3.10, 3.26])
  T-test: p < 0.0001 ✅ HIGHLY SIGNIFICANT
  Degradation: 29.6%
  Verdict: ⚠️ CAUTION

HRP:
  Iterations: 100
  Avg IS: 3.85, Avg OOS: 3.62 ± 0.28 (95% CI: [3.56, 3.68])
  T-test: p < 0.0001 ✅ HIGHLY SIGNIFICANT
  Degradation: 6.0%
  Verdict: ✅ EXCELLENT

RECOMMENDATION: HRP wins (lower degradation, tighter CI)
```

**Decision:** Deploy HRP with statistical confidence.

---

### **Example 2: Conservative Validation (3y train, many iterations)**

**Problem:** 3y train + 1y test on 6y data = only 2 sequential windows (useless statistics)

**Solution:** Monte Carlo generates 100-200 windows

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2018-01-01_2026-01-01_1d.csv \
    --start 2018-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --portfolio both \
    --iterations 150 \
    --train-years 3 --test-years 1 \
    --min-sharpe 0.5
```

**Result:**
```
Iterations: 150
Avg OOS Sharpe: 3.45 ± 0.25 (95% CI: [3.41, 3.49])
T-test: t=42.3, p < 0.0001 (df=149)

Verdict: ✅ EXCELLENT - Strategy validated with 150 independent tests
```

---

### **Example 3: Short-window High-frequency Test**

```bash
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --portfolio max_sharpe \
    --iterations 200 \
    --train-years 1 --test-years 1 \
    --min-sharpe 0.3 \
    --stage1-top-n 80 --stage2-target 30
```

**Why:**
- 1y train (short) allows more start points
- 200 iterations (maximum statistical power)
- Tests if strategy works with limited training data

**Expected:**
```
Iterations: 200
Higher overfitting (short train) but statistically proven
```

---

### **Example 4: Reproduce Results (Reproducibility)**

```bash
# Run 1
python validate_portfolio_oos.py \
    --csv data.csv --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo --iterations 100 --seed 42

# Run 2 (different day, same seed)
python validate_portfolio_oos.py \
    --csv data.csv --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo --iterations 100 --seed 42

# Result: IDENTICAL results (same windows, same OOS Sharpe)
```

**Use:** Reproducible research, auditing

---

## Interpreting Monte Carlo Results

### **Statistical Outputs**

**1. T-test (Is OOS Sharpe truly positive?):**
```
T-test (H0: OOS Sharpe = 0):
  t-statistic: 11.366
  p-value:     0.0000
  ✅ HIGHLY SIGNIFICANT (p < 0.01)
```

**Interpretation:**
- **p < 0.01:** 99% confident strategy is profitable
- **p < 0.05:** 95% confident
- **p ≥ 0.05:** Not confident (could be luck)

---

**2. Confidence Intervals:**
```
95% Confidence Interval: [1.66, 2.42]
  ✅ Lower bound > 0 (positive Sharpe guaranteed)
```

**Interpretation:**
- **Lower > 0:** Portfolio guaranteed profitable with 95% confidence
- **Lower ≤ 0:** Uncertain (might lose money)
- **Narrow CI:** Consistent strategy (tight range)
- **Wide CI:** Inconsistent strategy (high variance)

**Example:**
```
Portfolio A: OOS 3.2 ± 0.3, CI [2.9, 3.5] (tight, predictable)
Portfolio B: OOS 3.5 ± 1.2, CI [2.3, 4.7] (wide, unpredictable)

→ Portfolio A is better for deployment (more reliable)
```

---

**3. Overlap Analysis:**
```
Test Window Overlap Analysis:
  64/120 window pairs overlap (53.3%)
  ⚠️  High overlap (>50%) - results partially correlated
  Confidence intervals may be slightly optimistic
```

**What it means:**
- **< 30% overlap:** Results mostly independent (ideal)
- **30-60% overlap:** Moderate correlation (acceptable, reported transparently)
- **> 60% overlap:** High correlation (use conservative p-value thresholds)

**Not a problem if:**
- You use p < 0.01 threshold (instead of 0.05)
- You acknowledge overlap in decision-making
- Alternative (sequential) has low stat power anyway

---

### **Degradation Interpretation (Monte Carlo)**

**Typical ranges:**

| Avg Degradation | Verdict | Interpretation |
|-----------------|---------|----------------|
| **< 0%** | ✅ EXCELLENT | OOS > IS (rare, very good) |
| **0-10%** | ✅ EXCELLENT | Minimal overfitting |
| **10-20%** | ✅ ACCEPTABLE | Expected for complex strategies |
| **20-30%** | ⚠️ CAUTION | Significant overfitting |
| **> 30%** | 🚨 REJECT | Severe overfitting |

**Real-world example (from test):**
```
max_sharpe: Degradation 36.1% (REJECT)
hrp:        Degradation -31.1% (OOS BETTER than IS!)

Conclusion: HRP is robust, max_sharpe overfits
```

---

### **When Monte Carlo Says "REJECT"**

**Example:**
```
MAX_SHARPE (Monte Carlo, n=100):
  Avg IS: 5.2, Avg OOS: 2.8
  Degradation: 46.2%
  CI: [2.5, 3.1]
  Verdict: 🚨 REJECT
```

**Actions:**
1. **Switch to HRP** (test if it's better)
2. **Increase training window** (3y instead of 2y)
3. **Reduce stage2-target** (20 instead of 40 - fewer parameters)
4. **Accept reality** (your strategy overfits, don't deploy)

---

## Complete Workflow: All Three Modes

### **Step 1: Monte Carlo Validation (Prove Strategy)**

```bash
# High stat power, proves strategy fundamentally works
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --portfolio both \
    --iterations 100 \
    --train-years 2 --test-years 1
```

**Output:**
```
max_sharpe: OOS 3.2 ± 0.4, Degradation 28%, p<0.0001
hrp:        OOS 3.6 ± 0.3, Degradation 8%, p<0.0001

Winner: HRP (statistically proven)
```

---

### **Step 2: Sequential Validation (Deployment Reality)**

```bash
# Realistic simulation: Train on all past, test on future
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode expanding \
    --portfolio hrp \  # Use winner from Step 1
    --train-years 2 --test-months 12
```

**Output:**
```
Windows: 3
Avg OOS: 3.5 (matches Monte Carlo!)
Verdict: ✅ Works in deployment scenario too
```

---

### **Step 3: Regime Check (Rolling)**

```bash
# Does it work in ALL market conditions?
python validate_portfolio_oos.py \
    --csv ../../data/financial_data_combined_prices_2019-01-01_2026-01-01_1d.csv \
    --start 2019-01-01 --end 2026-01-01 \
    --mode rolling \
    --portfolio hrp \
    --train-years 2 --test-months 12
```

**Output:**
```
W1 (bull):  OOS 4.2
W2 (crash): OOS 2.1  ← Check if acceptable
W3 (recovery): OOS 3.8

If W2 > 1.0: Acceptable (still positive in crash)
If W2 < 0: Reject (loses money in crash)
```

---

### **Step 4: Deploy**

```
All 3 modes pass → Deploy with HIGH confidence
Monte Carlo pass + Sequential pass → Deploy with MEDIUM confidence
Only Monte Carlo pass → Consider more validation
```

---

## Use Case Matrix

| Scenario | Mode | Iterations | Train | Test | Command |
|----------|------|------------|-------|------|---------|
| **Pre-deployment (final check)** | expanding | N/A | 2y | 12m | `--mode expanding` |
| **Prove strategy works** | monte_carlo | 100 | 2y | 1y | `--mode monte_carlo --iterations 100` |
| **Compare max_sharpe vs HRP** | monte_carlo | 100 | 2y | 1y | `--mode monte_carlo --portfolio both` |
| **3y train (only 2 sequential windows)** | monte_carlo | 150 | 3y | 1y | `--mode monte_carlo --train-years 3 --iterations 150` |
| **Regime stability** | rolling | N/A | 2y | 12m | `--mode rolling` |
| **Maximum confidence** | monte_carlo | 200 | 3y | 1y | `--mode monte_carlo --iterations 200 --train-years 3` |

---

## Monte Carlo Parameters

### **iterations (Sample Size)**

| Iterations | Statistical Power | Use Case |
|------------|-------------------|----------|
| **20-30** | Low (barely sufficient) | Quick test |
| **50** | Medium (t-test works) | Standard |
| **100** | High (robust CIs) | **Recommended** |
| **150-200** | Very High (maximum confidence) | Final validation |

**Rule of thumb:** n ≥ 30 for Central Limit Theorem to apply

---

### **embargo_pct (Temporal Leakage Prevention)**

| Embargo % | Days (2y train) | Use Case |
|-----------|-----------------|----------|
| **0%** | 0 (no gap) | ❌ Not recommended (leakage risk) |
| **1%** | ~5 days | Minimum |
| **2%** | ~10 days | **Default (López de Prado standard)** |
| **3-5%** | 15-25 days | Conservative |

**Why embargo matters:**
```
Without embargo:
  Train ends: 2023-12-31
  Test starts: 2024-01-01 (next day!)
  
  Problem: Last 5 days of training correlate with first 5 days of test
  Result: Inflated OOS Sharpe (temporal leakage)

With 2% embargo:
  Train ends: 2023-12-31
  Embargo: 10 days
  Test starts: 2024-01-10 (gap prevents leakage)
```

---

### **seed (Reproducibility)**

```bash
# Same seed = same results (reproducible)
python validate_portfolio_oos.py --mode monte_carlo --seed 42
```

**Use:**
- Reproducible research
- Debugging (same windows every run)
- Auditing (prove results to others)

**Different seeds:**
```bash
# Test robustness to random sampling
python validate_portfolio_oos.py --mode monte_carlo --seed 42
python validate_portfolio_oos.py --mode monte_carlo --seed 123
python validate_portfolio_oos.py --mode monte_carlo --seed 999

# If all 3 give similar OOS → Very robust
# If results vary widely → Strategy sensitive to sample selection
```

---

## Troubleshooting Monte Carlo

### **Issue: High failure rate**

```
✅ Monte Carlo complete: 15 successful / 100 attempted (85 failed)
   ⚠️  High failure rate (85/100)
```

**Causes:**
1. **Strict filters:** min_sharpe=0.5 too high for short windows
2. **Short training:** 1y train → few quality assets
3. **Small universe:** < 50 assets total

**Solutions:**
```bash
# Lower min-sharpe
python validate_portfolio_oos.py --mode monte_carlo --min-sharpe 0.2

# Increase training window
python validate_portfolio_oos.py --mode monte_carlo --train-years 3

# Lower stage1 target
python validate_portfolio_oos.py --mode monte_carlo --stage1-top-n 50 --stage2-target 20
```

---

### **Issue: Too few iterations**

```
Iterations: 20
Statistical: Marginal (n < 30)
```

**Solution:**
```bash
python validate_portfolio_oos.py --mode monte_carlo --iterations 100
# n=100 gives robust statistics
```

---

### **Issue: All windows overlap >80%**

```
Test Window Overlap: 98/120 pairs (81.7%)
⚠️  Very high overlap - results highly correlated
```

**Cause:** Short data range + long test windows

**Impact:** Confidence intervals optimistic (overstated)

**Solutions:**
1. **Use p < 0.01** threshold (instead of 0.05)
2. **Acknowledge limitation** in decision-making
3. **Shorter test windows** (6m instead of 12m)
4. **Longer data range** (use 10y data instead of 5y)

**Example:**
```
6y data, 2y train, 1y test:
  Sequential: 3 windows (n=3)
  Monte Carlo: 100 windows but 80% overlap

Trade-off: n=100 with overlap > n=3 without
Monte Carlo still provides more information
```

---

## Best Practices: Monte Carlo

### **1. Always use with Sequential**

```
DON'T: Only Monte Carlo (not deployment-realistic)
DO: Monte Carlo (prove strategy) + Expanding (validate deployment)
```

**Workflow:**
```bash
# 1. Prove strategy works (Monte Carlo)
python validate_portfolio_oos.py --mode monte_carlo --iterations 100

# If passes:
# 2. Validate deployment (Expanding)
python validate_portfolio_oos.py --mode expanding

# If both pass:
# 3. Deploy to live trading
```

---

### **2. Use iterations ≥ 50**

```
n < 30: Insufficient (CLT doesn't apply)
n = 50: Minimum
n = 100: Recommended
n ≥ 150: Excellent
```

---

### **3. Report overlap percentage**

```
Always check overlap in logs:
  Overlap < 50%: Good (mostly independent)
  Overlap ≥ 50%: Acknowledge (partially correlated)
```

---

### **4. Conservative thresholds for high overlap**

```
If overlap > 60%:
  Use p < 0.01 (instead of p < 0.05)
  Require degradation < 15% (instead of < 20%)
  Be skeptical of tight CIs
```

---

