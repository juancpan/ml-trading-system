# Portfolio Validation - Quick Reference

## Two Validators, Three Questions

```
┌─────────────────────────────────────────────────────────────┐
│ "Is my strategy statistically sound?" (HIGH CONFIDENCE)     │
│ → validate_portfolio_oos.py --mode monte_carlo             │
│   Tests: 100 random windows, t-tests, CIs                  │
│   Statistical power: HIGH                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ "Should I reoptimize yearly?" (DEPLOYMENT)                  │
│ → validate_portfolio_oos.py --mode expanding               │
│   Tests: Sequential windows (realistic)                     │
│   Statistical power: LOW                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ "Will THESE weights work for 5 years?" (BUY-AND-HOLD)       │
│ → validate_fixed_portfolio.py                               │
│   Tests: SPECIFIC WEIGHTS, no re-optimization              │
│   Statistical power: MEDIUM-HIGH                            │
└─────────────────────────────────────────────────────────────┘
```

---

## validate_portfolio_oos.py (Strategy Validation)

### Three Modes

**1. Monte Carlo (HIGH stat power - Use First):**
```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode monte_carlo \
    --portfolio both \
    --iterations 100
```

**Output:** n=100, t-test, 95% CI, statistical proof

---

**2. Expanding (Deployment validation):**
```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode expanding \
    --portfolio both
```

**Output:** n=2-4, realistic, but low stat power

---

**3. Rolling (Regime detection):**
```bash
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --mode rolling \
    --portfolio hrp
```

**Output:** n=3-5, checks if works in all regimes

---

### What It Does (Monte Carlo Mode)

```
Iteration 1:  Random window → Optimize → Test → Sharpe 3.2
Iteration 2:  Random window → Optimize → Test → Sharpe 2.9
Iteration 3:  Random window → Optimize → Test → Sharpe 3.5
...
Iteration 100: Random window → Optimize → Test → Sharpe 3.1

Result:
  Avg OOS Sharpe: 3.18 ± 0.42 (95% CI: [3.10, 3.26])
  T-test: p < 0.0001 (HIGHLY SIGNIFICANT)

Statistical proof: Strategy is fundamentally sound
```

### Decision Criteria

| Avg Degradation | Verdict | Action |
|-----------------|---------|--------|
| < 10% | ✅ EXCELLENT | Deploy, reoptimize yearly |
| 10-20% | ✅ ACCEPTABLE | Deploy, monitor |
| 20-30% | ⚠️ CAUTION | Review, possibly use HRP |
| > 30% | 🚨 REJECT | Use HRP or reject |

---

## validate_fixed_portfolio.py (Fixed Weights Validation)

### One-Liner

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3
```

### What It Does

```
Extract weights from log: IAU 15%, NVDA 12%, ...

Period 1: Test on 2020 Q1-Q4 → Sharpe 4.2
Period 2: Test on 2021 Q1-Q4 → Sharpe 3.8 (same weights!)
Period 3: Test on 2022 Q1-Q4 → Sharpe 4.1 (same weights!)
...

Output: Stability (CV), statistical significance
```

### Decision Criteria

| CV | Verdict | Action |
|----|---------|--------|
| < 0.15 | ✅ VERY STABLE | Buy-and-hold 5+ years |
| 0.15-0.30 | ✅ STABLE | Buy-and-hold 2-3 years |
| 0.30-0.50 | ⚠️ MODERATE | Reoptimize yearly |
| > 0.50 | 🚨 UNSTABLE | Reject or quarterly rebalancing |

---

## Typical Workflow

```bash
# 1. Optimize
python portfolio_exploration_global.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2025-01-01 \
    --stage2-target 40

# 2. Test strategy (should I reoptimize?)
python validate_portfolio_oos.py \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --portfolio both

# 3. If max_sharpe wins: Test if you can hold it
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_*.log \
    --portfolio max_sharpe \
    --csv ../../data/your_data.csv \
    --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3

# 4. Decision tree:
#    CV < 0.30 → Buy-and-hold (no rebalancing)
#    CV > 0.30 → Reoptimize yearly
```

---

## Quick Examples

### Test if rebalancing helps

```bash
python validate_portfolio_oos.py \
    --csv data.csv --start 2020-01-01 --end 2026-01-01 \
    --portfolio max_sharpe --train-years 2 --test-months 12
```

### Test portfolio stability

```bash
python validate_fixed_portfolio.py \
    --log logs/portfolio_exploration_20260108_145404.log \
    --portfolio hrp \
    --csv data.csv --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 3
```

### Compare strategies

```bash
python validate_portfolio_oos.py \
    --csv data.csv --start 2020-01-01 --end 2026-01-01 \
    --portfolio both
```

---

**See OOS_VALIDATION_GUIDE.md for complete documentation.**
