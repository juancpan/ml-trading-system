# Model Selection & Deployment Workflow Guide

**Complete guide for your enhanced WFOV v2 validation framework**

---

## 🎯 **Your Profile: Answer A (Maximum Returns)**

**Trading Style:**
- Solo trader with real capital at risk
- Daily monitoring via Telegram + portfolio oversight dashboard
- Can redeploy models in < 1 day if needed
- Portfolio diversified (4 tickers, 3 currencies)
- Risk tolerance: High (willing to accept regime risk for maximum returns)

**Validation Philosophy:**
- Screen out obvious garbage (p > 0.20, negative Sharpe)
- Among valid models, **deploy highest Sharpe**
- Monitor regime risks, don't auto-reject
- React fast if live performance degrades

---

## 🚀 **Quick Start: 3 Workflow Options**

### **Option 1: Automated (Recommended)**

**Single command for complete workflow:**

```bash
# Full workflow: train → validate → rank → recommend
python scripts/model_selection_workflow.py --ticker NVDA

# Quick mode (20 iterations, ~5 minutes total)
python scripts/model_selection_workflow.py --ticker NVDA --quick

# Batch mode (all tickers)
python scripts/model_selection_workflow.py --batch
```

**What happens:**
1. Trains 4 model types (lstm, svm_optimized, xgb_optimized, li_reg)
2. Validates each with WFOV Monte Carlo (50 iterations)
3. Ranks by score = Sharpe + small adjustments
4. Shows deployment commands for winner

**Output:**
```
🚀 TIER 1 (DEPLOY):  SVM_OPTIMIZED
   Sharpe: 1.35 | p=0.003 | Score: 1.55
   ⚠️  Regime Risk: 4.2x better in bull markets
   → Deploy and monitor if regime shifts

Deployment commands:
1. python algos/backtest_code/run_backtest_optimized.py --model_name svm_optimized --ticker NVDA --start 2020-01-01 --end 2025-12-03
2. python deploy_models.py
3. python validate_config.py
```

---

### **Option 2: Manual Control**

**Step-by-step with visibility:**

```bash
# STEP 1: Train candidates (your choice which models)
python algos/backtest_code/run_backtest_optimized.py --model_name lstm --ticker NVDA --lookback_days 1260
python algos/backtest_code/run_backtest_optimized.py --model_name svm_optimized --ticker NVDA --lookback_days 1260
python algos/backtest_code/run_backtest_optimized.py --model_name xgb_optimized --ticker NVDA --lookback_days 1260

# STEP 2: Validate with WFOV (run in parallel)
for model in lstm svm_optimized xgb_optimized; do
    python -m algos.wfov.wfov_runner \\
        --mode monte_carlo \\
        --model_name $model \\
        --ticker NVDA \\
        --iterations 50 \\
        --seed 42 &
done
wait

# STEP 3: Review results (automated ranking)
python -m algos.wfov.model_ranker --ticker NVDA

# STEP 4: Pick best from TIER 1, deploy
# (Use commands shown by ranker)
```

---

### **Option 3: Expert Mode (Maximum Flexibility)**

**Full control over every parameter:**

```bash
# 1. Train with custom hyperparameters
python algos/backtest_code/run_backtest_optimized.py \\
    --model_name lstm \\
    --ticker NVDA \\
    --start 2018-01-01 \\
    --end 2025-12-03

# 2. Custom WFOV validation
python -m algos.wfov.wfov_runner \\
    --mode monte_carlo \\
    --model_name lstm \\
    --ticker NVDA \\
    --iterations 200 \\
    --min_lookback_days 730 \\
    --max_lookback_days 2555 \\
    --min_train_split 0.6 \\
    --max_train_split 0.85 \\
    --max_workers 8 \\
    --seed 42

# 3. Custom ranking thresholds
python -m algos.wfov.model_ranker \\
    --ticker NVDA \\
    --p_value_threshold 0.20 \\
    --regime_dependency_threshold 10.0  # Very high tolerance

# 4. Manual inspection of results
cat algos/wfov/results/summaries/montec_lstm_NVDA_*.json | jq '.statistical_rigor'

# 5. Deploy based on your judgment
python algos/backtest_code/run_backtest_optimized.py --model_name lstm --ticker NVDA --lookback_days 2555
python deploy_models.py
```

---

## 📊 **Model Ranking System**

### **3-Tier Categorization**

**TIER 1: 🚀 DEPLOY**
- High Sharpe (> 0.8) OR moderate Sharpe (> 0.25) AND p < 0.05
- No hard red flags (negative Sharpe, p > 0.20)
- **Action: Deploy highest Sharpe in this tier**
- **Flags shown but not disqualifying:**
  - Regime dependency → Monitor closely
  - High variance → Watch for instability
  - Deflated Sharpe lower → Expect conservative live performance

**TIER 2: ⚠️  REVIEW**
- Moderate Sharpe (0.25-0.8) AND marginal significance (0.05 < p < 0.15)
- OR high Sharpe with concerning flags
- **Action: Your decision**
  - Deploy if you understand and accept risks
  - Or use buy-and-hold for this ticker

**TIER 3: ❌ REJECT**
- p > 0.20 (likely noise)
- Negative Sharpe (loses money)
- Win rate < 45% (poor directional accuracy)
- **Action: Don't deploy**
  - Retrain with different hyperparameters
  - Try different features
  - Consider buy-and-hold

---

## 🔧 **Key Tools & When to Use**

| Tool | Purpose | When to Use | Time |
|------|---------|-------------|------|
| **model_selection_workflow.py** | Full automation | Monthly model updates | 15-30 min |
| **wfov_runner (Monte Carlo)** | Quick validation | Testing new model types | 5-15 min |
| **wfov_runner (Walk-Forward)** | Final OOS check | Before major deployment | 10-20 min |
| **model_ranker.py** | Compare models | After running multiple WFOVs | 10 sec |
| **compare_models_quick.py** | Quick comparison | Daily check of recent results | 5 sec |

---

## 📝 **Complete Workflows by Use Case**

### **Use Case 1: New Ticker Addition**

**Scenario:** Adding a new stock to portfolio

```bash
# 1. Run automated workflow
python scripts/model_selection_workflow.py --ticker MSFT

# 2. Review recommendation
cat algos/wfov/deployment_recommendations/deployment_recommendation_MSFT_*.txt

# 3. If TIER 1 found, deploy
python deploy_models.py

# 4. Update config manually if needed
vim execution/config.py  # Add MSFT to TARGET_ALLOCATION

# 5. Validate and start
python validate_config.py
python execution/main.py
```

---

### **Use Case 2: Model Refresh (Quarterly)**

**Scenario:** Updating existing ticker models

```bash
# For each live ticker
for ticker in NVDA AVGO 8002.T III.L; do
    # Quick screen with existing models
    python scripts/model_selection_workflow.py --ticker $ticker --quick
done

# Review all recommendations
ls algos/wfov/deployment_recommendations/deployment_recommendation_*_$(date +%Y%m%d).txt

# Deploy if better models found
python deploy_models.py
python validate_config.py
```

---

### **Use Case 3: Model Underperforming in Live Trading**

**Scenario:** NVDA model Sharpe dropped from 1.2 to 0.3 over 5 days

```bash
# Quick validation of alternative models
python -m algos.wfov.wfov_runner \\
    --mode monte_carlo \\
    --model_name lstm \\  # Try different model
    --ticker NVDA \\
    --iterations 30 \\
    --quick

# Compare to current model
python -m algos.wfov.model_ranker --ticker NVDA

# If better model found, redeploy immediately
python algos/backtest_code/run_backtest_optimized.py --model_name lstm --ticker NVDA --lookback_days 1825
python deploy_models.py
python execution/main.py
```

---

### **Use Case 4: Hyperparameter Tuning**

**Scenario:** Testing LSTM with different sequence lengths

```bash
# 1. Train with different hyperparameters
python algos/backtest_code/run_backtest_optimized.py \\
    --model_name lstm \\
    --ticker NVDA \\
    --lookback_days 1260 \\
    --lstm_sequence_length 5

python algos/backtest_code/run_backtest_optimized.py \\
    --model_name lstm \\
    --ticker NVDA \\
    --lookback_days 1260 \\
    --lstm_sequence_length 10

# 2. Validate both
for seq_len in 5 10; do
    python -m algos.wfov.wfov_runner \\
        --mode monte_carlo \\
        --model_name lstm \\
        --ticker NVDA \\
        --iterations 50 &
done
wait

# 3. Compare (ranker shows both, picks best)
python -m algos.wfov.model_ranker --ticker NVDA
```

---

## 🎓 **Decision Framework**

### **When to Deploy a Model**

```
┌─────────────────────────────────────────────┐
│ Is p-value < 0.15?                          │
│   NO  → ❌ REJECT (likely noise)            │
│   YES → Continue                            │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Is Sharpe > 0.25?                           │
│   NO  → ❌ REJECT (too weak)                │
│   YES → Continue                            │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Is this the HIGHEST Sharpe among valid?     │
│   NO  → Try another model                   │
│   YES → 🚀 DEPLOY!                          │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Check regime dependency                     │
│   < 3x   → ✅ No special monitoring needed  │
│   3-5x   → ⚠️  Monitor daily (set alerts)   │
│   > 5x   → ⚠️  Monitor closely (high risk)  │
└─────────────────────────────────────────────┘
```

### **Monitoring After Deployment**

**Daily Checks (automated via Telegram):**
- [ ] Live Sharpe > 0.3 (rolling 5-day)
- [ ] Drawdown < backtest max DD × 1.5
- [ ] Win rate > 45%

**Weekly Checks (portfolio oversight dashboard):**
- [ ] Compare live Sharpe vs backtest expectation
- [ ] Check if market regime changed
- [ ] Review if model still in top tier

**Quarterly Actions:**
- [ ] Rerun WFOV on current model
- [ ] Test new candidate models
- [ ] Redeploy if better option available

---

## 🛠️ **Advanced: Custom Thresholds**

**Adjust ranking criteria based on your risk tolerance:**

```bash
# More aggressive (deploy more models)
python -m algos.wfov.model_ranker \\
    --ticker NVDA \\
    --p_value_threshold 0.20 \\           # Accept weaker significance
    --min_sharpe_threshold 0.15 \\        # Lower Sharpe bar
    --regime_dependency_threshold 10.0    # Tolerate extreme regime risk

# More conservative (deploy fewer models)
python -m algos.wfov.model_ranker \\
    --ticker NVDA \\
    --p_value_threshold 0.05 \\           # Require strong significance
    --min_sharpe_threshold 0.5 \\         # Higher Sharpe bar
    --regime_dependency_threshold 2.0     # Low tolerance for regime risk
```

---

## 📚 **Complete Command Reference**

### **Training:**
```bash
# Single model
python algos/backtest_code/run_backtest_optimized.py \\
    --model_name lstm --ticker NVDA --lookback_days 1260

# With custom dates
python algos/backtest_code/run_backtest_optimized.py \\
    --model_name svm_optimized --ticker SPY \\
    --start 2020-01-01 --end 2025-12-03
```

### **Validation:**
```bash
# Monte Carlo (quick screening)
python -m algos.wfov.wfov_runner \\
    --mode monte_carlo \\
    --model_name lstm --ticker NVDA --iterations 50

# Walk-Forward Expanding (final check before deployment)
python -m algos.wfov.wfov_runner \\
    --mode walk_forward_expanding \\
    --model_name lstm --ticker NVDA \\
    --initial_train_days 1260 --test_days 252 --step_days 252

# Walk-Forward Rolling (regime change detection)
python -m algos.wfov.wfov_runner \\
    --mode walk_forward_rolling \\
    --model_name lstm --ticker NVDA \\
    --window_size 1260 --test_days 252 --step_days 252
```

### **Comparison & Ranking:**
```bash
# Rank models for single ticker
python -m algos.wfov.model_ranker --ticker NVDA

# Multiple tickers
python -m algos.wfov.model_ranker --tickers NVDA AVGO SPY

# Quick comparison (recent results only)
python scripts/compare_models_quick.py --ticker NVDA
```

### **Deployment:**
```bash
# Deploy selected models
python deploy_models.py

# Verify deployment
python validate_config.py --verbose

# Check manifest
cat execution/strategy_models/deployment_manifest.json
```

---

## 🔍 **Understanding the Output**

### **WFOV Console Output (v2)**

```
Mode: MONTE CARLO
Model: lstm | Ticker: NVDA

Results: 96/100 successful (96.0%)

📊 STATISTICAL RIGOR
────────────────────────────────────────────
  Sharpe Significance:     ✓ SIGNIFICANT (p < 0.05)
  95% Confidence Interval: [0.65, 1.15]
  Deflated Sharpe Ratio:   0.72 (vs observed 0.90)
  Interpretation:          Moderate: DSR > 0.5 (likely not luck)

🎯 REGIME ANALYSIS
────────────────────────────────────────────
  Regime Distribution:
    normal      : 32 iterations (33.3%)
    high_vol    : 30 iterations (31.3%)
    low_vol     : 34 iterations (35.4%)

  Performance by Regime (Sharpe Ratio):
    ✓ Normal     :  0.88 ± 0.25 (n=32)
    ○ High_vol   :  0.42 ± 0.35 (n=30)
    ✓ Low_vol    :  1.35 ± 0.18 (n=34)

  ⚠️  Strategy performs 3.2x better in low_vol markets (regime-dependent risk!)
```

**How to interpret:**
- **Significance**: p < 0.05 → ✅ Not due to luck, deploy
- **Deflated Sharpe**: 0.72 → Expect ~0.7 in live trading (vs 0.9 optimistic backtest)
- **Regime dependency**: 3.2x → ⚠️  Flag for monitoring, but still deploy (Answer A)

---

### **Model Ranker Output**

```
MODEL RANKING FOR NVDA
─────────────────────────────────────────────────────────────────
Rank   Model                Tier       Score    Sharpe   p-value
─────────────────────────────────────────────────────────────────
1      SVM_OPTIMIZED        🚀 DEPLOY  1.550    1.350    0.0030
2      LSTM                 🚀 DEPLOY  1.320    1.200    0.0080
3      XGB_OPTIMIZED        ⚠️  REVIEW  0.820    0.850    0.1200
4      LI_REG               ❌ REJECT   0.000    0.450    0.3500
─────────────────────────────────────────────────────────────────

DEPLOYMENT RECOMMENDATION
═════════════════════════════════════════════════════════════════

🚀 DEPLOY: SVM_OPTIMIZED
   Sharpe: 1.350 (highest among significant models)
   Statistical Sig: ✓ SIGNIFICANT (p=0.003)
   Win Rate: 58.5%

   ⚠️  MONITOR: 2.8x better in normal markets
      → Set alert: Switch model if 5-day Sharpe < 0.3
```

**Decision:**
- Deploy SVM_OPTIMIZED (highest score in TIER 1)
- Ignore regime warning (you can monitor and switch quickly)
- Expect ~1.2-1.3 Sharpe in live trading (deflated estimate)

---

## 💡 **Best Practices**

### **For Rapid Iteration (Your Style):**

1. **Use quick mode for initial screening:**
   ```bash
   python scripts/model_selection_workflow.py --ticker NVDA --quick
   # 20 iterations × 4 models = 5 minutes total
   ```

2. **Deploy highest TIER 1 model immediately:**
   - Don't overthink regime warnings
   - You monitor daily, can switch in 24 hours

3. **Set monitoring alerts:**
   ```python
   # In Telegram bot
   if live_sharpe_5day < 0.3:
       alert("NVDA model underperforming - consider switching")
   ```

4. **Quarterly full validation:**
   ```bash
   # Every 3 months, run comprehensive validation
   python scripts/model_selection_workflow.py --batch --iterations 100
   ```

### **When to Be Conservative:**

Use walk-forward validation if:
- Deploying with leverage > 3x
- Account size > $500K
- New ticker (no historical deployment experience)
- Model shows >5x regime dependency

```bash
# Run walk-forward before deploying
python -m algos.wfov.wfov_runner \\
    --mode walk_forward_expanding \\
    --model_name lstm --ticker NEW_TICKER \\
    --initial_train_days 1260 --test_days 252 --step_days 252

# If walk-forward confirms robustness → Deploy
# If walk-forward shows degradation → Reject or reduce allocation
```

---

## 🎯 **Flexibility Options**

You have **full control** at every level:

**Level 1: Thresholds (model_ranker.py)**
- Adjust p-value, Sharpe, regime dependency thresholds
- Make ranking more/less aggressive

**Level 2: WFOV Parameters**
- Iterations: 20 (quick) to 200 (comprehensive)
- Lookback range: Conservative (1-2 years) to aggressive (1-5 years)
- Embargo: 0% (none) to 5% (strict)

**Level 3: Manual Override**
- Ranker says "TIER 2 REVIEW" → You can still deploy if you accept risks
- Ranker says "TIER 1 DEPLOY #2" → You can deploy #2 instead of #1
- Complete freedom to ignore recommendations

**Level 4: Direct Result Analysis**
- Read JSON files directly
- Apply your own scoring formula
- Make decisions based on domain knowledge

---

## 📈 **Expected Outcomes**

### **With Enhanced WFOV v2:**

**Garbage Filtering:**
- 70-90% of models auto-rejected (p > 0.20, negative Sharpe)
- Saves you from deploying obvious failures

**Among Valid Models:**
- Deploy highest Sharpe (maximize returns)
- Get visibility into risks (regime dependency, variance)
- Make informed decision, not blind deployment

**Live Trading:**
- Higher expected returns (deploy highest performer)
- Better risk awareness (know regime dependencies)
- Faster reaction (clear monitoring metrics)

**What You AVOID:**
- Deploying pure noise (p > 0.20 filtered)
- Deploying money-losing models (negative Sharpe filtered)
- Blind to risks (regime analysis highlights)

**What You GAIN:**
- Maximum returns among validated models (highest Sharpe selected)
- Statistical confidence (p-values, CIs, deflated Sharpe)
- Actionable risk flags (monitor closely, not auto-reject)

---

## 🚀 **TL;DR: Your New Workflow**

**One command to rule them all:**

```bash
# Update model for NVDA (complete workflow)
python scripts/model_selection_workflow.py --ticker NVDA

# Output tells you:
#   - Which model to deploy (highest Sharpe among significant)
#   - What to monitor (regime risks, variance)
#   - Exact deployment commands
#   - Expected live performance (deflated Sharpe)

# Then deploy:
python deploy_models.py && python validate_config.py && python execution/main.py
```

**Philosophy:**
- Statistical rigor **INFORMS** decisions, doesn't **MAKE** them
- You still deploy highest Sharpe (maximize returns)
- But now you know what to watch for (regime shifts, variance)
- React fast when risks materialize (switch model in 1 day)

This is **Answer A with statistical intelligence**, not **Answer B conservatism**.
