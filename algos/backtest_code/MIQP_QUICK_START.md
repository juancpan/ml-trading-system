# MIQP Portfolio Optimization - Quick Start

## What is MIQP Mode?

**MIQP = Mixed-Integer Quadratic Programming**

Optimizes for **exact number of shares** (integers) instead of fractional weights.

**Why this matters:**
```
Continuous mode:  NVDA weight = 0.3456 → How many shares? (unclear)
MIQP mode:        NVDA shares = 125 → Exactly 125 shares (directly tradeable)
```

---

## Installation (One-Time)

```bash
# Install MIQP solvers
pip install cvxpy pyscipopt cylp

# Verify
python -c "import cvxpy as cp; print(cp.installed_solvers())"
# Should show: ['SCIP', 'CBC', 'GLPK', ...]
```

---

## Basic Usage

### **Step 1: Run continuous mode (baseline)**

```bash
python algos/backtest_code/portimization.py \\
  --start 2024-01-01 --end 2025-01-01
```

**Output:** Fractional weights (research/backtesting)

---

### **Step 2: Run MIQP mode (production)**

```bash
python algos/backtest_code/portimization.py \\
  --mode miqp \\
  --budget 50000 \\
  --exchange_rates JPY=0.0067 GBP=1.27 \\
  --lot_sizes 8002.T=100 III.L=1 NVDA=1 \\
  --start 2024-01-01 --end 2025-01-01
```

**Output:** Integer shares (directly tradeable)

---

## Required Parameters for MIQP

| Parameter | Required? | Example |
|-----------|-----------|---------|
| `--mode miqp` | ✅ Yes | Enable MIQP mode |
| `--budget` | ✅ Yes | `--budget 50000` (USD) |
| `--exchange_rates` | ✅ If multi-currency | `--exchange_rates JPY=0.0067 GBP=1.27` |
| `--lot_sizes` | ⚠️ Optional | `--lot_sizes 8002.T=100` (auto-detected if omitted) |
| `--miqp_solver` | ⚠️ Optional | `--miqp_solver SCIP` (default) |
| `--frontier_points` | ⚠️ Optional | `--frontier_points 5` (default: 7) |

---

## Exchange Rates (Get Current Rates)

### **Method 1: yfinance**
```python
import yfinance as yf

# JPY to USD
jpyusd = yf.download('JPYUSD=X', period='1d')['Close'].iloc[-1]
print(f"JPY={jpyusd:.4f}")  # e.g., JPY=0.0067

# GBP to USD
gbpusd = yf.download('GBPUSD=X', period='1d')['Close'].iloc[-1]
print(f"GBP={gbpusd:.4f}")  # e.g., GBP=1.27
```

### **Method 2: Google Search**
- Search: "JPY to USD"
- Result: 1 JPY = 0.0067 USD

### **Method 3: IBKR forex cache**
```bash
cat execution/forex_rates_cache.json
```

### **Quick Reference (2026-01-09):**
```
JPY=0.0067  (¥149 per dollar)
GBP=1.27    (£1 = $1.27)
HKD=0.13    (HK$7.70 per dollar)
EUR=1.05
AUD=0.65
CAD=0.71
SGD=0.74
```

---

## Lot Sizes (Auto-Detected)

**Most lot sizes are auto-detected from ExchangeManager:**
- Tokyo (TSEJ): 100 shares (all stocks)
- Hong Kong (SEHK): 100 shares (default)
- US (NASDAQ): 1 share (no requirement)
- London (LSE): 1 share

**Override if needed:**
```bash
--lot_sizes 1277.HK=2000 1288.HK=1000
```

**Check lot sizes:**
```python
from execution.exchange_manager import ExchangeManager
em = ExchangeManager()

print(em.get_lot_size('8002.T'))   # 100
print(em.get_lot_size('1277.HK'))  # 2000
print(em.get_lot_size('NVDA'))     # 1
```

---

## Complete Examples

### **Example 1: US Stocks Only (Simplest)**

```bash
python algos/backtest_code/portimization.py \\
  --mode miqp \\
  --budget 30000 \\
  --start 2024-01-01 --end 2025-01-01
```

**No exchange_rates needed** (all USD)
**No lot_sizes needed** (all = 1)
**Runtime:** ~20-30 minutes (7 frontier points)

---

### **Example 2: Current Live Portfolio (4 tickers)**

```bash
# Your current portfolio: NVDA, AVGO, 8002.T, III.L

python algos/backtest_code/portimization.py \\
  --mode miqp \\
  --budget 50000 \\
  --lot_sizes 8002.T=100 NVDA=1 AVGO=1 III.L=1 \\
  --exchange_rates JPY=0.0067 GBP=1.27 \\
  --start 2024-01-01 --end 2025-01-01 \\
  --frontier_points 5
```

**Runtime:** ~25 minutes
**Output:** Exact shares for live trading deployment

---

### **Example 3: Hong Kong Portfolio**

```bash
python algos/backtest_code/portimization.py \\
  --mode miqp \\
  --budget 40000 \\
  --lot_sizes 1277.HK=2000 0700.HK=100 \\
  --exchange_rates HKD=0.13 \\
  --start 2024-01-01 --end 2025-01-01
```

**Hong Kong specific lot sizes:**
- 1277.HK (Jiangxi Copper): 2000 shares
- 0700.HK (Tencent): 100 shares
- 1288.HK (ABC): 1000 shares

---

## Interpreting Output

### **MIQP Portfolio Result**

```
Maximum Sharpe MIQP Portfolio:
  Return:      0.2845 (28.45%)
  Volatility:  0.1523 (15.23%)
  Sharpe:      1.8125
  Total Value: $49,847
  Budget Used: 99.7%
  Solver:      SCIP (258.3s)

  Positions (4 assets):
  Ticker          Shares     Price        Currency Value USD   Weight   Lot Size
  --------------- ---------- ------------ -------- ------------ -------- ----------
  NVDA            125        137.45       USD      $17,181      34.5%    -
  AVGO            80         156.20       USD      $12,496      25.0%    -
  8002.T          2100       3758.00      JPY      $352         0.7%     100
  III.L           850        28.35        GBP      $30,604      61.4%    -

  1277.HK: 0 shares (not included - below minimum)
```

**Interpretation:**
- **4 positions** from 5 possible (1277.HK excluded by optimizer)
- **Tokyo stock:** 2100 shares = 21 lots (100-share lots)
- **Budget used:** 99.7% (fully invested)
- **Solve time:** 4.3 minutes (acceptable)

---

### **Efficient Frontier (MIQP)**

```
MIQP Efficient Frontier (7 points)
Point 1: Return 0.1234, Vol 0.0987, Sharpe 0.9876 (312.5s)
Point 2: Return 0.1567, Vol 0.1123, Sharpe 1.2345 (285.2s)
Point 3: Return 0.1890, Vol 0.1289, Sharpe 1.4567 (298.7s)
...
Point 7: Return 0.2845, Vol 0.1523, Sharpe 1.8125 (325.1s)

Total solve time: 2105.3s (35.1 minutes)
```

**Use:** Pick point with desired risk/return trade-off

---

## Common Errors & Solutions

### **"cvxpy not installed"**
```bash
pip install cvxpy pyscipopt cylp
```

### **"Missing exchange rate for JPY"**
```bash
--exchange_rates JPY=0.0067
```

### **"Lot size too restrictive"**

Increase budget or remove asset:
```bash
--budget 100000  # Increase budget
# OR remove expensive lot-size assets
```

### **"Budget allows ~2 positions"**

Accept concentration or increase budget:
```bash
--budget 50000  # Minimum $50k for international + lot sizes
```

### **Solve takes > 10 minutes per point**

Reduce frontier points or use Gurobi:
```bash
--frontier_points 5  # Fewer points
--miqp_solver GUROBI  # Faster solver (if installed)
```

---

## Quick Decision Tree

```
Do you need exact share counts for live trading?
├─ No → Use continuous mode (faster, research)
└─ Yes → Use MIQP mode
    ├─ US stocks only?
    │   └─ python portimization.py --mode miqp --budget 50000
    ├─ International stocks?
    │   └─ python portimization.py --mode miqp --budget 50000 \\
    │        --exchange_rates JPY=0.0067 GBP=1.27 \\
    │        --lot_sizes 8002.T=100
    └─ Need efficient frontier?
        └─ Add --frontier_points 7
```

---

**Ready to use! Start with continuous mode, then upgrade to MIQP for production deployment.**
