#!/usr/bin/env python3
"""
Pre-deployment preflight check.

Runs 6 checks before live trading. Each check is independent and fail-fast.
This script is READ-ONLY. It inspects state, it does not modify it.

Usage:
    python execution/preflight_check.py --nav 10500
    python execution/preflight_check.py --nav 10500 --with-ibkr
    python execution/preflight_check.py --nav 10500 --verbose

Exit codes:
    0 = all checks PASS (warnings allowed)
    1 = one or more checks FAIL
    2 = invalid invocation (bad args, missing files, etc.)

Checks:
    [A] Weights sanity         -- JSON + config.py consistency, sum=1.0
    [B] Data freshness         -- Parquet recency, outlier scan (GVR.IR class)
    [C] Covariance shrinkage   -- LedoitWolf not degenerate
    [D] ML model freshness     -- Model files exist, not stale, scalers paired
    [E] IBKR probe (optional)  -- Raw ibapi connect, account summary
    [F] Min-notional viability -- Share rounding drift check

Background: this script was created after the 2026-04-11 incident where
an unadjusted reverse split on GVR.IR produced an 11,892% single-day return,
collapsed LedoitWolf to shrinkage=1.0 (identity covariance), and made HRP
output equal weights across all 18 tickers. Check B is the permanent
detection fix for that class of data corruption. Check C is the detection
fix for the covariance-level symptom.

Author: Tier 1 hardening plan (2026-04-21)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env into os.environ (only for keys not already set). Robust whether
# run directly or as a module. See execution/env_loader.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import env_loader  # noqa: F401  (side-effect: populates os.environ)
except Exception:
    pass

# ============================================================================
# THRESHOLDS (tune here; intentionally not hidden in classes)
# ============================================================================

MAX_SHRINKAGE = 0.50
WARN_SHRINKAGE = 0.30
MIN_CONDITION_NUMBER = 3.0

MAX_SINGLE_DAY_RETURN = 0.30   # abs log-return; GVR.IR was 4.79
MAX_5DAY_RETURN = 0.50
MAX_DATA_AGE_DAYS = 5          # calendar days since last parquet row

MAX_MODEL_AGE_DAYS = 120

NOTIONAL_DRIFT_FAIL = 0.50     # share rounding drift vs target weight
NOTIONAL_DRIFT_WARN = 0.20

WEIGHTS_SUM_TOLERANCE = 1e-6
MAX_WEIGHT_CAP = 0.13          # slight buffer above configured 0.12564... BIL
MAX_WEIGHT_CAP_HARD = 0.15     # anything above this is definitely wrong

COVARIANCE_LOOKBACK_DAYS = 504  # ~2 years of trading days

# Exchange-suffix → currency map for Check F notional conversion.
# Portfolio tickers are drawn from yfinance-style suffixes. Parquet prices
# are stored in the exchange's local currency (not translated to USD).
# To compare deployed USD notional vs ticker price, we need to divide the
# price by XXX_TO_USD to get price in USD.
#
# Tickers without a suffix (e.g. 'BIL', 'TLT', 'WELL') are assumed USD.
EXCHANGE_SUFFIX_CURRENCY = {
    ".MC": "EUR",   # Madrid (Bolsa de Madrid)
    ".LS": "EUR",   # Euronext Lisbon
    ".MI": "EUR",   # Borsa Italiana (Milan)
    ".VI": "EUR",   # Wiener Börse (Vienna)
    ".IR": "EUR",   # Euronext Dublin (Ireland)
    ".AS": "EUR",   # Euronext Amsterdam
    ".BR": "EUR",   # Euronext Brussels
    ".PA": "EUR",   # Euronext Paris
    ".DE": "EUR",   # XETRA / Frankfurt
    ".F":  "EUR",   # Frankfurt Börse
    ".BD": "HUF",   # Budapest Stock Exchange
    ".TO": "CAD",   # Toronto Stock Exchange
    ".L":  "GBP",   # London Stock Exchange (note: some are pence — out of scope)
    ".ST": "SEK",   # Stockholm
    ".OL": "NOK",   # Oslo
    ".CO": "DKK",   # Copenhagen
    ".WA": "PLN",   # Warsaw
    ".PR": "CZK",   # Prague
    ".RO": "RON",   # Bucharest
    ".SW": "CHF",   # SIX Swiss
    ".HE": "EUR",   # Helsinki
    ".T":  "JPY",   # Tokyo
    ".HK": "HKD",   # Hong Kong
    ".AX": "AUD",   # ASX
    ".SI": "SGD",   # Singapore
    ".SR": "SAR",   # Tadawul (Saudi)
    ".AE": "AED",   # Dubai
    ".TA": "ILS",   # Tel Aviv
    ".JO": "ZAR",   # Johannesburg
}

# ============================================================================
# PATH DISCOVERY
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_DATA_DIR = REPO_ROOT / "data" / "market_data"
TICKER_UNIVERSE_PATH = MARKET_DATA_DIR / "ticker_universe.json"
STRATEGY_MODELS_DIR = REPO_ROOT / "execution" / "strategy_models"
WEIGHTS_JSON_PATH = (
    REPO_ROOT
    / "algos"
    / "backtest_code"
    / "data"
    / "portfolio_weights_hrp_H2_2026.json"
)
CONFIG_PY_PATH = REPO_ROOT / "execution" / "config.py"

# ============================================================================
# COLOR / FORMAT HELPERS (minimal, no deps)
# ============================================================================

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


PASS = _c("32;1", "PASS")
WARN = _c("33;1", "WARN")
FAIL = _c("31;1", "FAIL")
SKIP = _c("37;1", "SKIP")


def _line(name: str, status: str, detail: str = "") -> None:
    dots = "." * max(3, 30 - len(name))
    sys.stdout.write(f"  [{name}] {dots} {status}")
    if detail:
        sys.stdout.write(f"  ({detail})")
    sys.stdout.write("\n")
    sys.stdout.flush()


# ============================================================================
# CHECK A — WEIGHTS SANITY
# ============================================================================


def check_a_weights(verbose: bool = False) -> tuple[bool, dict, list[str]]:
    """
    Returns (pass, {weights, tickers}, warnings).

    Verifies:
      - JSON loads, is dict of ticker->float
      - Sum within tolerance of 1.0
      - Count matches TARGET_ALLOCATION keys in config.py
      - Cross-match to ASSET_SPECIFIC_CONFIGS keys
      - No weight exceeds hard cap
      - JSON and TARGET_ALLOCATION values agree to 1e-9
    """
    warnings: list[str] = []

    if not WEIGHTS_JSON_PATH.exists():
        print(f"  [A] FAIL: weights JSON not found: {WEIGHTS_JSON_PATH}")
        return False, {}, warnings

    try:
        with open(WEIGHTS_JSON_PATH) as f:
            weights = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [A] FAIL: weights JSON invalid: {e}")
        return False, {}, warnings

    if not isinstance(weights, dict) or not weights:
        print("  [A] FAIL: weights is not a non-empty dict")
        return False, {}, warnings

    for k, v in weights.items():
        if not isinstance(v, (int, float)):
            print(f"  [A] FAIL: {k} weight is not numeric: {type(v).__name__}")
            return False, {}, warnings

    s = sum(weights.values())
    if abs(s - 1.0) >= WEIGHTS_SUM_TOLERANCE:
        print(f"  [A] FAIL: weights sum={s:.10f}, drift={abs(s-1.0):.2e} >= {WEIGHTS_SUM_TOLERANCE:.0e}")
        return False, {}, warnings

    max_w = max(weights.values())
    if max_w > MAX_WEIGHT_CAP_HARD:
        print(f"  [A] FAIL: max weight {max_w:.4f} > hard cap {MAX_WEIGHT_CAP_HARD}")
        return False, {}, warnings
    if max_w > MAX_WEIGHT_CAP:
        warnings.append(f"max weight {max_w:.4f} exceeds soft cap {MAX_WEIGHT_CAP}")

    # Parse config.py to cross-check
    if not CONFIG_PY_PATH.exists():
        print(f"  [A] FAIL: config.py not found: {CONFIG_PY_PATH}")
        return False, {}, warnings

    with open(CONFIG_PY_PATH) as f:
        cfg_text = f.read()

    m_ta = re.search(r"TARGET_ALLOCATION\s*=\s*\{([^}]+)\}", cfg_text, re.S)
    if not m_ta:
        print("  [A] FAIL: could not find TARGET_ALLOCATION in config.py")
        return False, {}, warnings

    ta_pairs = re.findall(r'"([^"]+)":\s*([\d.eE+\-]+)', m_ta.group(1))
    target_alloc = {k: float(v) for k, v in ta_pairs}

    if set(weights.keys()) != set(target_alloc.keys()):
        only_json = set(weights) - set(target_alloc)
        only_cfg = set(target_alloc) - set(weights)
        print(f"  [A] FAIL: JSON vs TARGET_ALLOCATION key mismatch")
        if only_json:
            print(f"      only in JSON:   {sorted(only_json)}")
        if only_cfg:
            print(f"      only in config: {sorted(only_cfg)}")
        return False, {}, warnings

    for k in weights:
        if abs(weights[k] - target_alloc[k]) > 1e-9:
            print(
                f"  [A] FAIL: value drift on {k}: "
                f"json={weights[k]:.15f} vs config={target_alloc[k]:.15f}"
            )
            return False, {}, warnings

    # ASSET_SPECIFIC_CONFIGS keys must match
    m_asc = re.search(r"ASSET_SPECIFIC_CONFIGS\s*=\s*\{(.+?)^\}", cfg_text, re.S | re.M)
    if not m_asc:
        print("  [A] FAIL: could not find ASSET_SPECIFIC_CONFIGS in config.py")
        return False, {}, warnings

    asc_keys = set(re.findall(r'^\s*"([A-Z0-9._-]+)":\s*\{', m_asc.group(1), re.M))
    if asc_keys != set(weights.keys()):
        only_weights = set(weights) - asc_keys
        only_asc = asc_keys - set(weights)
        print(f"  [A] FAIL: JSON vs ASSET_SPECIFIC_CONFIGS key mismatch")
        if only_weights:
            print(f"      only in weights: {sorted(only_weights)}")
        if only_asc:
            print(f"      only in ASC:     {sorted(only_asc)}")
        return False, {}, warnings

    detail = f"{len(weights)} tickers, sum={s:.10f}"
    if verbose:
        for k, v in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"      {k:20s}  {v*100:7.4f}%")
    _line("A] Weights sanity           ", PASS, detail)
    return True, {"weights": weights, "tickers": list(weights.keys())}, warnings


# ============================================================================
# CHECK B — DATA FRESHNESS & OUTLIERS
# ============================================================================


def _resolve_parquet_path(ticker: str, reverse_map: dict) -> Path | None:
    """Resolve portfolio ticker to parquet file path."""
    # Try direct (portfolio uses yfinance names for most)
    p = MARKET_DATA_DIR / f"{ticker}.parquet"
    if p.exists():
        return p
    # Try reverse map: display name -> yfinance ticker
    yf_ticker = reverse_map.get(ticker)
    if yf_ticker:
        p = MARKET_DATA_DIR / f"{yf_ticker}.parquet"
        if p.exists():
            return p
    return None


def check_b_data(tickers: list[str], verbose: bool = False) -> tuple[bool, dict, list[str]]:
    """
    Returns (pass, {prices, returns}, warnings).
    """
    warnings: list[str] = []
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"  [B] FAIL: missing dependency: {e}")
        return False, {}, warnings

    if not TICKER_UNIVERSE_PATH.exists():
        print(f"  [B] FAIL: ticker_universe.json not found: {TICKER_UNIVERSE_PATH}")
        return False, {}, warnings

    with open(TICKER_UNIVERSE_PATH) as f:
        tu = json.load(f)
    tickers_map = tu.get("tickers", {})
    reverse_map = {v: k for k, v in tickers_map.items()}

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_cutoff = now - timedelta(days=MAX_DATA_AGE_DAYS)

    prices: dict[str, "pd.Series"] = {}
    failures: list[str] = []
    per_ticker_stats: list[tuple[str, str, float, float, float]] = []

    for t in tickers:
        p = _resolve_parquet_path(t, reverse_map)
        if p is None:
            failures.append(f"{t}: parquet not found")
            continue
        df = pd.read_parquet(p)
        if df.empty:
            failures.append(f"{t}: parquet empty")
            continue
        col = "adj_close" if "adj_close" in df.columns else "close"
        if col not in df.columns:
            failures.append(f"{t}: no {col} column")
            continue

        last_date = df.index[-1]
        if hasattr(last_date, "to_pydatetime"):
            last_date = last_date.to_pydatetime()
        if last_date.tzinfo is not None:
            last_date = last_date.replace(tzinfo=None)

        age_days = (now - last_date).days
        if age_days > MAX_DATA_AGE_DAYS:
            failures.append(
                f"{t}: last row {last_date.date()} is {age_days}d old (>{MAX_DATA_AGE_DAYS})"
            )
            continue

        s = df[col].dropna().iloc[-60:]
        if len(s) < 10:
            failures.append(f"{t}: only {len(s)} non-NaN prices in last 60d")
            continue

        log_ret = np.log(s / s.shift(1)).dropna()
        if len(log_ret) == 0:
            failures.append(f"{t}: no returns computable")
            continue

        max_abs_1d = float(log_ret.abs().max())
        if max_abs_1d > MAX_SINGLE_DAY_RETURN:
            failures.append(
                f"{t}: max |1d log-ret| = {max_abs_1d:.4f} > {MAX_SINGLE_DAY_RETURN} "
                f"(outlier detected — check for unadjusted corporate action)"
            )
            continue

        # 5-day rolling sum of log returns
        if len(log_ret) >= 5:
            max_abs_5d = float(log_ret.rolling(5).sum().abs().max())
        else:
            max_abs_5d = max_abs_1d
        if max_abs_5d > MAX_5DAY_RETURN:
            failures.append(
                f"{t}: max |5d rolling log-ret| = {max_abs_5d:.4f} > {MAX_5DAY_RETURN}"
            )
            continue

        prices[t] = df[col]
        per_ticker_stats.append(
            (t, last_date.strftime("%Y-%m-%d"), float(s.iloc[-1]), max_abs_1d, max_abs_5d)
        )

    if failures:
        print(f"  [B] FAIL:")
        for f in failures:
            print(f"      {f}")
        return False, {}, warnings

    # Summary: worst offender
    worst_1d = max(per_ticker_stats, key=lambda x: x[3])
    detail = f"max |1d|={worst_1d[3]:.3f} on {worst_1d[0]}"
    if verbose:
        print(f"      {'ticker':15s}  {'last_date':12s}  {'last_px':>10s}  {'|1d|max':>8s}  {'|5d|max':>8s}")
        for t, d, px, m1, m5 in per_ticker_stats:
            print(f"      {t:15s}  {d}  {px:>10.4f}  {m1:>8.4f}  {m5:>8.4f}")
    _line("B] Data freshness           ", PASS, detail)
    return True, {"prices": prices, "reverse_map": reverse_map}, warnings


# ============================================================================
# CHECK C — COVARIANCE SHRINKAGE GUARD
# ============================================================================


def check_c_covariance(
    prices: dict, tickers: list[str], verbose: bool = False
) -> tuple[bool, dict, list[str]]:
    warnings: list[str] = []
    try:
        import numpy as np
        import pandas as pd
        from sklearn.covariance import LedoitWolf
    except ImportError as e:
        print(f"  [C] FAIL: missing dependency: {e}")
        return False, {}, warnings

    if not prices:
        print("  [C] FAIL: no price data from check B")
        return False, {}, warnings

    # Build aligned price matrix, then log returns
    pdf = pd.DataFrame(prices)
    pdf = pdf.ffill().bfill()
    pdf = pdf.dropna(how="all")
    # Use last N trading days
    pdf = pdf.iloc[-(COVARIANCE_LOOKBACK_DAYS + 1) :]
    log_ret = np.log(pdf / pdf.shift(1)).dropna()
    if len(log_ret) < 60:
        print(f"  [C] FAIL: only {len(log_ret)} return rows (need >= 60)")
        return False, {}, warnings

    # Drop any columns with zero variance (would NaN the correlation)
    zero_var = [c for c in log_ret.columns if log_ret[c].std() == 0]
    if zero_var:
        warnings.append(f"zero-variance columns dropped: {zero_var}")
        log_ret = log_ret.drop(columns=zero_var)

    lw = LedoitWolf()
    try:
        lw.fit(log_ret.values)
    except Exception as e:
        print(f"  [C] FAIL: LedoitWolf fit raised: {e}")
        return False, {}, warnings

    shrinkage = float(lw.shrinkage_)
    cond = float(np.linalg.cond(lw.covariance_))

    detail = f"shrinkage={shrinkage:.4f}, cond={cond:.2f}, rows={len(log_ret)}"

    if shrinkage > MAX_SHRINKAGE:
        print(f"  [C] FAIL: shrinkage {shrinkage:.4f} > {MAX_SHRINKAGE} "
              f"(covariance degenerate — check for outliers, data bugs, or near-identical tickers)")
        return False, {}, warnings

    if cond < MIN_CONDITION_NUMBER:
        print(f"  [C] FAIL: condition number {cond:.2f} < {MIN_CONDITION_NUMBER} "
              f"(covariance too homogeneous — likely shrunk to identity)")
        return False, {}, warnings

    if shrinkage > WARN_SHRINKAGE:
        warnings.append(f"shrinkage {shrinkage:.4f} > warn threshold {WARN_SHRINKAGE}")

    if verbose:
        eigvals = sorted(np.linalg.eigvalsh(lw.covariance_), reverse=True)
        print(f"      returns matrix: {log_ret.shape}")
        print(f"      top 3 eigenvalues:    {eigvals[:3]}")
        print(f"      bottom 3 eigenvalues: {eigvals[-3:]}")

    _line("C] Covariance shrinkage     ", PASS, detail)
    return True, {"shrinkage": shrinkage, "cond": cond}, warnings


# ============================================================================
# CHECK D — ML MODEL FRESHNESS
# ============================================================================


def _parse_ml_gated_tickers(cfg_text: str) -> dict[str, dict]:
    """
    Parse ASSET_SPECIFIC_CONFIGS entries with strategy_type == 'ml_signal'.
    Returns {ticker: {model_type, strategy_model_path}}.

    Also includes the carry_USDJPY pair from CASH_PORTFOLIO_CONFIG.
    """
    m_asc = re.search(r"ASSET_SPECIFIC_CONFIGS\s*=\s*\{(.+?)^\}", cfg_text, re.S | re.M)
    if not m_asc:
        return {}
    body = m_asc.group(1)
    # Split per-ticker blocks
    block_re = re.compile(r'^\s*"([A-Z0-9._-]+)":\s*\{(.+?)^\s*\},', re.M | re.S)
    out: dict[str, dict] = {}
    for mb in block_re.finditer(body):
        ticker = mb.group(1)
        inner = mb.group(2)
        stype = re.search(r'"strategy_type":\s*"([^"]+)"', inner)
        if not stype or stype.group(1) != "ml_signal":
            continue
        mtype = re.search(r'"model_type":\s*"([^"]+)"', inner)
        mpath = re.search(r'"strategy_model_path":\s*"([^"]+)"', inner)
        out[ticker] = {
            "model_type": mtype.group(1) if mtype else None,
            "model_path": mpath.group(1) if mpath else None,
            "scaler_path": None,
            "kind": "asset",
        }

    # Carry trade pair
    m_carry = re.search(r'"carry_model":\s*\{(.+?)\}', cfg_text, re.S)
    m_carry_pair = re.search(r'"carry_pair":\s*"([^"]+)"', cfg_text)
    if m_carry and m_carry_pair:
        inner = m_carry.group(1)
        mtype = re.search(r'"model_type":\s*"([^"]+)"', inner)
        mpath = re.search(r'"strategy_model_path":\s*"([^"]+)"', inner)
        spath = re.search(r'"scaler_path":\s*"([^"]+)"', inner)
        out[f"carry_{m_carry_pair.group(1)}"] = {
            "model_type": mtype.group(1) if mtype else None,
            "model_path": mpath.group(1) if mpath else None,
            "scaler_path": spath.group(1) if spath else None,
            "kind": "carry",
        }

    return out


def _find_scaler(ticker: str, model_type: str | None) -> Path | None:
    """Try common scaler path conventions."""
    d = STRATEGY_MODELS_DIR
    candidates = [
        d / f"{model_type}_scaler_{ticker}.pkl" if model_type else None,
        d / f"scaler_{ticker}_{model_type}.pkl" if model_type else None,
        d / f"{ticker}_scaler.pkl",
        d / f"scaler_{ticker}.pkl",
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None


def check_d_models(verbose: bool = False) -> tuple[bool, dict, list[str]]:
    warnings: list[str] = []
    if not CONFIG_PY_PATH.exists():
        print(f"  [D] FAIL: config.py not found: {CONFIG_PY_PATH}")
        return False, {}, warnings

    with open(CONFIG_PY_PATH) as f:
        cfg_text = f.read()

    ml_items = _parse_ml_gated_tickers(cfg_text)
    if not ml_items:
        warnings.append("no ML-gated items found")
        _line("D] ML model freshness       ", WARN, "no ML items")
        return True, {"ml_items": {}}, warnings

    now = datetime.now()
    age_cutoff = MAX_MODEL_AGE_DAYS
    oldest = ("", -1.0)
    failures: list[str] = []
    results: list[tuple[str, str, float, str]] = []

    for ticker, info in ml_items.items():
        mpath_rel = info.get("model_path")
        if not mpath_rel:
            failures.append(f"{ticker}: no model_path in config")
            continue

        # Paths are relative to repo root (prefix "strategy_models/") or repo root with "execution/" prefix
        candidates = [
            REPO_ROOT / "execution" / mpath_rel,
            REPO_ROOT / mpath_rel,
            STRATEGY_MODELS_DIR / os.path.basename(mpath_rel),
        ]
        mpath = next((c for c in candidates if c.exists()), None)
        if mpath is None:
            failures.append(f"{ticker}: model file not found (tried {[str(c) for c in candidates]})")
            continue

        age_days = (now - datetime.fromtimestamp(mpath.stat().st_mtime)).total_seconds() / 86400
        if age_days > age_cutoff:
            failures.append(f"{ticker}: model {age_days:.0f}d old (>{age_cutoff}d)")
            continue

        # Scaler lookup: either explicit in config, or try conventional names.
        # VAR (statsmodels Vector Autoregression) models don't use scalers —
        # they're statistical time-series models operating on raw (or
        # differenced) series. Skip scaler lookup entirely for model_type=='var'.
        mtype = info.get("model_type")
        scaler_path: Path | None = None
        if mtype != "var":
            explicit_scaler = info.get("scaler_path")
            if explicit_scaler:
                scaler_candidates = [
                    REPO_ROOT / "execution" / explicit_scaler,
                    REPO_ROOT / explicit_scaler,
                    STRATEGY_MODELS_DIR / os.path.basename(explicit_scaler),
                ]
                scaler_path = next((c for c in scaler_candidates if c.exists()), None)
            if scaler_path is None:
                base = ticker[len("carry_"):] if info.get("kind") == "carry" else ticker
                scaler_path = _find_scaler(base, mtype)

            if scaler_path is None:
                if explicit_scaler:
                    failures.append(
                        f"{ticker}: config points to scaler {explicit_scaler} but file not found"
                    )
                else:
                    warnings.append(f"{ticker}: scaler not found via conventional names")
            else:
                scaler_age = (now - datetime.fromtimestamp(scaler_path.stat().st_mtime)).total_seconds() / 86400
                if scaler_age > age_cutoff:
                    warnings.append(f"{ticker}: scaler {scaler_age:.0f}d old")

        scaler_disp = scaler_path.name if scaler_path else ("(N/A for VAR)" if mtype == "var" else "MISSING")
        results.append((ticker, mpath.name, age_days, scaler_disp))
        if age_days > oldest[1]:
            oldest = (ticker, age_days)

    if failures:
        print("  [D] FAIL:")
        for f in failures:
            print(f"      {f}")
        return False, {}, warnings

    detail = f"oldest: {oldest[0]} {oldest[1]:.1f}d"
    if verbose:
        print(f"      {'ticker':20s}  {'model':40s}  {'age (d)':>8s}  scaler")
        for t, mn, ad, sn in results:
            print(f"      {t:20s}  {mn:40s}  {ad:>8.1f}  {sn}")
    _line("D] ML model freshness       ", PASS, detail)
    return True, {"ml_results": results}, warnings


# ============================================================================
# CHECK E — IBKR CONNECTION PROBE (OPTIONAL)
# ============================================================================


def check_e_ibkr(verbose: bool = False, region: str | None = None) -> tuple[bool, dict, list[str]]:
    """Raw ibapi probe; reuses pattern from test_direct_connection.py.

    When ``region`` is given, the probe connects with that region's REAL
    production client ID (config.get_client_id) instead of the old
    IB_CLIENT_ID+99 offset. This makes preflight representative: if main.py
    would be rejected with error 326 because another region still holds the ID,
    preflight now fails too (instead of passing on a different ID and letting
    main.py silently die — the gap that hid the 2026-06-15 outage). The probe
    always disconnects and waits briefly so the ID is free before main.py runs.
    """
    warnings: list[str] = []
    try:
        from ibapi.client import EClient  # type: ignore
        from ibapi.wrapper import EWrapper  # type: ignore
    except ImportError as e:
        print(f"  [E] FAIL: ibapi not installed: {e}")
        return False, {}, warnings

    # Import config via sys.path
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from execution.config import IB_HOST, IB_PORT, IB_CLIENT_ID  # type: ignore
    except Exception as e:
        print(f"  [E] FAIL: cannot import config: {e}")
        return False, {}, warnings

    # Allocate the probe's client ID from the shared rotator so it never
    # collides with a live trading/data session (and so it is released back to
    # the pool afterwards). probe=False because THIS preflight connect is itself
    # the liveness probe — no need for the rotator to also connect. We release
    # the id in _release_probe() below regardless of outcome.
    _probe_allocated = False
    try:
        from algos.common.client_id_rotator import (
            allocate_client_id as _alloc_cid,
            release_client_id as _release_cid,
        )
        probe_client_id = _alloc_cid(
            host=IB_HOST, port=IB_PORT, probe=False, label="preflight"
        )
        _probe_allocated = True
    except Exception as e:
        # Rotator unavailable — fall back to a high static id unlikely to clash.
        print(f"  [E] note: client-id rotator unavailable ({e}); using fallback probe id.")
        probe_client_id = (IB_CLIENT_ID + 99) % 100000

    class Probe(EWrapper, EClient):  # type: ignore[misc]
        def __init__(self):
            EClient.__init__(self, self)
            self.connected_ok = False
            self.next_id = None
            self.err: list[tuple[int, int, str]] = []
            self.summary: dict[str, str] = {}
            self.summary_end = False

        def error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject=""):  # noqa: N802
            # IBAPI >= 10.37 signature (6 positional args incl. errorTime).
            # See MEMORY.md "IBKR API Gotcha".
            # 2104/2106/2158 are benign "data farm connection OK" messages
            if errorCode not in (2104, 2106, 2158, 2103, 2105, 2107, 2168, 2169):
                self.err.append((reqId, errorCode, errorString))

        def connectAck(self):  # noqa: N802
            self.connected_ok = True

        def nextValidId(self, orderId):  # noqa: N802
            self.next_id = orderId

        def accountSummary(self, reqId, account, tag, value, currency):  # noqa: N802
            self.summary[tag] = f"{value} {currency}"

        def accountSummaryEnd(self, reqId):  # noqa: N802
            self.summary_end = True

    import threading
    app = Probe()

    def _release_probe():
        # Guaranteed teardown: disconnect the probe and hand its rotator-
        # allocated client ID back to the shared pool. The settle sleep lets the
        # gateway reap the socket before the next session connects.
        try:
            app.cancelAccountSummary(9999)
        except Exception:
            pass
        try:
            app.disconnect()
        except Exception:
            pass
        if _probe_allocated:
            try:
                _release_cid(probe_client_id, host=IB_HOST, port=IB_PORT)
            except Exception:
                pass
        time.sleep(0.5)

    try:
        try:
            app.connect(IB_HOST, IB_PORT, probe_client_id)
        except Exception as e:
            print(f"  [E] FAIL: connect raised: {e}")
            return False, {}, warnings

        t = threading.Thread(target=app.run, daemon=True)
        t.start()

        t0 = time.time()
        while time.time() - t0 < 5.0:
            if app.connected_ok and app.next_id is not None:
                break
            time.sleep(0.1)

        if not (app.connected_ok and app.isConnected()):
            # Surface a client-id collision (326) explicitly so the operator
            # sees the real reason main.py would also fail.
            if any(c == 326 for _, c, _ in app.err):
                print(
                    f"  [E] FAIL: client ID {probe_client_id} already in use (326) — "
                    f"another session holds it. main.py would also fail to connect."
                )
            else:
                errs = "; ".join(f"{c}:{s}" for _, c, s in app.err) if app.err else "timeout"
                print(f"  [E] FAIL: did not connect to IB ({IB_HOST}:{IB_PORT}) — {errs}")
            return False, {}, warnings

        # Request account summary
        app.reqAccountSummary(9999, "All", "NetLiquidation,TotalCashValue,BuyingPower,ExcessLiquidity")
        t1 = time.time()
        while time.time() - t1 < 3.0:
            if app.summary_end:
                break
            time.sleep(0.1)

        if not app.summary:
            print(f"  [E] FAIL: connected but no account summary received within 3s")
            return False, {}, warnings

        nav = app.summary.get("NetLiquidation", "?")
        cash = app.summary.get("TotalCashValue", "?")
        bp = app.summary.get("BuyingPower", "?")
        detail = f"NAV={nav} cash={cash}"
        if verbose:
            print(f"      host={IB_HOST}:{IB_PORT} client_id={probe_client_id}")
            for k, v in sorted(app.summary.items()):
                print(f"      {k:25s} {v}")
        _line("E] IBKR connection          ", PASS, detail)
        return True, {"summary": app.summary}, warnings
    finally:
        _release_probe()


# ============================================================================
# CHECK F — MINIMUM NOTIONAL VIABILITY
# ============================================================================


def _ticker_currency(ticker: str) -> str:
    """Resolve a ticker's trading currency from its exchange suffix."""
    for suffix, ccy in EXCHANGE_SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return ccy
    return "USD"


def _load_fx_rates() -> dict[str, float]:
    """Load CURRENCY_RATE_FALLBACKS from config.py via text parse."""
    if not CONFIG_PY_PATH.exists():
        return {}
    with open(CONFIG_PY_PATH) as f:
        text = f.read()
    m = re.search(r"CURRENCY_RATE_FALLBACKS\s*=\s*\{(.+?)^\}", text, re.S | re.M)
    if not m:
        return {}
    rates: dict[str, float] = {}
    for ccy, rate in re.findall(r'"([A-Z]{3})":\s*([\d.]+)', m.group(1)):
        rates[ccy] = float(rate)
    rates["USD"] = 1.0
    return rates


def check_f_notional(
    weights: dict, prices: dict, nav: float, leverage: float = 1.3, verbose: bool = False
) -> tuple[bool, dict, list[str]]:
    warnings: list[str] = []
    deployed = nav * leverage
    fx_rates = _load_fx_rates()

    rows = []
    failures: list[str] = []
    max_drift_warn = 0.0
    max_drift_warn_ticker = ""

    for t, w in weights.items():
        if t not in prices:
            failures.append(f"{t}: price series missing (Check B must pass first)")
            continue
        px_local = float(prices[t].dropna().iloc[-1])
        if px_local <= 0:
            failures.append(f"{t}: non-positive price {px_local}")
            continue

        ccy = _ticker_currency(t)
        fx = fx_rates.get(ccy)
        if fx is None:
            failures.append(f"{t}: no FX rate for currency {ccy} in CURRENCY_RATE_FALLBACKS")
            continue

        px_usd = px_local * fx
        expected_notional_usd = deployed * w
        raw_shares = expected_notional_usd / px_usd
        # Round to nearest integer but never below 0
        rounded = max(0, round(raw_shares))
        # If rounding to 0 would cause full miss and raw_shares >= 0.5, use 1 instead
        # (because round() banker-rounds; we want ceil-ish for small pos)
        if rounded == 0 and raw_shares >= 0.5:
            rounded = 1

        actual_notional_usd = rounded * px_usd
        actual_weight = actual_notional_usd / deployed
        drift_abs = abs(actual_weight - w)
        drift_rel = drift_abs / w if w > 0 else float("inf")

        rows.append((t, w, px_local, ccy, px_usd, raw_shares, rounded, actual_weight, drift_rel))

        if rounded == 0:
            failures.append(
                f"{t}: rounds to 0 shares (expected_notional=${expected_notional_usd:.2f}, "
                f"price={px_local:.2f} {ccy} = ${px_usd:.2f}, raw={raw_shares:.3f}) — weight unachievable"
            )
            continue

        if drift_rel > NOTIONAL_DRIFT_FAIL:
            failures.append(
                f"{t}: drift {drift_rel:.1%} > FAIL threshold {NOTIONAL_DRIFT_FAIL:.0%} "
                f"(target w={w:.4%}, actual={actual_weight:.4%}, "
                f"price {px_local:.2f} {ccy})"
            )
            continue

        if drift_rel > NOTIONAL_DRIFT_WARN:
            warnings.append(
                f"{t}: drift {drift_rel:.1%} > WARN {NOTIONAL_DRIFT_WARN:.0%} "
                f"(target {w:.4%}, actual {actual_weight:.4%}, {rounded} sh "
                f"@ {px_local:.2f} {ccy})"
            )
            if drift_rel > max_drift_warn:
                max_drift_warn = drift_rel
                max_drift_warn_ticker = t

    if failures:
        print("  [F] FAIL:")
        for f in failures:
            print(f"      {f}")
        return False, {}, warnings

    if max_drift_warn_ticker:
        detail = f"max drift {max_drift_warn:.1%} on {max_drift_warn_ticker}"
    else:
        # rows schema: (t, w, px_local, ccy, px_usd, raw_shares, rounded, actual_weight, drift_rel)
        max_drift = max((r[8] for r in rows), default=0.0)
        max_ticker = max(rows, key=lambda r: r[8])[0] if rows else ""
        detail = f"max drift {max_drift:.1%} on {max_ticker}"

    if verbose:
        print(f"      NAV=${nav:,.0f} × leverage {leverage} = ${deployed:,.0f}")
        print(
            f"      {'ticker':15s}  {'target':>8s}  {'px_local':>9s} {'ccy':>4s}  "
            f"{'px_usd':>9s}  {'raw sh':>8s}  {'rounded':>7s}  {'actual':>8s}  {'drift':>7s}"
        )
        for t, w, pxl, ccy, pxu, rs, r, aw, dr in sorted(rows, key=lambda x: -x[1]):
            print(
                f"      {t:15s}  {w*100:>7.3f}%  {pxl:>9.2f} {ccy:>4s}  "
                f"{pxu:>9.2f}  {rs:>8.3f}  {r:>7d}  {aw*100:>7.3f}%  {dr*100:>6.2f}%"
            )

    _line("F] Min-notional viability   ", PASS, detail)
    return True, {"rows": rows}, warnings


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-deployment preflight check for portfolio_weights_hrp_H2_2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--nav",
        type=float,
        default=float(os.environ.get("PREFLIGHT_NAV", 0)),
        help="Net asset value in USD (or env PREFLIGHT_NAV). Required for Check F.",
    )
    ap.add_argument(
        "--leverage", type=float, default=1.3, help="Leverage multiplier (default 1.3)"
    )
    ap.add_argument(
        "--with-ibkr", action="store_true", help="Probe IBKR connection (Check E)"
    )
    ap.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region whose REAL production client ID Check E should probe with "
        "(makes preflight representative of main.py). Omit for the legacy "
        "offset probe.",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Per-ticker details")
    args = ap.parse_args()

    if args.nav <= 0:
        print("ERROR: --nav must be positive. Pass --nav 10500 or set PREFLIGHT_NAV.", file=sys.stderr)
        return 2

    print(f"Preflight Check — NAV=${args.nav:,.0f} × leverage={args.leverage} "
          f"= ${args.nav*args.leverage:,.0f} deployed\n")

    all_warnings: list[tuple[str, str]] = []
    any_fail = False

    # A
    ok_a, ctx_a, w_a = check_a_weights(args.verbose)
    all_warnings.extend(("A", w) for w in w_a)
    if not ok_a:
        any_fail = True
        # short-circuit: everything else depends on weights
        _finalize(any_fail, all_warnings)
        return 1

    tickers = ctx_a["tickers"]
    weights = ctx_a["weights"]

    # B
    ok_b, ctx_b, w_b = check_b_data(tickers, args.verbose)
    all_warnings.extend(("B", w) for w in w_b)
    if not ok_b:
        any_fail = True

    # C (depends on B prices)
    if ok_b:
        ok_c, _ctx_c, w_c = check_c_covariance(ctx_b["prices"], tickers, args.verbose)
        all_warnings.extend(("C", w) for w in w_c)
        if not ok_c:
            any_fail = True
    else:
        _line("C] Covariance shrinkage     ", SKIP, "Check B failed")

    # D
    ok_d, _ctx_d, w_d = check_d_models(args.verbose)
    all_warnings.extend(("D", w) for w in w_d)
    if not ok_d:
        any_fail = True

    # E
    if args.with_ibkr:
        ok_e, _ctx_e, w_e = check_e_ibkr(args.verbose, region=args.region)
        all_warnings.extend(("E", w) for w in w_e)
        if not ok_e:
            any_fail = True
    else:
        _line("E] IBKR connection          ", SKIP, "use --with-ibkr to enable")

    # F (depends on B prices)
    if ok_b:
        ok_f, _ctx_f, w_f = check_f_notional(
            weights, ctx_b["prices"], args.nav, args.leverage, args.verbose
        )
        all_warnings.extend(("F", w) for w in w_f)
        if not ok_f:
            any_fail = True
    else:
        _line("F] Min-notional viability   ", SKIP, "Check B failed")

    _finalize(any_fail, all_warnings)
    return 1 if any_fail else 0


def _finalize(any_fail: bool, warnings: list[tuple[str, str]]) -> None:
    print()
    if warnings:
        print("Warnings:")
        for chk, w in warnings:
            print(f"  [{chk}] {w}")
        print()
    if any_fail:
        print(f"OVERALL: {FAIL} — DO NOT DEPLOY")
    else:
        print(f"OVERALL: {PASS} — safe to deploy")


if __name__ == "__main__":
    sys.exit(main())
