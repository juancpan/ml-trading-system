#!/usr/bin/env python3
"""Retrain and deploy the live ML models from IBKR-sourced parquet data.

DATA SOURCE (GOLD STANDARD)
---------------------------
This script trains from the parquet store at ``data/market_data/*.parquet``,
which is populated by ``python -m algos.common.update_market_data`` (the Sunday
IBKR data refresh). IBKR is the single source of truth for all market data;
yfinance is DEPRECATED and is NOT used for training. The parquet files contain
IBKR-sourced OHLCV bars (with priceMagnifier corrections already applied).

WHAT IT DOES
------------
For each ``ml_signal`` ticker in ``execution/config.py`` (plus the JPY carry
pair), it:
  1. Invokes ``run_backtest_optimized.py --model_name <type> --data_path
     data/market_data/<ticker>.parquet --ticker <ticker>`` to retrain.
  2. Finds the newest trained model in ``algos/model_dumps/``.
  3. Deploys it to the EXACT ``strategy_model_path`` from config (preserving the
     ``_optimized`` suffix — the old deploy_models.py bug mapped
     ``lstm_optimized`` → ``lstm``, breaking live loading).
  4. Deploys the scaler to the live-expected name.
  5. Updates ``strategy_models/deployment_manifest.json``.

Only models older than ``--max-age-days`` (default 30) are retrained unless
``--force``. Use ``--tickers`` to retrain a subset.

REVISION PROTOCOL NOTE
----------------------
Per docs/REVISION_POLICY.md, any model retrain requires regenerating the
baseline file. This script logs that requirement; it does NOT auto-regenerate
the baseline (that is an operator decision per the protocol).

Usage:
    python scripts/retrain_models.py                  # retrain stale models
    python scripts/retrain_models.py --force          # retrain all
    python scripts/retrain_models.py --tickers TLT GLD
    python scripts/retrain_models.py --dry-run        # show what would happen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
IBKR_DIR = REPO_ROOT / "execution"
MODEL_DUMPS = REPO_ROOT / "algos" / "model_dumps"
SCALERS_DIR = REPO_ROOT / "algos" / "scalers"
STRATEGY_MODELS = IBKR_DIR / "strategy_models"
PARQUET_DIR = REPO_ROOT / "data" / "market_data"
MANIFEST = STRATEGY_MODELS / "deployment_manifest.json"
PYTHON = os.environ.get("PYTHON", sys.executable)
LOOKBACK_DAYS = "1260"  # ~5 years

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{_ts()}] WARNING: {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[{_ts()}] ERROR: {msg}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Discover what to retrain from config.py
# ---------------------------------------------------------------------------
def load_ml_signal_targets() -> list[dict]:
    """Read ml_signal tickers + carry pair from execution/config.py.

    Returns a list of dicts:
        {ticker, model_type, model_path (relative), scaler_path (relative or None),
         parquet_path (absolute)}
    """
    sys.path.insert(0, str(IBKR_DIR))
    import config

    targets: list[dict] = []
    for ticker, cfg in config.ASSET_SPECIFIC_CONFIGS.items():
        if cfg.get("strategy_type") != "ml_signal":
            continue
        model_type = cfg.get("model_type", "")
        model_path = cfg.get("strategy_model_path", "")
        scaler_path = cfg.get("scaler_path")
        parquet = PARQUET_DIR / f"{ticker}.parquet"
        targets.append(
            {
                "ticker": ticker,
                "model_type": model_type,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "parquet_path": parquet,
                "label": ticker,
            }
        )

    # Carry pair
    carry_cfg = config.CASH_PORTFOLIO_CONFIG.get("carry_model", {})
    if carry_cfg:
        carry_pair = config.CASH_PORTFOLIO_CONFIG.get("carry_pair", "USDJPY")
        parquet = PARQUET_DIR / f"{carry_pair}.parquet"
        targets.append(
            {
                "ticker": carry_pair,
                "model_type": carry_cfg.get("model_type", "gnb"),
                "model_path": carry_cfg.get("strategy_model_path", ""),
                "scaler_path": carry_cfg.get("scaler_path"),
                "parquet_path": parquet,
                "label": f"carry:{carry_pair}",
            }
        )
    return targets


def model_age_days(rel_path: str) -> float:
    """Age in days of a deployed model file (by mtime)."""
    p = IBKR_DIR / rel_path
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 86400.0


# ---------------------------------------------------------------------------
# Retrain one ticker
# ---------------------------------------------------------------------------
def find_newest_model(ticker: str, model_type: str) -> Path | None:
    """Find the newest trained model in model_dumps for this ticker+type."""
    candidates: list[Path] = []
    for p in MODEL_DUMPS.glob(f"{model_type}_algorithm_{ticker}_*.pkl"):
        candidates.append(p)
    for p in MODEL_DUMPS.glob(f"{model_type}_algorithm_{ticker}_*.keras"):
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def find_newest_scaler(ticker: str, model_type: str) -> Path | None:
    """Find the newest scaler in algos/scalers for this ticker+type."""
    patterns = [
        f"{model_type}_scaler_{ticker}_latest.pkl",
        f"{model_type}_scaler_{ticker}_*.pkl",
        f"scaler_{ticker}_latest.pkl",
        f"scaler_{ticker}_*.pkl",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(SCALERS_DIR.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def retrain_ticker(target: dict) -> bool:
    """Run run_backtest_optimized.py for one ticker. Returns True on success."""
    ticker = target["ticker"]
    model_type = target["model_type"]
    parquet = target["parquet_path"]

    if not parquet.exists():
        warn(f"{target['label']}: parquet not found at {parquet} — skipping.")
        warn(
            "  Run `python -m algos.common.update_market_data` first to populate "
            "the IBKR-sourced parquet store."
        )
        return False

    cmd = [
        PYTHON,
        "-m",
        "algos.backtest_code.run_backtest_optimized",
        "--model_name",
        model_type,
        "--data_path",
        str(parquet),
        "--ticker",
        ticker,
        "--lookback_days",
        LOOKBACK_DAYS,
        "--symbol",
        "Adj Close",
        "--interval",
        "1d",
    ]
    log(f"{target['label']}: training {model_type} from IBKR-sourced parquet ({parquet.name})...")
    log(f"  IBKR is the gold standard for market data; yfinance is deprecated.")
    log(f"  command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired:
        warn(f"{target['label']}: training timed out after 30min.")
        return False
    if result.returncode != 0:
        warn(f"{target['label']}: training exited {result.returncode}.")
        _print_tail("stderr", result.stderr)
        _print_tail("stdout", result.stdout)
        return False
    log(f"{target['label']}: training complete.")
    return True


def _print_tail(label: str, text: str, *, lines: int = 12) -> None:
    """Print a bounded diagnostic tail from subprocess output."""
    if not text:
        return
    print(f"    --- {label} tail ---")
    for line in text.strip().splitlines()[-lines:]:
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Deploy one ticker
# ---------------------------------------------------------------------------
def deploy_ticker(target: dict) -> bool:
    """Deploy the newest trained model + scaler to strategy_models/."""
    ticker = target["ticker"]
    model_type = target["model_type"]

    # --- Model ---
    newest = find_newest_model(ticker, model_type)
    if newest is None:
        warn(f"{target['label']}: no trained model found in {MODEL_DUMPS} for {model_type}/{ticker}.")
        return False

    dest_model = IBKR_DIR / target["model_path"]
    dest_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, dest_model)
    log(
        f"{target['label']}: deployed model {newest.name} -> {dest_model.relative_to(REPO_ROOT)} "
        f"(model_type={model_type}, _optimized suffix preserved)"
    )

    # --- Scaler ---
    scaler_src = find_newest_scaler(ticker, model_type)
    if scaler_src is not None:
        if target["scaler_path"]:
            dest_scaler = IBKR_DIR / target["scaler_path"]
        else:
            # The live loader (config_loader.py) searches for scalers by a
            # fixed set of names. For lstm_optimized it looks for
            # "lstm_scaler_{symbol}.pkl" (bare "lstm", NOT "lstm_optimized_"),
            # plus "scaler_{symbol}.pkl" and "{symbol}_scaler.pkl". Map the
            # trained model_type to the loader-expected base name so the live
            # system actually finds the fresh scaler.
            loader_base = "lstm" if model_type.startswith("lstm") else model_type
            dest_scaler = STRATEGY_MODELS / f"{loader_base}_scaler_{ticker}.pkl"
        dest_scaler.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scaler_src, dest_scaler)
        log(f"  deployed scaler {scaler_src.name} -> {dest_scaler.relative_to(REPO_ROOT)}")
    else:
        # Some models (gnb, var) don't use scalers; that's fine.
        log(f"  no scaler to deploy for {model_type}/{ticker} (may be expected).")

    return True


def update_manifest(deployed: list[dict]) -> None:
    """Update deployment_manifest.json with the new deployed_at timestamp."""
    try:
        manifest = json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        manifest = {"items": []}
    manifest["deployed_at"] = datetime.now(timezone.utc).isoformat()
    if "items" not in manifest:
        manifest["items"] = []
    log(f"Updated deployment_manifest.json deployed_at={manifest['deployed_at']}")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain + deploy live ML models from IBKR parquet data.")
    ap.add_argument("--force", action="store_true", help="Retrain all, even if not stale.")
    ap.add_argument("--max-age-days", type=float, default=30.0, help="Retrain models older than this (default 30d).")
    ap.add_argument("--tickers", nargs="*", help="Only retrain these tickers (subset).")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without training/deploying.")
    ap.add_argument("--deploy-only", action="store_true", help="Skip training; just deploy newest existing models.")
    args = ap.parse_args()

    log("=" * 60)
    log("ML MODEL RETRAIN + DEPLOY (IBKR parquet → strategy_models)")
    log("=" * 60)
    log("DATA SOURCE: data/market_data/*.parquet (IBKR-sourced, gold standard).")
    log("yfinance is DEPRECATED for training; this script does NOT use it.")

    targets = load_ml_signal_targets()
    if args.tickers:
        want = set(args.tickers)
        targets = [t for t in targets if t["ticker"] in want]
    if not targets:
        fail("No ml_signal targets found in config (or --tickers filter matched nothing).")

    log(f"\nFound {len(targets)} ml_signal target(s):")
    for t in targets:
        age = model_age_days(t["model_path"])
        stale = age > args.max_age_days
        log(f"  {t['label']:20s} model_type={t['model_type']:18s} age={age:.0f}d stale={stale}")

    if args.dry_run:
        log("\n--dry-run: would retrain + deploy:")
        for t in targets:
            age = model_age_days(t["model_path"])
            if args.force or age > args.max_age_days:
                log(f"  {t['label']}: retrain {t['model_type']} from {t['parquet_path'].name}")
            else:
                log(f"  {t['label']}: fresh ({age:.0f}d), skip")
        log("\nDRY RUN complete. No changes made.")
        return 0

    deployed: list[dict] = []
    any_failure = False
    for t in targets:
        age = model_age_days(t["model_path"])
        if not args.force and age <= args.max_age_days:
            log(f"\n{t['label']}: model is fresh ({age:.0f}d <= {args.max_age_days}d). Skipping.")
            continue

        if not args.deploy_only:
            if not retrain_ticker(t):
                any_failure = True
                continue

        if deploy_ticker(t):
            deployed.append(t)
        else:
            any_failure = True

    if deployed:
        update_manifest(deployed)
        log(f"\nDeployed {len(deployed)} model(s).")
        log(
            "REVISION PROTOCOL REMINDER (docs/REVISION_POLICY.md): a model "
            "retrain requires regenerating the baseline file. Do that as a "
            "separate operator decision per the protocol."
        )
    else:
        log("\nNo models deployed.")

    if any_failure:
        log("One or more targets failed. See warnings above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
