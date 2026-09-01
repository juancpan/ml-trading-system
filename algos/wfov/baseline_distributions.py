"""Backtest distribution baseline computation (Phase 2.1).

Extracts per-portfolio rolling-N-day percentiles for return, Sharpe, and
drawdown from WFOV summary JSONs. The output JSON is consumed by Phase
2.3 ``revision_triggers.py`` and Phase 1.4 ``revision_dashboard.py``.

For each WFOV summary in ``algos/wfov/results/summaries/``, we read the
``performance_metrics`` block (already contains percentile_25 / 75 / 95
for annual_return, sharpe_ratio, max_drawdown, etc.).

We then aggregate ALL summaries matching the live universe into a single
baseline JSON, with one entry per metric × ticker × model.

The baseline file is committed to git (alongside the live config) so
the kill-switch thresholds and trigger logic have a reproducible source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WFOV_RESULTS = REPO_ROOT / "algos" / "wfov" / "results" / "summaries"
BASELINE_DIR = REPO_ROOT / "algos" / "wfov" / "baselines"


# Metrics we treat as canonical baselines.
_PRIMARY_METRICS = (
    "annual_return", "annual_volatility", "sharpe_ratio",
    "max_drawdown", "longest_drawdown_days", "hit_ratio",
    "skewness", "kurtosis",
)


def _load_summary(p: Path) -> Optional[dict]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as exc:
        LOGGER.debug("Skipping unreadable %s: %s", p.name, exc)
        return None


def _summary_matches(summary: dict, ticker: str, model: Optional[str]) -> bool:
    meta = summary.get("metadata", {})
    if meta.get("ticker") != ticker:
        return False
    if model is not None and meta.get("model_name") != model:
        return False
    return True


def collect_per_ticker_baselines(
    portfolio: dict[str, dict],
    *,
    results_dir: Path = WFOV_RESULTS,
) -> dict:
    """Walk ``portfolio`` (ticker -> {model_type: ...}) and gather the
    matching WFOV summaries' percentile statistics into a single baseline.

    Returns a dict with structure::

        {
            "portfolio_hash": "...",
            "generated_at": "...",
            "tickers": {
                "AAA": {
                    "model": "gnb",
                    "source_summary": "montec_gnb_BK_...summary.json",
                    "metrics": {
                        "sharpe_ratio": {"mean": ..., "p25": ..., "p75": ...},
                        ...
                    }
                },
                ...
            }
        }
    """
    out_tickers: dict[str, Any] = {}
    known_gaps: list[dict] = []

    # Pre-scan: enumerate tickers that have ANY summary, regardless of
    # model match. Used to distinguish "no_summary_for_ticker" from
    # "model_name_mismatch" in the known_gaps reason field.
    all_candidates = sorted(results_dir.glob("*_summary.json"))
    tickers_with_any_summary: set[str] = set()
    for c in all_candidates:
        s = _load_summary(c)
        if s is None:
            continue
        t = (s.get("metadata") or {}).get("ticker")
        if t:
            tickers_with_any_summary.add(t)

    for ticker, cfg in portfolio.items():
        cfg_is_dict = isinstance(cfg, dict)
        model = cfg.get("model_type") if cfg_is_dict else None
        live_weight = cfg.get("live_weight") if cfg_is_dict else None

        # Prefer model-matching files; fall back to any file for the ticker.
        matched: list[Path] = []
        for c in all_candidates:
            s = _load_summary(c)
            if s is None:
                continue
            if _summary_matches(s, ticker, model):
                matched.append(c)
            elif model is None and _summary_matches(s, ticker, None):
                matched.append(c)

        if not matched:
            # Classify the reason for the gap so future readers don't have
            # to re-do today's diagnostic work.
            if ticker in tickers_with_any_summary:
                reason = "model_name_mismatch"
            else:
                reason = "no_summary_for_ticker"
            LOGGER.warning(
                "No WFOV summary found for ticker=%s model=%s (reason=%s)",
                ticker, model, reason,
            )
            known_gaps.append({
                "ticker": ticker,
                "requested_model": model,
                "reason": reason,
                "live_weight": _coerce_float(live_weight),
            })
            continue

        # Use the LATEST matching summary (mtime).
        chosen = max(matched, key=lambda p: p.stat().st_mtime)
        summary = _load_summary(chosen)
        pm = summary.get("performance_metrics", {}) if summary else {}

        metrics_out: dict[str, dict] = {}
        for metric in _PRIMARY_METRICS:
            stats = pm.get(metric)
            if not isinstance(stats, dict):
                continue
            metrics_out[metric] = {
                "mean": _coerce_float(stats.get("mean")),
                "std": _coerce_float(stats.get("std")),
                "min": _coerce_float(stats.get("min")),
                "max": _coerce_float(stats.get("max")),
                "median": _coerce_float(stats.get("median")),
                "p25": _coerce_float(stats.get("percentile_25")),
                "p75": _coerce_float(stats.get("percentile_75")),
                "p95": _coerce_float(stats.get("percentile_95")),
                "count": int(stats.get("count", 0)),
            }

        out_tickers[ticker] = {
            "model": model,
            "source_summary": chosen.name,
            "metrics": metrics_out,
            "live_weight": _coerce_float(live_weight),
        }

    coverage = _compute_coverage(portfolio, out_tickers, known_gaps)

    # Emit a loud warning when coverage by weight is poor, so future runs
    # surface under-coverage explicitly rather than silently producing a
    # biased baseline.
    if coverage["coverage_by_weight"] is not None and coverage["coverage_by_weight"] < 0.80:
        LOGGER.warning(
            "Baseline coverage by weight is %.1f%% (< 80%% threshold); "
            "%d/%d tickers missing. See known_gaps for details.",
            coverage["coverage_by_weight"] * 100,
            len(known_gaps), len(portfolio),
        )

    return {
        "portfolio_hash": _portfolio_hash(portfolio),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": out_tickers,
        "known_gaps": known_gaps,
        "coverage": coverage,
    }


def _compute_coverage(
    portfolio: dict, out_tickers: dict, known_gaps: list[dict],
) -> dict:
    """Compute coverage statistics. Returns:

        {
            "covered_count": int,
            "total_count": int,
            "coverage_by_count": float,           # always present
            "covered_weight": float | None,       # None if no live_weights provided
            "total_weight": float | None,
            "coverage_by_weight": float | None,
        }
    """
    total = len(portfolio)
    covered_count = len(out_tickers)
    by_count = covered_count / total if total else 0.0

    # Weight-based coverage requires at least one ticker in portfolio to
    # carry a live_weight value. We fall back to None if absent.
    has_weights = any(
        isinstance(cfg, dict) and cfg.get("live_weight") is not None
        for cfg in portfolio.values()
    )
    covered_weight = None
    total_weight = None
    by_weight = None
    if has_weights:
        total_weight = sum(
            float(cfg.get("live_weight") or 0.0)
            for cfg in portfolio.values()
            if isinstance(cfg, dict)
        )
        covered_weight = sum(
            float((cfg.get("live_weight") or 0.0))
            for tkr, cfg in portfolio.items()
            if isinstance(cfg, dict) and tkr in out_tickers
        )
        by_weight = (covered_weight / total_weight) if total_weight else 0.0

    return {
        "covered_count": covered_count,
        "total_count": total,
        "coverage_by_count": by_count,
        "covered_weight": covered_weight,
        "total_weight": total_weight,
        "coverage_by_weight": by_weight,
    }


def aggregate_portfolio_baseline(per_ticker: dict) -> dict:
    """Aggregate per-ticker percentiles into a single portfolio-level
    distribution by averaging across tickers, weighted by ``count`` if
    available. This is a rough approximation — Phase 2's plan flagged
    this; a true portfolio backtest would be more accurate."""
    agg: dict[str, list[float]] = defaultdict(list)
    weights: dict[str, list[int]] = defaultdict(list)
    for _ticker, td in per_ticker.get("tickers", {}).items():
        for metric, stats in td.get("metrics", {}).items():
            for k, v in stats.items():
                if k == "count" or v is None:
                    continue
                agg[f"{metric}.{k}"].append(float(v))
                weights[f"{metric}.{k}"].append(int(stats.get("count", 1)))

    portfolio_metrics: dict[str, float] = {}
    for k, values in agg.items():
        ws = weights[k]
        if not values:
            continue
        if sum(ws) == 0:
            portfolio_metrics[k] = float(np.mean(values))
        else:
            portfolio_metrics[k] = float(np.average(values, weights=ws))

    return portfolio_metrics


def write_baseline(per_ticker: dict, *, out_dir: Path = BASELINE_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"portfolio_{per_ticker['portfolio_hash']}_baseline.json"
    path = out_dir / name
    aggregated = aggregate_portfolio_baseline(per_ticker)
    payload = {**per_ticker, "portfolio_aggregated_metrics": aggregated}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_live_portfolio() -> dict[str, dict]:
    """Load the live universe from ``execution.config``.

    Returns a dict ``{ticker: {model_type, live_weight, ...}}`` so the
    baseline can compute weight-based coverage. ``live_weight`` is
    pulled from ``TARGET_ALLOCATION``; tickers absent from that dict
    receive ``live_weight=None``.
    """
    import importlib.util
    cfg_path = REPO_ROOT / "execution" / "config.py"
    spec = importlib.util.spec_from_file_location("ibkr_config", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    asset_configs = dict(getattr(mod, "ASSET_SPECIFIC_CONFIGS", {}))
    target_allocation = dict(getattr(mod, "TARGET_ALLOCATION", {}))
    out: dict[str, dict] = {}
    for ticker, cfg in asset_configs.items():
        merged = dict(cfg) if isinstance(cfg, dict) else {}
        merged["live_weight"] = target_allocation.get(ticker)
        out[ticker] = merged
    return out


def _portfolio_hash(portfolio: dict) -> str:
    keys = sorted(portfolio.keys())
    blob = json.dumps(
        {k: portfolio[k].get("model_type") if isinstance(portfolio[k], dict) else None
         for k in keys},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _coerce_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
        if np.isnan(out) or np.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute backtest baseline distributions")
    parser.add_argument("--results-dir", type=Path, default=WFOV_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    portfolio = load_live_portfolio()
    LOGGER.info("Loaded live universe: %d tickers", len(portfolio))
    pt = collect_per_ticker_baselines(portfolio, results_dir=args.results_dir)
    LOGGER.info("Matched %d / %d tickers to WFOV summaries",
                len(pt["tickers"]), len(portfolio))
    out = write_baseline(pt, out_dir=args.out_dir)
    print(f"Baseline written: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
