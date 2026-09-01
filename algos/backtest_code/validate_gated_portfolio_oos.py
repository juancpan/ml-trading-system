"""
Backtest harness for the ML-Gated Weekly Portfolio.

Simulates the weekly gated strategy over historical data. Accepts a pluggable
``signal_fn`` callback so different signal sources (ML models, heuristics, etc.)
can be swapped in without changing the harness.
"""

import argparse
import glob as glob_mod
import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algos.backtest_code.weekly_gate_engine import apply_gates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """Compute standard portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    risk_free_rate : float
        Annualised risk-free rate (default 4 %).
    periods_per_year : int
        Trading days per year (default 252).

    Returns
    -------
    dict
        Keys: sharpe, annual_return, annual_volatility, max_drawdown,
              sortino, calmar, win_rate
    """
    n = len(returns)
    if n < 5:
        return {
            "sharpe": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
        }

    mean_daily = returns.mean()
    std_daily = returns.std(ddof=1)

    annual_return = mean_daily * periods_per_year
    annual_volatility = std_daily * np.sqrt(periods_per_year)

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf

    # Sharpe
    if std_daily == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_daily - daily_rf) * np.sqrt(periods_per_year) / std_daily

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        sortino = 0.0
    else:
        sortino = (excess.mean() * np.sqrt(periods_per_year)) / (
            downside.std(ddof=1) * np.sqrt(periods_per_year)
        )

    # Max drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_drawdown = drawdowns.min()  # negative value

    # Calmar
    if max_drawdown == 0:
        calmar = 0.0
    else:
        calmar = annual_return / abs(max_drawdown)

    # Win rate
    win_rate = (returns > 0).sum() / n

    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

# Day-of-week mapping for pandas Timestamp.dayofweek
_DOW_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
}


def run_gated_portfolio_backtest(
    returns_df: pd.DataFrame,
    hrp_weights: Dict[str, float],
    signal_fn: Callable,
    max_weight: float = 0.25,
    min_active_tickers: int = 3,
    rebalance_day: str = "Monday",
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """Run the weekly ML-gated portfolio backtest.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns indexed by business date, columns = tickers.
    hrp_weights : dict
        {ticker: weight} from HRP optimisation.
    signal_fn : callable
        ``signal_fn(ticker, hist_returns_df, week_date) -> int``
        Returns > 0 for gate-open, <= 0 for gate-closed.
    max_weight : float
        Per-ticker cap passed to the gate engine.
    min_active_tickers : int
        Minimum active tickers before falling back to no-scale mode.
    rebalance_day : str
        Day of week to rebalance (default ``"Monday"``).
    risk_free_rate : float
        Annualised risk-free rate for metrics computation.
    periods_per_year : int
        Trading days per year for annualisation.

    Returns
    -------
    dict
        Keys: portfolio_returns, metrics, weekly_allocations,
              benchmark_returns, benchmark_metrics
    """
    # --- Align tickers between weights and returns -----------------------
    common_tickers = sorted(set(hrp_weights.keys()) & set(returns_df.columns))
    if not common_tickers:
        raise ValueError("No common tickers between hrp_weights and returns_df")

    returns_df = returns_df[common_tickers]

    # Renormalise HRP weights to common tickers
    raw_sum = sum(hrp_weights[t] for t in common_tickers)
    if raw_sum <= 0:
        raise ValueError("Sum of HRP weights for common tickers is <= 0")
    hrp_norm = {t: hrp_weights[t] / raw_sum for t in common_tickers}

    # --- Setup -----------------------------------------------------------
    rebalance_dow = _DOW_MAP.get(rebalance_day)
    if rebalance_dow is None:
        raise ValueError(
            f"Invalid rebalance_day '{rebalance_day}'. "
            f"Must be one of {list(_DOW_MAP.keys())}"
        )

    dates = returns_df.index
    n_tickers = len(common_tickers)

    # Current weights: start with HRP norm (before first rebalance)
    current_weights = np.array([hrp_norm[t] for t in common_tickers])

    portfolio_rets: List[float] = []
    weekly_allocations: List[Tuple] = []
    prev_weights = current_weights.copy()
    turnover_values: List[float] = []

    # --- Walk through each trading day -----------------------------------
    for i, date in enumerate(dates):
        is_rebalance = date.dayofweek == rebalance_dow

        if is_rebalance:
            # Gather signals for each ticker
            hist_returns = returns_df.iloc[:i] if i > 0 else returns_df.iloc[:1]
            signals = {}
            for ticker in common_tickers:
                signals[ticker] = signal_fn(ticker, hist_returns, date)

            # Apply gates
            gated = apply_gates(
                hrp_norm,
                signals,
                max_weight=max_weight,
                min_active_tickers=min_active_tickers,
            )

            new_weights = np.array([gated[t] for t in common_tickers])

            # Track turnover
            turnover = np.sum(np.abs(new_weights - prev_weights))
            turnover_values.append(turnover)

            current_weights = new_weights
            prev_weights = current_weights.copy()

            weekly_allocations.append(
                (date, {t: current_weights[j] for j, t in enumerate(common_tickers)})
            )

        # Daily portfolio return
        day_returns = returns_df.loc[date].values
        port_ret = np.sum(day_returns * current_weights)
        portfolio_rets.append(port_ret)

    # --- Build results ---------------------------------------------------
    portfolio_returns = pd.Series(portfolio_rets, index=dates)

    # Static HRP benchmark
    bench_weights = np.array([hrp_norm[t] for t in common_tickers])
    benchmark_returns = (returns_df * bench_weights).sum(axis=1)

    # Metrics
    metrics = _compute_metrics(portfolio_returns, risk_free_rate, periods_per_year)
    metrics["turnover"] = float(np.mean(turnover_values)) if turnover_values else 0.0

    benchmark_metrics = _compute_metrics(
        benchmark_returns, risk_free_rate, periods_per_year
    )

    return {
        "portfolio_returns": portfolio_returns,
        "metrics": metrics,
        "weekly_allocations": weekly_allocations,
        "benchmark_returns": benchmark_returns,
        "benchmark_metrics": benchmark_metrics,
    }


# ---------------------------------------------------------------------------
# Model-based signal function factory (for CLI use)
# ---------------------------------------------------------------------------


def _make_model_signal_fn(model_dir: str) -> Callable:
    """Create a signal function that uses deployed ML models.

    Lazy-loads models from pkl files matching ``{ticker}_trading_model_*.pkl``.

    Parameters
    ----------
    model_dir : str
        Path to directory containing model pickle files.

    Returns
    -------
    callable
        ``signal_fn(ticker, features_df, week_date) -> int``
    """
    model_cache: Dict[str, object] = {}
    model_dir_path = Path(model_dir)

    def signal_fn(ticker: str, features_df: pd.DataFrame, week_date) -> int:
        # Lazy-load model
        if ticker not in model_cache:
            pattern = str(model_dir_path / f"{ticker}_trading_model_*.pkl")
            matches = sorted(glob_mod.glob(pattern))
            if not matches:
                logger.warning("No model found for %s, defaulting gate open", ticker)
                model_cache[ticker] = None
            else:
                model_path = matches[-1]  # most recent
                try:
                    with open(model_path, "rb") as f:
                        model_cache[ticker] = pickle.load(f)
                    logger.info("Loaded model for %s from %s", ticker, model_path)
                except Exception as exc:
                    logger.error("Failed to load model for %s: %s", ticker, exc)
                    model_cache[ticker] = None

        model = model_cache[ticker]
        if model is None:
            return 1  # fallback: gate open

        try:
            prediction = model.predict(features_df)
            # Use the last prediction value
            pred_val = prediction[-1] if hasattr(prediction, "__len__") else prediction
            return int(np.sign(pred_val)) if pred_val != 0 else 1
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", ticker, exc)
            return 1  # fallback: gate open

    return signal_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for running the gated portfolio backtest."""
    parser = argparse.ArgumentParser(
        description="Backtest the ML-Gated Weekly Portfolio strategy."
    )
    parser.add_argument("--csv", required=True, help="Path to combined prices CSV")
    parser.add_argument(
        "--hrp-weights", required=True, help="Path to HRP weights JSON file"
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model-dir",
        default="execution/strategy_models",
        help="Path to deployed models directory",
    )
    parser.add_argument(
        "--max-weight", type=float, default=0.25, help="Max weight per ticker"
    )
    parser.add_argument("--min-active", type=int, default=3, help="Min active tickers")
    parser.add_argument(
        "--rebalance-day", default="Monday", help="Day of week to rebalance"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.04, help="Annualised risk-free rate"
    )

    args = parser.parse_args()

    # Load data
    from algos.backtest_code.portimization import load_and_preprocess_data

    returns_df = load_and_preprocess_data(args.csv, args.start, args.end)

    # Load HRP weights
    with open(args.hrp_weights, "r") as f:
        hrp_weights = json.load(f)

    # Build signal function from models
    signal_fn = _make_model_signal_fn(args.model_dir)

    # Run backtest
    result = run_gated_portfolio_backtest(
        returns_df=returns_df,
        hrp_weights=hrp_weights,
        signal_fn=signal_fn,
        max_weight=args.max_weight,
        min_active_tickers=args.min_active,
        rebalance_day=args.rebalance_day,
        risk_free_rate=args.risk_free_rate,
    )

    # Print side-by-side comparison
    gated_m = result["metrics"]
    bench_m = result["benchmark_metrics"]

    print("\n" + "=" * 65)
    print("  ML-Gated Weekly Portfolio  vs  Static HRP Benchmark")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Gated':>15} {'Static HRP':>15}")
    print("-" * 65)

    metric_fmt = {
        "sharpe": ("{:.3f}", "{:.3f}"),
        "annual_return": ("{:.2%}", "{:.2%}"),
        "annual_volatility": ("{:.2%}", "{:.2%}"),
        "max_drawdown": ("{:.2%}", "{:.2%}"),
        "sortino": ("{:.3f}", "{:.3f}"),
        "calmar": ("{:.3f}", "{:.3f}"),
        "win_rate": ("{:.2%}", "{:.2%}"),
    }

    for metric_name, (gfmt, bfmt) in metric_fmt.items():
        g_val = gated_m.get(metric_name, np.nan)
        b_val = bench_m.get(metric_name, np.nan)
        g_str = gfmt.format(g_val) if not np.isnan(g_val) else "N/A"
        b_str = bfmt.format(b_val) if not np.isnan(b_val) else "N/A"
        print(f"  {metric_name:<25} {g_str:>15} {b_str:>15}")

    # Turnover (gated only)
    turnover = gated_m.get("turnover", 0.0)
    print(f"  {'turnover':<25} {turnover:>15.4f} {'---':>15}")

    # Excess metrics
    print("-" * 65)
    for metric_name in ["sharpe", "annual_return", "max_drawdown"]:
        g_val = gated_m.get(metric_name, np.nan)
        b_val = bench_m.get(metric_name, np.nan)
        if np.isnan(g_val) or np.isnan(b_val):
            diff_str = "N/A"
        else:
            diff = g_val - b_val
            diff_str = f"{diff:+.4f}"
        print(f"  {'excess_' + metric_name:<25} {diff_str:>15}")

    print("=" * 65)
    print(f"  Rebalance events: {len(result['weekly_allocations'])}")
    print(f"  Trading days:     {len(result['portfolio_returns'])}")
    print()


if __name__ == "__main__":
    main()
