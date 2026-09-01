"""
Metrics extraction and aggregation for WFOV.

Extracts all performance metrics from individual backtest results and provides
aggregation utilities for computing statistics across multiple iterations.

Author: jcp
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from algos.common.metrics import calculate_strategy_performance
from algos.common.risk_analysis import calculate_risk_metrics


def extract_all_metrics(
    test_results: pd.DataFrame,
    model_name: str,
    log_prefix: str = "wfov_iter",
    max_leverage: float = 4.0,
    no_plots: bool = False,
    bh_returns: pd.Series = None,
) -> Dict[str, float]:
    """
    Extract all 11 performance metrics from a single backtest iteration.

    Combines metrics from:
    - calculate_strategy_performance() (annualized return, volatility, Sharpe, Kelly)
    - calculate_risk_metrics() (drawdown, VaR, CVaR, skewness, kurtosis)
    - Direct calculation (hit ratio)

    Args:
        test_results: DataFrame with columns ['returns', 'direction', 'position', 'strategy', 'strategy_tc']
        model_name: Name of the model (for logging)
        log_prefix: Prefix for log files (default: wfov_iter)
        max_leverage: Maximum leverage cap for Kelly criterion

    Returns:
        Dict with 11 metrics:
        {
            'hit_ratio': float,          # Win rate (accuracy_score)
            'annual_return': float,       # Annualized return
            'annual_volatility': float,   # Annualized volatility
            'sharpe_ratio': float,        # Sharpe ratio
            'max_drawdown': float,        # Maximum drawdown
            'longest_drawdown_days': int, # Longest drawdown period
            'daily_var_95': float,        # Value at Risk (95%)
            'daily_cvar_95': float,       # Conditional VaR (95%)
            'skewness': float,            # Skewness
            'kurtosis': float,            # Kurtosis
            'kelly_leverage': float       # Optimal Kelly leverage
        }

    Raises:
        ValueError: If test_results missing required columns
        ValueError: If test_results is empty

    Example:
        >>> metrics = extract_all_metrics(test_df, 'svm_optimized')
        >>> len(metrics)
        11
        >>> 'sharpe_ratio' in metrics
        True
    """
    # Validate input
    if test_results is None or test_results.empty:
        raise ValueError("test_results DataFrame is empty or None")

    required_cols = ["returns", "direction", "position", "strategy", "strategy_tc"]
    missing_cols = [col for col in required_cols if col not in test_results.columns]
    if missing_cols:
        raise ValueError(f"test_results missing required columns: {missing_cols}")

    # 1. Calculate hit ratio (win rate) - NOT returned by existing functions
    try:
        hit_ratio = accuracy_score(test_results["direction"], test_results["position"])
    except Exception as e:
        print(f"Warning: Hit ratio calculation failed: {e}")
        hit_ratio = np.nan

    # 2. Get performance metrics
    try:
        perf_metrics = calculate_strategy_performance(
            test_results,
            model_name,
            log_prefix,
            max_leverage=max_leverage,
            no_plots=no_plots,
            bh_returns=bh_returns,
        )
    except Exception as e:
        print(f"Warning: Performance metrics calculation failed: {e}")
        perf_metrics = {
            "annualized_strategy_return": np.nan,
            "annualized_strategy_volatility": np.nan,
            "strategy_sharpe_ratio": np.nan,
            "optimal_strategy_kelly_leverage_full": np.nan,
        }

    # 3. Get risk metrics
    try:
        risk_metrics = calculate_risk_metrics(test_results, model_name, log_prefix)
    except Exception as e:
        print(f"Warning: Risk metrics calculation failed: {e}")
        risk_metrics = {
            "max_drawdown": np.nan,
            "longest_drawdown_period_days": np.nan,
            "daily_var_95": np.nan,
            "daily_cvar_95": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }

    # 4. Combine all metrics into standard output format
    combined_metrics = {
        "hit_ratio": hit_ratio,
        "annual_return": perf_metrics.get("annualized_strategy_return", np.nan),
        "annual_volatility": perf_metrics.get("annualized_strategy_volatility", np.nan),
        "sharpe_ratio": perf_metrics.get("strategy_sharpe_ratio", np.nan),
        "kelly_leverage": perf_metrics.get(
            "optimal_strategy_kelly_leverage_full", np.nan
        ),
        "max_drawdown": risk_metrics.get("max_drawdown", np.nan),
        "longest_drawdown_days": risk_metrics.get(
            "longest_drawdown_period_days", np.nan
        ),
        "daily_var_95": risk_metrics.get("daily_var_95", np.nan),
        "daily_cvar_95": risk_metrics.get("daily_cvar_95", np.nan),
        "skewness": risk_metrics.get("skewness", np.nan),
        "kurtosis": risk_metrics.get("kurtosis", np.nan),
        # Buy-and-hold benchmark (computed by metrics.py)
        "bh_annual_return": perf_metrics.get("bh_annualized_return", np.nan),
        "bh_sharpe_ratio": perf_metrics.get("bh_sharpe_ratio", np.nan),
        "excess_return": perf_metrics.get("excess_return", np.nan),
        "excess_sharpe": perf_metrics.get("excess_sharpe", np.nan),
        "information_ratio": perf_metrics.get("information_ratio", np.nan),
    }

    return combined_metrics


def aggregate_metrics(iterations_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate distribution statistics for all metrics across iterations.

    Args:
        iterations_df: DataFrame with one row per iteration, containing all metrics

    Returns:
        Dict of dicts with statistics for each metric:
        {
            'hit_ratio': {
                'mean': 0.593,
                'std': 0.045,
                'min': 0.48,
                'max': 0.68,
                'median': 0.590,
                'percentile_25': 0.56,
                'percentile_75': 0.625,
                'percentile_95': 0.66
            },
            'sharpe_ratio': {...},
            ...
        }

    Example:
        >>> df = pd.DataFrame({'sharpe_ratio': [0.5, 0.6, 0.7, 0.8]})
        >>> stats = aggregate_metrics(df)
        >>> stats['sharpe_ratio']['mean']
        0.65
    """
    metric_columns = [
        "hit_ratio",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "kelly_leverage",
        "max_drawdown",
        "longest_drawdown_days",
        "daily_var_95",
        "daily_cvar_95",
        "skewness",
        "kurtosis",
        "bh_annual_return",
        "bh_sharpe_ratio",
        "excess_return",
        "excess_sharpe",
        "information_ratio",
    ]

    aggregated = {}

    for metric in metric_columns:
        if metric not in iterations_df.columns:
            print(f"Warning: Metric '{metric}' not found in iterations DataFrame")
            continue

        # Drop NaN values for statistics
        values = iterations_df[metric].dropna()

        if len(values) == 0:
            print(f"Warning: No valid values for metric '{metric}'")
            aggregated[metric] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "percentile_25": np.nan,
                "percentile_75": np.nan,
                "percentile_95": np.nan,
                "count": 0,
            }
            continue

        # Calculate statistics
        aggregated[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "percentile_25": float(values.quantile(0.25)),
            "percentile_75": float(values.quantile(0.75)),
            "percentile_95": float(values.quantile(0.95)),
            "count": int(len(values)),
        }

    return aggregated


def aggregate_parameter_statistics(
    iterations_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics on randomized parameters.

    Args:
        iterations_df: DataFrame with columns: lookback_days, train_split, embargo_pct

    Returns:
        Dict with statistics for each parameter:
        {
            'lookback_days': {'mean': 1095, 'std': 250, ...},
            'train_split': {'mean': 0.65, 'std': 0.08, ...},
            'embargo_pct': {'mean': 0.015, 'std': 0.005, ...}
        }
    """
    param_columns = ["lookback_days", "train_split", "embargo_pct"]

    aggregated = {}

    for param in param_columns:
        if param not in iterations_df.columns:
            continue

        values = iterations_df[param].dropna()

        aggregated[param] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }

    return aggregated
