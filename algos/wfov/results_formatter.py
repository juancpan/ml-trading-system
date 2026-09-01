"""
Results formatting and output for WFOV.

Handles CSV (incremental), JSON (summary), and statistics aggregation.

Author: jcp
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def save_iteration_to_csv(
    iteration: int,
    model_name: str,
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_days: int,
    train_split: float,
    embargo_pct: float,
    metrics: Dict[str, float],
    random_seed: int,
    filepath: Path,
    write_header: bool = False,
    validation_mode: str = "monte_carlo",
    regime: str = "unknown",
):
    """
    Append single iteration result to CSV file (incremental save).

    Args:
        iteration: Iteration number
        model_name: Model name
        ticker: Ticker symbol
        start_date: Window start date
        end_date: Window end date
        lookback_days: Lookback period used
        train_split: Train/test split ratio used
        embargo_pct: Embargo percentage used
        metrics: Dict of 11 performance metrics
        random_seed: Seed used for this iteration
        filepath: Path to CSV file
        write_header: If True, write CSV header (first iteration)
        validation_mode: Validation mode used (v2 enhancement)
        regime: Market regime for this window (v2 enhancement)

    CSV Columns (v2 - backward compatible):
        [Original 19 columns]
        iteration, model, ticker, start_date, end_date, lookback_days,
        train_split, embargo_pct, hit_ratio, annual_return, annual_volatility,
        sharpe_ratio, kelly_leverage, max_drawdown, longest_drawdown_days,
        daily_var_95, daily_cvar_95, skewness, kurtosis, random_seed,

        [New columns - backward compatible]
        validation_mode, regime
    """
    # Prepare row data (original 19 columns)
    row_data = {
        "iteration": iteration,
        "model": model_name,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "lookback_days": lookback_days,
        "train_split": train_split,
        "embargo_pct": embargo_pct,
        "hit_ratio": metrics.get("hit_ratio", np.nan),
        "annual_return": metrics.get("annual_return", np.nan),
        "annual_volatility": metrics.get("annual_volatility", np.nan),
        "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
        "kelly_leverage": metrics.get("kelly_leverage", np.nan),
        "max_drawdown": metrics.get("max_drawdown", np.nan),
        "longest_drawdown_days": metrics.get("longest_drawdown_days", np.nan),
        "daily_var_95": metrics.get("daily_var_95", np.nan),
        "daily_cvar_95": metrics.get("daily_cvar_95", np.nan),
        "skewness": metrics.get("skewness", np.nan),
        "kurtosis": metrics.get("kurtosis", np.nan),
        "bh_annual_return": metrics.get("bh_annual_return", np.nan),
        "bh_sharpe_ratio": metrics.get("bh_sharpe_ratio", np.nan),
        "excess_return": metrics.get("excess_return", np.nan),
        "excess_sharpe": metrics.get("excess_sharpe", np.nan),
        "information_ratio": metrics.get("information_ratio", np.nan),
        "random_seed": random_seed,
        # NEW: v2 enhancements (backward compatible - at end)
        "validation_mode": validation_mode,
        "regime": regime,
    }

    # Convert to DataFrame
    df = pd.DataFrame([row_data])

    # Write to CSV
    if write_header or not filepath.exists():
        df.to_csv(filepath, mode="w", index=False, float_format="%.6f")
    else:
        df.to_csv(filepath, mode="a", index=False, header=False, float_format="%.6f")


def save_iterations_batch_to_csv(
    rows: List[Dict],
    filepath: Path,
):
    """
    Write all iteration results to CSV in a single batch operation.

    Much faster than appending one row at a time (eliminates N file open/close cycles).

    Args:
        rows: List of row dicts (each with iteration, model, ticker, metrics, etc.)
        filepath: Path to CSV file
    """
    if not rows:
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(filepath, mode="w", index=False, float_format="%.6f")


def generate_summary_statistics(
    iterations_df: pd.DataFrame,
    model_name: str,
    ticker: str,
    iterations_requested: int,
    execution_time_seconds: float,
    master_seed: int,
    start_date: str,
    end_date: str,
    validation_mode: str = "monte_carlo",
    feature_config_hash: str = None,
    n_features: int = None,
    feature_names: list = None,
) -> Dict:
    """
    Generate comprehensive summary statistics from all iterations (v2 enhanced).

    Args:
        iterations_df: DataFrame with all iteration results
        model_name: Model name
        ticker: Ticker symbol
        iterations_requested: Total iterations requested
        execution_time_seconds: Total execution time
        master_seed: Master seed used
        start_date: Full date range start
        end_date: Full date range end
        feature_config_hash: Hash of the feature config used (for provenance tracking)
        n_features: Number of features used
        feature_names: List of feature column names

    Returns:
        Dict with:
        - metadata (model, ticker, iterations, time, seed, feature info)
        - parameter_statistics (lookback, train_split, embargo stats)
        - performance_metrics (11 metrics with full distribution stats)
        - best/worst iterations
    """
    # Count successful vs failed
    valid_rows = iterations_df.dropna(subset=["sharpe_ratio"])
    iterations_successful = len(valid_rows)
    iterations_failed = iterations_requested - iterations_successful

    # Metadata
    metadata = {
        "model_name": model_name,
        "ticker": ticker,
        "iterations_requested": iterations_requested,
        "iterations_successful": iterations_successful,
        "iterations_failed": iterations_failed,
        "success_rate": float(iterations_successful / iterations_requested)
        if iterations_requested > 0
        else 0.0,
        "execution_time_seconds": float(execution_time_seconds),
        "execution_time_minutes": float(execution_time_seconds / 60),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "master_seed": master_seed,
        "date_range_start": start_date,
        "date_range_end": end_date,
        "feature_config_hash": feature_config_hash,
        "n_features": n_features,
        "feature_names": feature_names,
    }

    # Parameter statistics
    param_stats = {}
    for param in ["lookback_days", "train_split", "embargo_pct"]:
        if param in iterations_df.columns:
            values = iterations_df[param].dropna()
            param_stats[param] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
            }

    # Performance metrics statistics
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

    performance_stats = {}
    for metric in metric_columns:
        if metric not in iterations_df.columns:
            continue

        values = iterations_df[metric].dropna()

        if len(values) == 0:
            performance_stats[metric] = {
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
        else:
            performance_stats[metric] = {
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

    # Find best/worst iterations
    best_iteration = None
    worst_iteration = None

    if "sharpe_ratio" in iterations_df.columns:
        valid_sharpe = iterations_df.dropna(subset=["sharpe_ratio"])
        if not valid_sharpe.empty:
            best_idx = valid_sharpe["sharpe_ratio"].idxmax()
            worst_idx = valid_sharpe["sharpe_ratio"].idxmin()

            best_iteration = {
                "iteration": int(valid_sharpe.loc[best_idx, "iteration"]),
                "sharpe_ratio": float(valid_sharpe.loc[best_idx, "sharpe_ratio"]),
                "hit_ratio": float(valid_sharpe.loc[best_idx, "hit_ratio"])
                if "hit_ratio" in valid_sharpe.columns
                else np.nan,
                "annual_return": float(valid_sharpe.loc[best_idx, "annual_return"])
                if "annual_return" in valid_sharpe.columns
                else np.nan,
            }

            worst_iteration = {
                "iteration": int(valid_sharpe.loc[worst_idx, "iteration"]),
                "sharpe_ratio": float(valid_sharpe.loc[worst_idx, "sharpe_ratio"]),
                "hit_ratio": float(valid_sharpe.loc[worst_idx, "hit_ratio"])
                if "hit_ratio" in valid_sharpe.columns
                else np.nan,
                "annual_return": float(valid_sharpe.loc[worst_idx, "annual_return"])
                if "annual_return" in valid_sharpe.columns
                else np.nan,
            }

    # NEW: Statistical rigor section (v2 enhancement)
    statistical_rigor = {}
    try:
        from algos.wfov.statistical_tests import compute_all_statistical_tests

        # Compute all statistical tests (significance, deflated Sharpe, etc.)
        stat_results = compute_all_statistical_tests(
            iterations_df,
            n_trials_global=1,  # Single model tested here; cross-model correction applied in model_ranker
        )

        statistical_rigor = stat_results

    except Exception as e:
        print(f"Warning: Could not compute statistical tests: {e}")
        statistical_rigor = {"error": str(e)}

    # NEW: Regime analysis section (v2 enhancement)
    regime_analysis = {}
    if "regime" in iterations_df.columns:
        try:
            from algos.wfov.regime_analyzer import (
                compute_regime_distribution,
                compute_regime_metrics_summary,
                detect_regime_dependent_strategy,
            )

            # Regime distribution
            regime_dist = compute_regime_distribution(iterations_df)

            # Performance by regime
            regime_metrics = compute_regime_metrics_summary(iterations_df)

            # Regime dependency detection
            regime_dep = detect_regime_dependent_strategy(
                regime_metrics, metric="sharpe_ratio", threshold_ratio=2.0
            )

            regime_analysis = {
                "regime_distribution": regime_dist,
                "performance_by_regime": regime_metrics,
                "regime_dependency": regime_dep,
            }

        except Exception as e:
            print(f"Warning: Could not compute regime analysis: {e}")
            regime_analysis = {"error": str(e)}

    # NEW: Validation mode info (v2 enhancement)
    is_walk_forward = validation_mode.startswith("walk_forward")
    validation_mode_info = {
        "mode": validation_mode,
        "description": _get_mode_description(validation_mode),
        "temporal_chain": is_walk_forward,
        "overlapping_test_periods": validation_mode == "monte_carlo",
        # Walk-forward has non-overlapping test periods → valid independence assumption
        "statistical_inference_valid": is_walk_forward,
        "validation_role": "inference" if is_walk_forward else "screening",
    }

    # Monte Carlo descriptors (screening metrics, not for statistical inference)
    mc_descriptors = {}
    if validation_mode == "monte_carlo" and not valid_rows.empty:
        sharpe_col = valid_rows["sharpe_ratio"]
        mc_descriptors = {
            "win_rate": float((sharpe_col > 0).mean()),
            "failure_rate": float(iterations_failed / max(iterations_requested, 1)),
            "sharpe_cv": float(abs(sharpe_col.std() / sharpe_col.mean()))
            if sharpe_col.mean() != 0
            else float("inf"),
        }

    # Combine all statistics (v1 structure + v2 enhancements)
    summary = {
        # V1 sections (backward compatible)
        "metadata": metadata,
        "parameter_statistics": param_stats,
        "performance_metrics": performance_stats,
        "best_iteration": best_iteration,
        "worst_iteration": worst_iteration,
        # V2 sections (new, backward compatible)
        "statistical_rigor": statistical_rigor,
        "regime_analysis": regime_analysis,
        "validation_mode_info": validation_mode_info,
        "mc_descriptors": mc_descriptors,
    }

    return summary


def _get_mode_description(validation_mode: str) -> str:
    """Get human-readable description of validation mode."""
    descriptions = {
        "monte_carlo": "Monte Carlo random sampling with stratified quartiles",
        "walk_forward_expanding": "Walk-forward expanding window (growing training set)",
        "walk_forward_rolling": "Walk-forward rolling window (fixed-size sliding window)",
    }
    return descriptions.get(validation_mode, "Unknown validation mode")


def save_summary_json(summary: Dict, filepath: Path):
    """
    Save summary statistics as formatted JSON.

    Args:
        summary: Summary dict from generate_summary_statistics()
        filepath: Path to JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {filepath}")


def format_console_summary(summary: Dict) -> str:
    """
    Format summary statistics for console output.

    Args:
        summary: Summary dict from generate_summary_statistics()

    Returns:
        Formatted string for console display
    """
    meta = summary["metadata"]
    params = summary["parameter_statistics"]
    perf = summary["performance_metrics"]

    lines = []
    lines.append("\n" + "━" * 80)
    lines.append("WFOV SESSION COMPLETE!")
    lines.append("━" * 80)
    lines.append(f"\nModel: {meta['model_name']} | Ticker: {meta['ticker']}")
    lines.append(f"Date Range: {meta['date_range_start']} to {meta['date_range_end']}")
    lines.append(f"Execution Time: {meta['execution_time_minutes']:.1f} minutes")
    lines.append(f"\nResults:")
    lines.append(
        f"  Successful: {meta['iterations_successful']}/{meta['iterations_requested']} ({meta['success_rate'] * 100:.1f}%)"
    )
    lines.append(
        f"  Failed: {meta['iterations_failed']}/{meta['iterations_requested']} ({(1 - meta['success_rate']) * 100:.1f}%)"
    )

    # Parameter randomization
    lines.append(f"\nParameter Randomization:")
    if "lookback_days" in params:
        p = params["lookback_days"]
        lines.append(
            f"  Lookback Days:    {p['mean']:.0f} ± {p['std']:.0f} (range: {p['min']:.0f}-{p['max']:.0f})"
        )
    if "train_split" in params:
        p = params["train_split"]
        lines.append(
            f"  Train Split:      {p['mean']:.2f} ± {p['std']:.2f} (range: {p['min']:.2f}-{p['max']:.2f})"
        )
    if "embargo_pct" in params:
        p = params["embargo_pct"]
        lines.append(
            f"  Embargo %:        {p['mean'] * 100:.2f}% ± {p['std'] * 100:.2f}% (range: {p['min'] * 100:.1f}%-{p['max'] * 100:.1f}%)"
        )

    # Performance metrics
    lines.append(f"\nAverage Performance Metrics:")

    metric_display = [
        ("hit_ratio", "Hit Ratio:", False, ""),
        ("annual_return", "Annual Return:", True, "%"),
        ("annual_volatility", "Annual Volatility:", True, "%"),
        ("sharpe_ratio", "Sharpe Ratio:", False, ""),
        ("max_drawdown", "Max Drawdown:", True, "%"),
        ("longest_drawdown_days", "Longest DD (days):", False, ""),
        ("daily_var_95", "Daily VaR (95%):", True, "%"),
        ("daily_cvar_95", "Daily CVaR (95%):", True, "%"),
        ("skewness", "Skewness:", False, ""),
        ("kurtosis", "Kurtosis:", False, ""),
        ("bh_annual_return", "B&H Return:", True, "%"),
        ("bh_sharpe_ratio", "B&H Sharpe:", False, ""),
        ("excess_return", "Excess Return:", True, "%"),
        ("excess_sharpe", "Excess Sharpe:", False, ""),
        ("information_ratio", "Information Ratio:", False, ""),
    ]

    for metric_key, label, is_percent, unit in metric_display:
        if metric_key in perf and perf[metric_key]["count"] > 0:
            m = perf[metric_key]
            if is_percent:
                mean_val = m["mean"] * 100
                std_val = m["std"] * 100
                median_val = m["median"] * 100
                lines.append(
                    f"  {label:22} {mean_val:6.2f}% ± {std_val:5.2f}% (median: {median_val:6.2f}%)"
                )
            else:
                lines.append(
                    f"  {label:22} {m['mean']:6.3f} ± {m['std']:5.3f} (median: {m['median']:6.3f})"
                )

    # NEW: Statistical Rigor Section (v2)
    if "statistical_rigor" in summary and "error" not in summary["statistical_rigor"]:
        stat_rig = summary["statistical_rigor"]

        lines.append(f"\n📊 STATISTICAL RIGOR")
        lines.append("─" * 80)

        # Sharpe significance
        if "sharpe_significance" in stat_rig:
            sharpe_sig = stat_rig["sharpe_significance"]
            p_value = sharpe_sig.get("p_value_two_tailed", np.nan)
            ci = sharpe_sig.get("confidence_interval", (np.nan, np.nan))

            if p_value < 0.05:
                sig_label = "✓ SIGNIFICANT (p < 0.05)"
            elif p_value < 0.10:
                sig_label = "⚠ MARGINAL (p < 0.10)"
            else:
                sig_label = "✗ NOT SIGNIFICANT (p ≥ 0.10)"

            lines.append(f"  Sharpe Significance:     {sig_label}")
            lines.append(f"  95% Confidence Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
            lines.append(f"  p-value (two-tailed):    {p_value:.4f}")

        # Deflated Sharpe
        if "deflated_sharpe" in stat_rig:
            defl = stat_rig["deflated_sharpe"]
            if "deflated_sharpe" in defl:
                obs_sharpe = perf.get("sharpe_ratio", {}).get("mean", np.nan)
                defl_sharpe = defl["deflated_sharpe"]
                lines.append(
                    f"  Deflated Sharpe Ratio:   {defl_sharpe:.3f} (vs observed {obs_sharpe:.3f})"
                )
                lines.append(f"  Interpretation:          {defl['interpretation']}")

    # NEW: Regime Analysis Section (v2)
    if "regime_analysis" in summary and "error" not in summary["regime_analysis"]:
        regime_ana = summary["regime_analysis"]

        lines.append(f"\n🎯 REGIME ANALYSIS")
        lines.append("─" * 80)

        # Regime distribution
        if "regime_distribution" in regime_ana:
            regime_dist = regime_ana["regime_distribution"]
            if "regime_counts" in regime_dist:
                lines.append(f"  Regime Distribution:")
                for regime, count in regime_dist["regime_counts"].items():
                    pct = regime_dist["regime_percentages"].get(regime, 0)
                    lines.append(
                        f"    {regime:12s}: {count:3d} iterations ({pct:4.1f}%)"
                    )

        # Performance by regime
        if (
            "performance_by_regime" in regime_ana
            and "sharpe_ratio" in regime_ana["performance_by_regime"]
        ):
            regime_perf = regime_ana["performance_by_regime"]["sharpe_ratio"]
            lines.append(f"\n  Performance by Regime (Sharpe Ratio):")

            for regime in [
                "bull",
                "bear",
                "sideways",
                "low_vol",
                "normal",
                "high_vol",
                "mixed",
            ]:
                if regime in regime_perf and "error" not in regime_perf[regime]:
                    stats = regime_perf[regime]
                    mean = stats["mean"]
                    std = stats["std"]
                    count = stats["count"]

                    # Color code by performance
                    if mean > 0.5:
                        status = "✓"
                    elif mean > 0.0:
                        status = "○"
                    else:
                        status = "✗"

                    lines.append(
                        f"    {status} {regime.capitalize():12s}: {mean:6.3f} ± {std:.3f} (n={count})"
                    )

        # Regime dependency warning
        if "regime_dependency" in regime_ana:
            dep = regime_ana["regime_dependency"]
            if "warning" in dep:
                lines.append(f"\n  {dep['warning']}")

    # Best/worst iterations
    if summary.get("best_iteration"):
        best = summary["best_iteration"]
        lines.append(
            f"\nBest Iteration: #{best['iteration']} (Sharpe: {best['sharpe_ratio']:.2f}, Hit: {best.get('hit_ratio', np.nan):.2f})"
        )

    if summary.get("worst_iteration"):
        worst = summary["worst_iteration"]
        lines.append(
            f"Worst Iteration: #{worst['iteration']} (Sharpe: {worst['sharpe_ratio']:.2f}, Hit: {worst.get('hit_ratio', np.nan):.2f})"
        )

    lines.append("\n" + "━" * 80)

    return "\n".join(lines)


def load_iterations_csv(filepath: Path) -> pd.DataFrame:
    """
    Load iterations CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with all iteration results
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Iterations CSV not found: {filepath}")

    df = pd.DataFrame(filepath)
    return df


def generate_wfov_filename_base(
    model_name: str,
    ticker: str,
    iterations: int,
    start_date: str,
    end_date: str,
    timestamp: str = None,
) -> str:
    """
    Generate base filename for WFOV outputs.

    Args:
        model_name: Model name
        ticker: Ticker symbol
        iterations: Number of iterations
        start_date: Date range start
        end_date: Date range end
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Base filename: wfov_{model}_{ticker}_{iterations}iter_{start}_{end}_{timestamp}

    Example:
        >>> generate_wfov_filename_base('svm_optimized', 'SPY', 100, '2020-01-01', '2025-01-01')
        'wfov_svm_optimized_SPY_100iter_2020-01-01_2025-01-01_20251202_153045'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"wfov_{model_name}_{ticker}_{iterations}iter_{start_date}_{end_date}_{timestamp}"
