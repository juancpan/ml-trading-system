"""
Market regime detection and analysis for WFOV validation framework.

Implements multiple regime classification methods:
- Volatility-based (realized vol quantiles)
- Trend-based (SMA crossover)
- Combined (volatility + trend)

Author: jcp
Date: 2025-12-03
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def detect_market_regimes(
    data: pd.DataFrame, method: str = "volatility_quantile", window: int = 30
) -> pd.Series:
    """
    Classify each date into market regime.

    Args:
        data: DataFrame with 'returns' and 'price' columns
        method: Regime detection method:
            - 'volatility_quantile': Use realized vol percentiles (default)
            - 'trend_sma': Use SMA crossover
            - 'combined': Both volatility + trend
        window: Rolling window for calculations (default: 30 days)

    Returns:
        Series with regime labels:
        - Volatility method: 'low_vol', 'normal', 'high_vol'
        - Trend method: 'bull', 'bear', 'sideways'
        - Combined: 'bull_high_vol', 'bear_low_vol', etc.

    Example:
        >>> data = pd.DataFrame({
        ...     'returns': np.random.normal(0, 0.02, 500),
        ...     'price': 100 * (1 + np.random.normal(0, 0.02, 500)).cumprod()
        ... })
        >>> regimes = detect_market_regimes(data, method='volatility_quantile')
        >>> regimes.value_counts()
        normal      250
        high_vol    125
        low_vol     125
    """
    if "returns" not in data.columns:
        raise ValueError("Data must contain 'returns' column")

    if method == "volatility_quantile":
        return _detect_volatility_regimes(data, window)

    elif method == "trend_sma":
        return _detect_trend_regimes(data, sma_window=200)

    elif method == "combined":
        vol_regimes = _detect_volatility_regimes(data, window)
        trend_regimes = _detect_trend_regimes(data, sma_window=200)

        # Combine: trend_volatility
        combined = trend_regimes.astype(str) + "_" + vol_regimes.astype(str)
        return combined

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'volatility_quantile', 'trend_sma', or 'combined'"
        )


def _detect_volatility_regimes(data: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Detect volatility regimes using rolling realized volatility.

    Classifies into terciles: low_vol (< 33rd pct), normal (33-67th), high_vol (> 67th)

    Args:
        data: DataFrame with 'returns' column
        window: Rolling window for volatility calculation

    Returns:
        Series with regime labels: 'low_vol', 'normal', 'high_vol'
    """
    returns = data["returns"]

    # Calculate realized volatility (annualized)
    rolling_vol = returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

    # Use expanding-window quantiles to prevent lookahead bias:
    # At each point in time, thresholds are based only on past data.
    expanding_33 = rolling_vol.expanding(min_periods=window).quantile(0.33)
    expanding_67 = rolling_vol.expanding(min_periods=window).quantile(0.67)

    # Classify
    regimes = pd.Series("normal", index=data.index)
    regimes[rolling_vol <= expanding_33] = "low_vol"
    regimes[rolling_vol > expanding_67] = "high_vol"

    # Handle NaN at start (before rolling window fills)
    regimes[rolling_vol.isna()] = "unknown"
    # Mark early periods where expanding window is insufficient
    regimes[expanding_33.isna()] = "unknown"

    return regimes


def _detect_trend_regimes(
    data: pd.DataFrame, sma_window: int = 200, threshold_pct: float = 0.05
) -> pd.Series:
    """
    Detect trend regimes using Simple Moving Average.

    Classifies based on price relative to SMA:
    - Bull: price > SMA * (1 + threshold)
    - Bear: price < SMA * (1 - threshold)
    - Sideways: within threshold band

    Args:
        data: DataFrame with 'price' column
        sma_window: SMA window (default: 200 days)
        threshold_pct: Percentage threshold for bull/bear (default: 5%)

    Returns:
        Series with regime labels: 'bull', 'bear', 'sideways'
    """
    if "price" not in data.columns:
        raise ValueError("Data must contain 'price' column for trend detection")

    price = data["price"]

    # Calculate SMA
    sma = price.rolling(sma_window, min_periods=sma_window // 2).mean()

    # Classify based on price relative to SMA
    regimes = pd.Series("sideways", index=data.index)
    regimes[price > sma * (1 + threshold_pct)] = "bull"
    regimes[price < sma * (1 - threshold_pct)] = "bear"

    # Handle NaN at start
    regimes[sma.isna()] = "unknown"

    return regimes


def assign_window_regime(
    window_start: str, window_end: str, regimes: pd.Series, threshold: float = 0.6
) -> str:
    """
    Assign dominant regime for a backtest window.

    If no regime has > threshold proportion, labeled as 'mixed'.

    Args:
        window_start: Window start date (YYYY-MM-DD)
        window_end: Window end date (YYYY-MM-DD)
        regimes: Series with date index and regime labels
        threshold: Threshold for dominance (default: 0.6 = 60%)

    Returns:
        Dominant regime label or 'mixed'

    Example:
        >>> regimes = pd.Series(
        ...     ['bull'] * 70 + ['bear'] * 30,
        ...     index=pd.date_range('2020-01-01', periods=100)
        ... )
        >>> regime = assign_window_regime('2020-01-01', '2020-04-09', regimes)
        >>> regime
        'bull'  # 70% bull > 60% threshold
    """
    try:
        window_regimes = regimes.loc[window_start:window_end]

        if len(window_regimes) == 0:
            return "unknown"

        # Count regime frequencies
        regime_counts = window_regimes.value_counts()

        # Filter out 'unknown' if present
        if "unknown" in regime_counts.index:
            regime_counts = regime_counts.drop("unknown")

        if len(regime_counts) == 0:
            return "unknown"

        # Check if dominant regime exceeds threshold
        dominant_regime = regime_counts.index[0]
        dominant_proportion = regime_counts.iloc[0] / len(window_regimes)

        if dominant_proportion > threshold:
            return dominant_regime
        else:
            return "mixed"

    except Exception:
        return "unknown"


def analyze_performance_by_regime(
    iterations_df: pd.DataFrame, metric_name: str = "sharpe_ratio"
) -> Dict:
    """
    Break down performance by market regime.

    Args:
        iterations_df: DataFrame with iteration results (must have 'regime' column)
        metric_name: Metric to analyze (default: 'sharpe_ratio')

    Returns:
        Dict with regime statistics:
        {
            'bull': {'mean': 1.2, 'std': 0.3, 'median': 1.1, 'count': 35},
            'bear': {'mean': 0.1, 'std': 0.4, 'median': 0.2, 'count': 28},
            ...
        }

    Example:
        >>> regime_perf = analyze_performance_by_regime(df, 'sharpe_ratio')
        >>> regime_perf['bull']['mean']
        1.2
    """
    if "regime" not in iterations_df.columns:
        return {"error": "No regime column found in iterations DataFrame"}

    if metric_name not in iterations_df.columns:
        return {"error": f"Metric {metric_name} not found in iterations DataFrame"}

    regime_stats = {}

    # Get unique regimes (excluding unknown)
    unique_regimes = iterations_df["regime"].unique()
    unique_regimes = [r for r in unique_regimes if r not in ["unknown", None, np.nan]]

    for regime in unique_regimes:
        regime_data = iterations_df[iterations_df["regime"] == regime][metric_name]
        regime_data = regime_data.dropna()

        if len(regime_data) > 0:
            regime_stats[regime] = {
                "mean": float(regime_data.mean()),
                "std": float(regime_data.std()),
                "median": float(regime_data.median()),
                "min": float(regime_data.min()),
                "max": float(regime_data.max()),
                "count": int(len(regime_data)),
            }

    return regime_stats


def compute_regime_distribution(iterations_df: pd.DataFrame) -> Dict:
    """
    Compute distribution of regimes across iterations.

    Args:
        iterations_df: DataFrame with 'regime' column

    Returns:
        Dict with:
        {
            'total_iterations': int,
            'regime_counts': {'bull': 35, 'bear': 28, ...},
            'regime_percentages': {'bull': 35.0, 'bear': 28.0, ...}
        }

    Example:
        >>> dist = compute_regime_distribution(df)
        >>> dist['regime_percentages']['bull']
        35.0
    """
    if "regime" not in iterations_df.columns:
        return {
            "error": "No regime column found",
            "total_iterations": len(iterations_df),
            "regime_counts": {},
            "regime_percentages": {},
        }

    regime_counts = iterations_df["regime"].value_counts().to_dict()

    # Remove unknown if present
    if "unknown" in regime_counts:
        del regime_counts["unknown"]

    total = len(iterations_df)
    regime_percentages = {
        regime: (count / total) * 100 for regime, count in regime_counts.items()
    }

    return {
        "total_iterations": total,
        "regime_counts": regime_counts,
        "regime_percentages": regime_percentages,
    }


def identify_regime_transitions(
    regimes: pd.Series, window_before: int = 5, window_after: int = 5
) -> pd.DataFrame:
    """
    Identify dates where market regime transitions occur.

    Args:
        regimes: Series with regime labels
        window_before: Days before transition to include
        window_after: Days after transition to include

    Returns:
        DataFrame with columns:
        - transition_date
        - regime_before
        - regime_after
        - transition_type (e.g., 'bull_to_bear')

    Example:
        >>> transitions = identify_regime_transitions(regimes)
        >>> transitions.head()
                       transition_date regime_before regime_after transition_type
        0        2020-03-15          bull         bear    bull_to_bear
        1        2020-06-10          bear         bull    bear_to_bull
    """
    transitions = []

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != regimes.iloc[i - 1]:
            # Regime changed
            transition_date = regimes.index[i]
            regime_before = regimes.iloc[i - 1]
            regime_after = regimes.iloc[i]

            # Skip unknown regimes
            if regime_before == "unknown" or regime_after == "unknown":
                continue

            transitions.append(
                {
                    "transition_date": transition_date,
                    "regime_before": regime_before,
                    "regime_after": regime_after,
                    "transition_type": f"{regime_before}_to_{regime_after}",
                }
            )

    return pd.DataFrame(transitions)


def get_regime_statistics_summary(regimes: pd.Series, data: pd.DataFrame) -> Dict:
    """
    Get comprehensive regime statistics for the full dataset.

    Args:
        regimes: Series with regime classifications
        data: DataFrame with 'returns' and 'price'

    Returns:
        Dict with statistics per regime (returns, volatility, duration)

    Example:
        >>> summary = get_regime_statistics_summary(regimes, data)
        >>> summary['bull']['avg_return_annual']
        0.15  # 15% annual return in bull regime
    """
    regime_summary = {}

    unique_regimes = [r for r in regimes.unique() if r not in ["unknown", None]]

    for regime in unique_regimes:
        regime_mask = regimes == regime
        regime_returns = data.loc[regime_mask, "returns"]

        if len(regime_returns) > 0:
            # Calculate statistics
            avg_return_daily = regime_returns.mean()
            avg_return_annual = avg_return_daily * 252
            volatility_daily = regime_returns.std()
            volatility_annual = volatility_daily * np.sqrt(252)

            # Duration statistics
            regime_days = regime_mask.sum()
            regime_pct = (regime_days / len(regimes)) * 100

            regime_summary[regime] = {
                "days": int(regime_days),
                "percentage": float(regime_pct),
                "avg_return_daily": float(avg_return_daily),
                "avg_return_annual": float(avg_return_annual),
                "volatility_daily": float(volatility_daily),
                "volatility_annual": float(volatility_annual),
                "sharpe_regime": float(avg_return_annual / volatility_annual)
                if volatility_annual > 0
                else 0.0,
            }

    return regime_summary


def stratify_windows_by_regime(
    windows: List[Dict],
    regimes: pd.Series,
    target_distribution: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Stratify windows to ensure representative regime coverage.

    Args:
        windows: List of window dicts
        regimes: Series with regime classifications
        target_distribution: Target regime distribution (optional)
            E.g., {'bull': 0.33, 'bear': 0.33, 'sideways': 0.34}

    Returns:
        Tuple of (stratified_windows, actual_distribution)

    Example:
        >>> windows = [...]  # 100 windows
        >>> regimes = pd.Series(...)
        >>> stratified, dist = stratify_windows_by_regime(windows, regimes)
        >>> dist
        {'bull': 0.33, 'bear': 0.33, 'sideways': 0.34}
    """
    # Assign regimes to windows
    for window in windows:
        window["regime"] = assign_window_regime(
            window["start_date"], window["end_date"], regimes, threshold=0.6
        )

    # Count regime distribution
    regime_counts = {}
    for window in windows:
        regime = window.get("regime", "unknown")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    total = len(windows)
    actual_distribution = {
        regime: count / total for regime, count in regime_counts.items()
    }

    # If target distribution specified, resample windows
    if target_distribution is not None:
        stratified_windows = _resample_windows_to_target(
            windows, regime_counts, target_distribution
        )
    else:
        stratified_windows = windows

    return stratified_windows, actual_distribution


def _resample_windows_to_target(
    windows: List[Dict],
    current_counts: Dict[str, int],
    target_distribution: Dict[str, float],
) -> List[Dict]:
    """
    Resample windows to match target regime distribution.

    Args:
        windows: List of windows with 'regime' field
        current_counts: Current regime counts
        target_distribution: Target regime proportions

    Returns:
        Resampled list of windows
    """
    # Group windows by regime
    regime_windows = {}
    for window in windows:
        regime = window.get("regime", "unknown")
        if regime not in regime_windows:
            regime_windows[regime] = []
        regime_windows[regime].append(window)

    # Calculate target counts
    total_windows = len(windows)
    target_counts = {
        regime: int(total_windows * proportion)
        for regime, proportion in target_distribution.items()
    }

    # Resample each regime
    rng = np.random.default_rng(42)
    stratified_windows = []

    for regime, target_count in target_counts.items():
        if regime in regime_windows:
            available = regime_windows[regime]

            if len(available) >= target_count:
                # Sample without replacement
                sampled = rng.choice(
                    available, size=target_count, replace=False
                ).tolist()
            else:
                # Sample with replacement if not enough
                sampled = rng.choice(
                    available, size=target_count, replace=True
                ).tolist()

            stratified_windows.extend(sampled)

    return stratified_windows


def compute_regime_metrics_summary(
    iterations_df: pd.DataFrame, metrics: List[str] = None
) -> Dict:
    """
    Compute comprehensive regime-conditional metrics summary.

    Args:
        iterations_df: DataFrame with iteration results and 'regime' column
        metrics: List of metrics to analyze (default: all 11 WFOV metrics)

    Returns:
        Dict with regime breakdown for all metrics:
        {
            'sharpe_ratio': {
                'bull': {'mean': ..., 'std': ..., 'count': ...},
                'bear': {...},
                ...
            },
            'hit_ratio': {...},
            ...
        }

    Example:
        >>> summary = compute_regime_metrics_summary(df)
        >>> summary['sharpe_ratio']['bull']['mean']
        1.2
    """
    if metrics is None:
        # Default: all 11 WFOV metrics
        metrics = [
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
        ]

    regime_metrics = {}

    for metric in metrics:
        if metric in iterations_df.columns:
            regime_metrics[metric] = analyze_performance_by_regime(
                iterations_df, metric
            )

    return regime_metrics


def detect_regime_dependent_strategy(
    regime_metrics: Dict, metric: str = "sharpe_ratio", threshold_ratio: float = 2.0
) -> Dict:
    """
    Detect if strategy is heavily dependent on specific market regimes.

    Flags strategies that work well in one regime but fail in another.

    Args:
        regime_metrics: Output from analyze_performance_by_regime()
        metric: Metric to analyze (default: 'sharpe_ratio')
        threshold_ratio: Ratio threshold for flagging dependency (default: 2.0)

    Returns:
        Dict with:
        {
            'is_regime_dependent': bool,
            'best_regime': str,
            'worst_regime': str,
            'performance_ratio': float,  # best / worst
            'warning': str
        }

    Example:
        >>> dep = detect_regime_dependent_strategy(regime_metrics)
        >>> dep['is_regime_dependent']
        True
        >>> dep['warning']
        '⚠️  Strategy performs 3.2x better in bull markets (risk!)'
    """
    if metric not in regime_metrics:
        return {"error": f"Metric {metric} not found in regime metrics"}

    regime_perf = regime_metrics[metric]

    # Get valid regimes (exclude 'mixed', 'unknown')
    valid_regimes = {
        k: v
        for k, v in regime_perf.items()
        if k not in ["mixed", "unknown", "error"] and "mean" in v
    }

    if len(valid_regimes) < 2:
        return {
            "is_regime_dependent": False,
            "warning": "Insufficient regime diversity for dependency analysis",
        }

    # Find best and worst regimes
    best_regime = max(valid_regimes.keys(), key=lambda k: valid_regimes[k]["mean"])
    worst_regime = min(valid_regimes.keys(), key=lambda k: valid_regimes[k]["mean"])

    best_performance = valid_regimes[best_regime]["mean"]
    worst_performance = valid_regimes[worst_regime]["mean"]

    # Calculate ratio (handle negative/zero values)
    if worst_performance <= 0:
        performance_ratio = np.inf if best_performance > 0 else 1.0
        is_dependent = True
    else:
        performance_ratio = best_performance / worst_performance
        is_dependent = performance_ratio > threshold_ratio

    # Generate warning
    if is_dependent:
        warning = f"⚠️  Strategy performs {performance_ratio:.1f}x better in {best_regime} markets (regime-dependent risk!)"
    else:
        warning = f"✓ Balanced performance across regimes (robust)"

    return {
        "is_regime_dependent": bool(is_dependent),
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "best_performance": float(best_performance),
        "worst_performance": float(worst_performance),
        "performance_ratio": float(performance_ratio),
        "warning": warning,
        "metric_analyzed": metric,
    }
