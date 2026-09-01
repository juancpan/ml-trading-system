"""
Fixed Portfolio Validator - Test Specific Weights Over Time

Tests whether YOUR SPECIFIC portfolio weights perform consistently across time.
NO re-optimization - just tests the exact weights you provide.

Use Case:
    You optimized a portfolio on 2020-2025, got weights:
        IAU: 15%, NVDA: 12%, WELL: 10%, ...

    Question: "Will these EXACT weights work in 2021, 2022, 2023, 2024, 2025?"

    This validator: Tests those fixed weights on each year separately
    Reports: Consistency, stability, regime dependence

Difference from validate_portfolio_oos.py:
    validate_portfolio_oos.py: Tests optimization STRATEGY (re-optimizes each window)
    validate_fixed_portfolio.py: Tests SPECIFIC WEIGHTS (no optimization)

Usage Examples:
    # Example 1: Extract weights from log, test on yearly periods
    python validate_fixed_portfolio.py \\
        --log logs/portfolio_exploration_20260108_145404.log \\
        --portfolio max_sharpe \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --test-period-months 12

    # Example 2: Test HRP stability (quarterly)
    python validate_fixed_portfolio.py \\
        --log logs/portfolio_exploration_20260108_145404.log \\
        --portfolio hrp \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --test-period-months 3

    # Example 3: Manual weights (no log file)
    python validate_fixed_portfolio.py \\
        --weights "IAU:0.15,NVDA:0.12,WELL:0.10,WMT:0.08,..." \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --test-period-months 12

Author: Algorithmic Trading System
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import re
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats

from portimization import load_and_preprocess_data, BASE_LOG_DIR


def get_periods_per_year(interval: str = "1d") -> int:
    """Get annualization factor based on data interval."""
    interval_map = {
        "1d": 252,  # Daily trading days
        "1wk": 52,  # Weekly
        "1mo": 12,  # Monthly
        "1h": 252 * 7,  # Hourly (approximate)
    }
    return interval_map.get(interval, 252)


def get_min_periods(interval: str = "1d") -> dict:
    """Get minimum period requirements based on data interval."""
    # Returns dict with min_train_periods, min_test_periods
    interval_defaults = {
        "1d": {
            "min_train_periods": 252,
            "min_test_periods": 21,
        },  # 1 year train, 1 month test
        "1wk": {
            "min_train_periods": 52,
            "min_test_periods": 4,
        },  # 1 year train, 1 month test
        "1mo": {
            "min_train_periods": 12,
            "min_test_periods": 1,
        },  # 1 year train, 1 month test
        "1h": {
            "min_train_periods": 252 * 7,
            "min_test_periods": 168,
        },  # 1 year train, 1 week test
    }
    return interval_defaults.get(interval, interval_defaults["1d"])


# =============================================================================
# METRIC THRESHOLDS (based on academic literature and industry standards)
# =============================================================================
METRIC_THRESHOLDS = {
    "sharpe_ratio": {"excellent": 2.0, "good": 1.0, "acceptable": 0.5, "poor": 0.0},
    "sortino_ratio": {"excellent": 3.0, "good": 1.5, "acceptable": 0.7, "poor": 0.0},
    "calmar_ratio": {"excellent": 3.0, "good": 1.5, "acceptable": 0.5, "poor": 0.0},
    "omega_ratio": {"excellent": 2.0, "good": 1.5, "acceptable": 1.2, "poor": 1.0},
    "annual_return": {"excellent": 0.20, "good": 0.10, "acceptable": 0.05, "poor": 0.0},
    "annual_volatility": {
        "excellent": 0.10,
        "good": 0.15,
        "acceptable": 0.25,
        "poor": 0.40,
    },  # Lower is better
    "max_drawdown": {
        "excellent": -0.10,
        "good": -0.20,
        "acceptable": -0.30,
        "poor": -0.50,
    },  # Closer to 0 is better
    "var_95": {
        "excellent": -0.01,
        "good": -0.02,
        "acceptable": -0.03,
        "poor": -0.05,
    },  # Closer to 0 is better
    "cvar_95": {
        "excellent": -0.015,
        "good": -0.025,
        "acceptable": -0.04,
        "poor": -0.06,
    },  # Closer to 0 is better
    "win_rate": {"excellent": 0.55, "good": 0.52, "acceptable": 0.50, "poor": 0.45},
    "psr": {"excellent": 0.95, "good": 0.85, "acceptable": 0.70, "poor": 0.50},
    "concentration_top3": {
        "excellent": 0.30,
        "good": 0.50,
        "acceptable": 0.70,
        "poor": 0.90,
    },  # Lower is better
}


def calculate_sortino_ratio(returns, risk_free_rate=0.01, periods_per_year=252):
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility).

    Sortino = (Annual Return - Rf) / Downside Deviation
    Only penalizes negative returns, not upside volatility.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside, perfect Sortino

    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(
        periods_per_year
    )
    annual_return = returns.mean() * periods_per_year

    if downside_deviation == 0:
        return np.inf

    return (annual_return - risk_free_rate) / downside_deviation


def calculate_calmar_ratio(returns, max_drawdown, periods_per_year=252):
    """
    Calculate Calmar Ratio (annual return / max drawdown).

    Measures return per unit of drawdown risk.
    """
    annual_return = returns.mean() * periods_per_year

    if max_drawdown == 0:
        return np.inf if annual_return > 0 else 0

    return annual_return / abs(max_drawdown)


def calculate_omega_ratio(returns, threshold=0.0):
    """
    Calculate Omega Ratio (probability-weighted ratio of gains to losses).

    Omega = Sum(returns > threshold) / Sum(returns < threshold)

    Unlike Sharpe, considers the entire return distribution.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]

    sum_losses = losses.sum()

    if sum_losses == 0:
        return np.inf if gains.sum() > 0 else 1.0

    return gains.sum() / sum_losses


def calculate_var(returns, confidence=0.95):
    """
    Calculate Value at Risk at given confidence level.

    VaR = percentile of daily returns (negative value indicates loss)
    """
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns, confidence=0.95):
    """
    Calculate Conditional VaR (Expected Shortfall).

    CVaR = average of returns below VaR (expected loss in worst cases)
    """
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_psr(
    sharpe_observed, n_returns, sharpe_benchmark=0.0, skewness=0.0, kurtosis=3.0
):
    """
    Calculate Probabilistic Sharpe Ratio (Bailey & López de Prado).

    PSR = probability that true Sharpe > benchmark, given observed data.

    Accounts for:
    - Sample size (more data = higher confidence)
    - Non-normality (skewness, kurtosis)

    Reference: Bailey & López de Prado (2012)
    """
    if n_returns < 2:
        return 0.5  # Insufficient data

    # Standard error of Sharpe ratio (accounting for non-normality)
    se_sharpe = np.sqrt(
        (
            1
            + 0.5 * sharpe_observed**2
            - skewness * sharpe_observed
            + (kurtosis - 3) / 4 * sharpe_observed**2
        )
        / (n_returns - 1)
    )

    if se_sharpe == 0:
        return 1.0 if sharpe_observed > sharpe_benchmark else 0.0

    # Z-score
    z = (sharpe_observed - sharpe_benchmark) / se_sharpe

    # Probability (using normal CDF)
    return stats.norm.cdf(z)


def calculate_concentration_top3(weights):
    """
    Calculate concentration in top 3 holdings.

    Lower is better (more diversified).
    """
    sorted_weights = sorted(weights.values(), reverse=True)
    top3_sum = sum(sorted_weights[:3])
    return top3_sum


def interpret_metric(metric_name, value):
    """
    Interpret a metric value against thresholds.

    Returns: (rating, emoji)
    """
    if metric_name not in METRIC_THRESHOLDS:
        return ("N/A", "•")

    thresholds = METRIC_THRESHOLDS[metric_name]

    # For metrics where lower is better
    lower_is_better = metric_name in [
        "annual_volatility",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "concentration_top3",
    ]

    if lower_is_better:
        if value <= thresholds["excellent"]:
            return ("Excellent", "✓")
        elif value <= thresholds["good"]:
            return ("Good", "✓")
        elif value <= thresholds["acceptable"]:
            return ("Acceptable", "•")
        else:
            return ("Poor", "✗")
    else:
        if value >= thresholds["excellent"]:
            return ("Excellent", "✓")
        elif value >= thresholds["good"]:
            return ("Good", "✓")
        elif value >= thresholds["acceptable"]:
            return ("Acceptable", "•")
        else:
            return ("Poor", "✗")


# Logging setup
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(BASE_LOG_DIR, f"fixed_portfolio_validation_{TIMESTAMP}.log")

logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def extract_portfolio_from_log(log_path, portfolio_name="max_sharpe"):
    """
    Extract portfolio weights from portfolio_exploration_global.py log file.

    Parses the "All Holdings" section for the specified portfolio.

    Parameters:
    -----------
    log_path : str
        Path to portfolio_exploration log
    portfolio_name : str
        'max_sharpe', 'hrp', or 'min_volatility'

    Returns:
    --------
    dict {asset: weight} or None if not found
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Find the portfolio section
    portfolio_header = f"{portfolio_name.upper().replace('_', ' ')}:"
    in_section = False
    in_holdings = False
    weights = {}

    for i, line in enumerate(lines):
        # Start of portfolio section
        if portfolio_header in line:
            in_section = True
            continue

        # Start of holdings subsection
        if in_section and "All Holdings:" in line:
            in_holdings = True
            continue

        # End of holdings (next portfolio or section)
        if in_holdings and (
            "MIN VOLATILITY:" in line
            or "HRP:" in line
            or "MAX SHARPE:" in line
            or "ASSET OVERLAP" in line
        ):
            break

        # Parse holding line
        if in_holdings and line.strip():
            # Format: "    1. NVDA                             0.1234 ( 12.34%)"
            match = re.search(r"\d+\.\s+(\S+)\s+(\d+\.\d+)\s+\(", line)
            if match:
                asset = match.group(1)
                weight = float(match.group(2))
                weights[asset] = weight

    if weights:
        # Normalize to ensure sum = 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return weights
    else:
        return None


def parse_manual_weights(weights_string):
    """
    Parse manually specified weights.

    Format: "NVDA:0.15,IAU:0.12,WELL:0.10,..."

    Returns:
    --------
    dict {asset: weight}
    """
    weights = {}

    for pair in weights_string.split(","):
        pair = pair.strip()
        if ":" in pair:
            asset, weight = pair.split(":")
            weights[asset.strip()] = float(weight.strip())

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def calculate_period_metrics(
    portfolio_weights, period_returns, risk_free_rate=0.01, periods_per_year=252
):
    """
    Calculate comprehensive portfolio performance metrics for a specific time period.

    Parameters:
    -----------
    portfolio_weights : dict
        {asset: weight}
    period_returns : pd.DataFrame
        Log returns for this period
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Annualization factor (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    dict with performance metrics including:
        - Basic: sharpe, annual_return, annual_volatility, total_return
        - Risk: max_drawdown, var_95, cvar_95
        - Risk-adjusted: sortino, calmar, omega
        - Statistical: win_rate, psr, skewness, kurtosis
        - Coverage: assets_available, assets_total, coverage
    """
    # Build weight array
    weight_array = np.zeros(len(period_returns.columns))
    assets_found = 0

    for i, asset in enumerate(period_returns.columns):
        if asset in portfolio_weights:
            weight_array[i] = portfolio_weights[asset]
            assets_found += 1

    # Normalize (handle missing assets)
    if weight_array.sum() > 0:
        weight_array = weight_array / weight_array.sum()
    else:
        return None

    # Portfolio returns
    portfolio_returns = (period_returns * weight_array).sum(axis=1)

    # Annualized metrics
    mean_ret = portfolio_returns.mean() * periods_per_year
    std_ret = portfolio_returns.std() * np.sqrt(periods_per_year)
    sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0

    # Cumulative
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0

    # Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (
        (portfolio_returns > 0).sum() / len(portfolio_returns)
        if len(portfolio_returns) > 0
        else 0
    )

    # Volatility
    daily_vol = portfolio_returns.std()

    # === NEW METRICS ===

    # Sortino Ratio (downside deviation)
    sortino = calculate_sortino_ratio(
        portfolio_returns, risk_free_rate, periods_per_year
    )

    # Calmar Ratio (return / max drawdown)
    calmar = calculate_calmar_ratio(portfolio_returns, max_drawdown, periods_per_year)

    # Omega Ratio (probability-weighted gains/losses)
    omega = calculate_omega_ratio(portfolio_returns, threshold=0.0)

    # VaR and CVaR (95% confidence)
    var_95 = calculate_var(portfolio_returns, confidence=0.95)
    cvar_95 = calculate_cvar(portfolio_returns, confidence=0.95)

    # Skewness and Kurtosis (for PSR calculation)
    skewness = stats.skew(portfolio_returns) if len(portfolio_returns) > 2 else 0
    kurtosis = (
        stats.kurtosis(portfolio_returns, fisher=False)
        if len(portfolio_returns) > 3
        else 3
    )  # Excess kurtosis

    # Probabilistic Sharpe Ratio
    psr = calculate_psr(
        sharpe,
        len(portfolio_returns),
        sharpe_benchmark=0.0,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Handle infinities for serialization
    sortino = min(sortino, 99.99) if np.isfinite(sortino) else 99.99
    calmar = min(calmar, 99.99) if np.isfinite(calmar) else 99.99
    omega = min(omega, 99.99) if np.isfinite(omega) else 99.99

    return {
        # Basic metrics
        "sharpe": sharpe,
        "annual_return": mean_ret,
        "annual_volatility": std_ret,
        "total_return": total_return,
        "daily_volatility": daily_vol,
        # Risk metrics
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        # Risk-adjusted metrics
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        # Statistical metrics
        "win_rate": win_rate,
        "psr": psr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        # Coverage metrics
        "n_days": len(period_returns),
        "assets_available": assets_found,
        "assets_total": len(portfolio_weights),
        "coverage": assets_found / len(portfolio_weights)
        if len(portfolio_weights) > 0
        else 0,
    }


def validate_fixed_portfolio(
    weights, returns_df, test_period_months=12, risk_free_rate=0.01, interval="1d"
):
    """
    Test fixed portfolio weights on non-overlapping time periods.

    Parameters:
    -----------
    weights : dict
        {asset: weight} - FIXED weights to test
    returns_df : pd.DataFrame
        Full returns data
    test_period_months : int
        Size of each test period in months
    risk_free_rate : float
        Annual risk-free rate
    interval : str
        Data interval ('1d', '1wk', '1mo')

    Returns:
    --------
    list of dicts with period results
    """
    # Get interval-specific settings
    periods_per_year = get_periods_per_year(interval)
    min_periods = get_min_periods(interval)

    logging.info("\n" + "=" * 80)
    logging.info(" FIXED PORTFOLIO CONSISTENCY TEST")
    logging.info("=" * 80)
    logging.info(f"\nPortfolio Weights ({len(weights)} assets):")

    # Show weights
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for i, (asset, weight) in enumerate(sorted_weights[:10], 1):
        logging.info(f"  {i:2d}. {asset:20s} {weight:6.2%}")
    if len(weights) > 10:
        logging.info(f"  ... ({len(weights) - 10} more)")

    logging.info(f"\nTest Configuration:")
    logging.info(f"  Test Period Size:   {test_period_months} months")
    logging.info(f"  Interval:           {interval} ({periods_per_year} periods/year)")
    logging.info(
        f"  Date Range:         {returns_df.index[0].date()} to {returns_df.index[-1].date()}"
    )
    logging.info(f"  Total Periods:      {len(returns_df)}")

    # Check asset coverage UPFRONT before running any periods
    csv_columns = set(returns_df.columns)
    portfolio_assets = set(weights.keys())
    matched_assets = portfolio_assets & csv_columns
    missing_assets = portfolio_assets - csv_columns
    coverage_pct = (
        len(matched_assets) / len(portfolio_assets) * 100 if portfolio_assets else 0
    )

    logging.info(f"\nAsset Coverage Check:")
    logging.info(f"  Portfolio Assets:   {len(portfolio_assets)}")
    logging.info(f"  CSV Columns:        {len(csv_columns)}")
    logging.info(f"  Matched:            {len(matched_assets)} ({coverage_pct:.1f}%)")

    if matched_assets:
        logging.info(f"\n  ✅ MATCHED ASSETS ({len(matched_assets)}):")
        matched_sorted = sorted(matched_assets, key=lambda a: weights[a], reverse=True)
        for asset in matched_sorted[:10]:
            logging.info(f"      + {asset} (weight: {weights[asset]:.2%})")
        if len(matched_assets) > 10:
            logging.info(f"      ... and {len(matched_assets) - 10} more")
        matched_weight = sum(weights[a] for a in matched_assets)
        logging.info(f"      Total matched weight: {matched_weight:.2%}")

    if missing_assets:
        logging.info(f"\n  ❌ MISSING ASSETS ({len(missing_assets)}):")
        missing_sorted = sorted(missing_assets, key=lambda a: weights[a], reverse=True)
        for asset in missing_sorted[:10]:
            logging.info(f"      - {asset} (weight: {weights[asset]:.2%})")
        if len(missing_assets) > 10:
            logging.info(f"      ... and {len(missing_assets) - 10} more")
        missing_weight = sum(weights[a] for a in missing_assets)
        logging.info(f"      Total missing weight: {missing_weight:.2%}")

    if coverage_pct < 50:
        logging.error(f"\n🚨 CRITICAL: Asset coverage is only {coverage_pct:.1f}%!")
        logging.error(f"   This means the validation is testing a DIFFERENT portfolio")
        logging.error(f"   than what was optimized. Results will be INVALID.")
        logging.error(
            f"\n   Likely cause: Wrong CSV file (optimization used different data)"
        )
        logging.error(
            f"   Fix: Use the same CSV that was used for portfolio optimization"
        )
        logging.error(f"\n   Continuing anyway, but results should be discarded...\n")
    elif coverage_pct < 80:
        logging.warning(f"\n⚠️  WARNING: Asset coverage is only {coverage_pct:.1f}%")
        logging.warning(f"   Some portfolio assets are missing from the CSV.")
        logging.warning(
            f"   Results may not accurately represent the optimized portfolio.\n"
        )

    # Split returns into non-overlapping periods
    results = []
    period_start = returns_df.index[0]
    period_num = 1

    while True:
        period_end = period_start + relativedelta(months=test_period_months)

        # Check if we have data for this period
        if period_end > returns_df.index[-1]:
            # If no periods have been evaluated yet, use all remaining data
            # so that short date ranges still produce at least one result.
            if not results:
                period_end = returns_df.index[-1] + pd.Timedelta(days=1)
                logging.warning(
                    f"  ⚠️  Requested {test_period_months}-month period exceeds "
                    f"available data. Using full range as a single period."
                )
            else:
                break

        # Extract period data
        period_returns = returns_df.loc[
            (returns_df.index >= period_start) & (returns_df.index < period_end)
        ].copy()

        if (
            len(period_returns) < min_periods["min_test_periods"]
        ):  # Minimum data based on interval
            logging.warning(
                f"  ⚠️  Period {period_num}: Insufficient data ({len(period_returns)} periods, need {min_periods['min_test_periods']}), skipping"
            )
            period_start = period_end
            period_num += 1
            continue  # Skip this period, don't break entire loop

        logging.info(f"\n{'=' * 80}")
        logging.info(f"PERIOD {period_num}")
        logging.info(f"{'=' * 80}")
        logging.info(f"  Date Range: {period_start.date()} to {period_end.date()}")
        logging.info(f"  Trading Days: {len(period_returns)}")

        # Calculate metrics
        metrics = calculate_period_metrics(
            weights, period_returns, risk_free_rate, periods_per_year
        )

        if metrics:
            # Basic performance
            logging.info(f"\n  Performance:")
            logging.info(f"    Sharpe Ratio:      {metrics['sharpe']:.3f}")
            logging.info(f"    Sortino Ratio:     {metrics['sortino']:.3f}")
            logging.info(f"    Annual Return:     {metrics['annual_return']:.3f}")
            logging.info(f"    Annual Volatility: {metrics['annual_volatility']:.3f}")
            logging.info(f"    Total Return:      {metrics['total_return']:.2%}")

            # Risk metrics
            logging.info(f"\n  Risk Metrics:")
            logging.info(f"    Max Drawdown:      {metrics['max_drawdown']:.2%}")
            logging.info(f"    VaR (95%):         {metrics['var_95']:.2%}")
            logging.info(f"    CVaR (95%):        {metrics['cvar_95']:.2%}")
            logging.info(f"    Calmar Ratio:      {metrics['calmar']:.3f}")
            logging.info(f"    Omega Ratio:       {metrics['omega']:.3f}")

            # Statistical
            logging.info(f"\n  Statistical:")
            logging.info(f"    Win Rate:          {metrics['win_rate']:.1%}")
            logging.info(f"    PSR:               {metrics['psr']:.1%}")
            logging.info(f"    Skewness:          {metrics['skewness']:.3f}")
            logging.info(f"    Kurtosis:          {metrics['kurtosis']:.3f}")

            logging.info(
                f"\n  Asset Coverage:    {metrics['assets_available']}/{metrics['assets_total']} ({metrics['coverage']:.1%})"
            )

            results.append(
                {
                    "period": period_num,
                    "start_date": period_start.strftime("%Y-%m-%d"),
                    "end_date": period_end.strftime("%Y-%m-%d"),
                    "n_days": metrics["n_days"],
                    # Basic metrics
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "annual_return": metrics["annual_return"],
                    "annual_volatility": metrics["annual_volatility"],
                    "total_return": metrics["total_return"],
                    # Risk metrics
                    "max_drawdown": metrics["max_drawdown"],
                    "var_95": metrics["var_95"],
                    "cvar_95": metrics["cvar_95"],
                    "calmar": metrics["calmar"],
                    "omega": metrics["omega"],
                    # Statistical
                    "win_rate": metrics["win_rate"],
                    "psr": metrics["psr"],
                    "skewness": metrics["skewness"],
                    "kurtosis": metrics["kurtosis"],
                    # Coverage
                    "coverage": metrics["coverage"],
                }
            )
        else:
            logging.warning(f"  ⚠️  No overlapping assets in this period")

        # Move to next period
        period_start = period_end
        period_num += 1

    return results


def summarize_fixed_portfolio_results(results, portfolio_weights=None):
    """
    Comprehensive summary of fixed portfolio consistency across periods.

    Includes:
    - Average metrics (return, volatility, max DD, VaR, CVaR)
    - Risk-adjusted ratios (Sharpe, Sortino, Calmar, Omega)
    - Statistical significance (t-test, confidence intervals, PSR)
    - Stability analysis (coefficient of variation)
    - Metric thresholds and interpretations
    """
    if not results:
        logging.error("\n❌ No results to summarize")
        return

    df = pd.DataFrame(results)
    n_periods = len(df)

    # ==========================================================================
    # AGGREGATE STATISTICS
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" COMPREHENSIVE METRICS SUMMARY")
    logging.info(f"{'=' * 80}")

    # Calculate aggregates for all metrics
    metrics_summary = {}

    # Basic metrics
    basic_metrics = [
        "sharpe",
        "sortino",
        "annual_return",
        "annual_volatility",
        "total_return",
    ]
    # Risk metrics
    risk_metrics = ["max_drawdown", "var_95", "cvar_95", "calmar", "omega"]
    # Statistical metrics
    stat_metrics = ["win_rate", "psr", "skewness", "kurtosis"]

    all_metrics = basic_metrics + risk_metrics + stat_metrics

    for metric in all_metrics:
        if metric in df.columns:
            metrics_summary[metric] = {
                "mean": df[metric].mean(),
                "std": df[metric].std(),
                "min": df[metric].min(),
                "max": df[metric].max(),
                "median": df[metric].median(),
            }

    # ==========================================================================
    # PERFORMANCE METRICS TABLE
    # ==========================================================================
    logging.info(f"\nPerformance Across {n_periods} Periods:")
    logging.info(
        f"\n{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Rating':<12}"
    )
    logging.info(f"{'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    # Map metric names to display names and threshold keys
    metric_display = {
        "sharpe": ("Sharpe Ratio", "sharpe_ratio"),
        "sortino": ("Sortino Ratio", "sortino_ratio"),
        "calmar": ("Calmar Ratio", "calmar_ratio"),
        "omega": ("Omega Ratio", "omega_ratio"),
        "annual_return": ("Annual Return", "annual_return"),
        "annual_volatility": ("Annual Volatility", "annual_volatility"),
        "max_drawdown": ("Max Drawdown", "max_drawdown"),
        "var_95": ("VaR (95%)", "var_95"),
        "cvar_95": ("CVaR (95%)", "cvar_95"),
        "win_rate": ("Win Rate", "win_rate"),
        "psr": ("PSR", "psr"),
    }

    for metric in [
        "sharpe",
        "sortino",
        "calmar",
        "omega",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "win_rate",
        "psr",
    ]:
        if metric in metrics_summary:
            m = metrics_summary[metric]
            display_name, threshold_key = metric_display.get(metric, (metric, metric))
            rating, emoji = interpret_metric(threshold_key, m["mean"])

            # Format values appropriately
            if metric in [
                "annual_return",
                "total_return",
                "max_drawdown",
                "var_95",
                "cvar_95",
                "win_rate",
                "psr",
            ]:
                mean_str = f"{m['mean']:.2%}"
                std_str = f"{m['std']:.2%}"
                min_str = f"{m['min']:.2%}"
                max_str = f"{m['max']:.2%}"
            else:
                mean_str = f"{m['mean']:.3f}"
                std_str = f"{m['std']:.3f}"
                min_str = f"{m['min']:.3f}"
                max_str = f"{m['max']:.3f}"

            logging.info(
                f"{display_name:<20} {mean_str:>10} {std_str:>10} {min_str:>10} {max_str:>10} {emoji} {rating:<10}"
            )

    # ==========================================================================
    # AVERAGE RETURN, VOLATILITY, MAX DD (User requested specifically)
    # ==========================================================================
    avg_return = metrics_summary["annual_return"]["mean"]
    avg_vol = metrics_summary["annual_volatility"]["mean"]
    avg_max_dd = metrics_summary["max_drawdown"]["mean"]
    worst_dd = metrics_summary["max_drawdown"]["min"]

    logging.info(f"\n{'=' * 80}")
    logging.info(" KEY AVERAGES (Across All Periods)")
    logging.info(f"{'=' * 80}")
    logging.info(f"\n  Avg Annual Return:     {avg_return:.2%}")
    logging.info(f"  Avg Annual Volatility: {avg_vol:.2%}")
    logging.info(f"  Avg Max Drawdown:      {avg_max_dd:.2%}")
    logging.info(f"  Worst Max Drawdown:    {worst_dd:.2%}")

    # Risk metrics averages
    if "var_95" in metrics_summary:
        avg_var = metrics_summary["var_95"]["mean"]
        avg_cvar = metrics_summary["cvar_95"]["mean"]
        logging.info(f"  Avg Daily VaR (95%):   {avg_var:.2%}")
        logging.info(f"  Avg Daily CVaR (95%):  {avg_cvar:.2%}")

    # ==========================================================================
    # STABILITY ANALYSIS
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" STABILITY ANALYSIS")
    logging.info(f"{'=' * 80}")

    avg_sharpe = metrics_summary["sharpe"]["mean"]
    std_sharpe = metrics_summary["sharpe"]["std"]

    # Coefficient of variation (CV) = std / mean
    cv = std_sharpe / avg_sharpe if avg_sharpe > 0 else 999

    logging.info(f"\n  Sharpe Ratio Statistics:")
    logging.info(f"    Mean:      {avg_sharpe:.3f}")
    logging.info(f"    Std Dev:   {std_sharpe:.3f}")
    logging.info(f"    CV:        {cv:.2f}")

    if cv < 0.15:
        stability_verdict = "✅ VERY STABLE"
        stability_detail = "Consistent across all periods"
    elif cv < 0.30:
        stability_verdict = "✅ STABLE"
        stability_detail = "Minor variation acceptable"
    elif cv < 0.50:
        stability_verdict = "⚠️  MODERATE"
        stability_detail = "Some regime dependence"
    else:
        stability_verdict = "🚨 UNSTABLE"
        stability_detail = "High regime dependence"

    logging.info(f"\n  Stability: {stability_verdict}")
    logging.info(f"    {stability_detail} (CV={cv:.2f})")

    # ==========================================================================
    # RISK ANALYSIS
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" RISK ANALYSIS")
    logging.info(f"{'=' * 80}")

    # Drawdown analysis
    logging.info(f"\n  Drawdown Analysis:")
    logging.info(f"    Average Max DD:  {avg_max_dd:.2%}")
    logging.info(f"    Worst Max DD:    {worst_dd:.2%}")

    if worst_dd > -0.20:
        dd_verdict = "✅ Low risk (worst DD > -20%)"
    elif worst_dd > -0.30:
        dd_verdict = "⚠️  Moderate risk (-30% < worst DD ≤ -20%)"
    else:
        dd_verdict = "🚨 High risk (worst DD ≤ -30%)"
    logging.info(f"    {dd_verdict}")

    # VaR/CVaR analysis
    if "var_95" in metrics_summary:
        logging.info(f"\n  Tail Risk Analysis:")
        logging.info(f"    Avg VaR (95%):   {avg_var:.2%} (expect 1 in 20 days worse)")
        logging.info(f"    Avg CVaR (95%):  {avg_cvar:.2%} (expected loss on bad days)")

        if avg_cvar > -0.025:
            tail_verdict = "✅ Low tail risk"
        elif avg_cvar > -0.04:
            tail_verdict = "⚠️  Moderate tail risk"
        else:
            tail_verdict = "🚨 High tail risk"
        logging.info(f"    {tail_verdict}")

    # ==========================================================================
    # PSR ANALYSIS (Probabilistic Sharpe Ratio)
    # ==========================================================================
    if "psr" in metrics_summary:
        avg_psr = metrics_summary["psr"]["mean"]
        logging.info(f"\n{'=' * 80}")
        logging.info(" PROBABILISTIC SHARPE RATIO (PSR)")
        logging.info(f"{'=' * 80}")
        logging.info(f"\n  Average PSR: {avg_psr:.1%}")
        logging.info(
            f"  Interpretation: {avg_psr:.0%} probability that true Sharpe > 0"
        )

        if avg_psr >= 0.95:
            psr_verdict = "✅ Very high confidence in positive Sharpe"
        elif avg_psr >= 0.85:
            psr_verdict = "✅ High confidence in positive Sharpe"
        elif avg_psr >= 0.70:
            psr_verdict = "⚠️  Moderate confidence - some luck possible"
        else:
            psr_verdict = "🚨 Low confidence - may be noise"

        logging.info(f"  {psr_verdict}")

    # ==========================================================================
    # PERIOD-BY-PERIOD TABLE (Enhanced)
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" PERIOD-BY-PERIOD BREAKDOWN")
    logging.info(f"{'=' * 80}")

    # Header with more metrics
    header = f"{'Period':<6} {'Date Range':<24} {'Sharpe':>7} {'Sortino':>8} {'Return':>7} {'Vol':>6} {'MaxDD':>7} {'VaR':>6} {'PSR':>6}"
    logging.info(f"\n{header}")
    logging.info(
        f"{'-' * 6} {'-' * 24} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 6} {'-' * 6}"
    )

    for _, row in df.iterrows():
        date_range = f"{row['start_date'][:10]} - {row['end_date'][:10]}"
        sortino = row.get("sortino", 0)
        var_95 = row.get("var_95", 0)
        psr = row.get("psr", 0)

        logging.info(
            f"P{int(row['period']):<5} {date_range:<24} "
            f"{row['sharpe']:>7.2f} {sortino:>8.2f} {row['annual_return']:>6.1%} "
            f"{row['annual_volatility']:>5.1%} {row['max_drawdown']:>6.1%} {var_95:>5.1%} {psr:>5.0%}"
        )

    # ==========================================================================
    # STATISTICAL SIGNIFICANCE
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" STATISTICAL SIGNIFICANCE")
    logging.info(f"{'=' * 80}")

    if n_periods >= 3:
        sharpe_values = df["sharpe"].values

        # T-test
        t_stat, p_value = stats.ttest_1samp(sharpe_values, 0)

        logging.info(f"\n  T-test (H0: Sharpe = 0):")
        logging.info(f"    t-statistic: {t_stat:.3f}")
        logging.info(f"    p-value:     {p_value:.4f}")

        if p_value < 0.01:
            sig_verdict = "✅ HIGHLY SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            sig_verdict = "✅ SIGNIFICANT (p < 0.05)"
        elif p_value < 0.10:
            sig_verdict = "⚠️  MARGINALLY SIGNIFICANT (p < 0.10)"
        else:
            sig_verdict = "🚨 NOT SIGNIFICANT (p ≥ 0.10)"

        logging.info(f"    {sig_verdict}")

        # Confidence interval
        if n_periods >= 2:
            ci_lower, ci_upper = stats.t.interval(
                0.95,
                len(sharpe_values) - 1,
                loc=np.mean(sharpe_values),
                scale=stats.sem(sharpe_values),
            )
            logging.info(
                f"\n  95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]"
            )

            if ci_lower > 0:
                logging.info(
                    f"    ✅ Lower bound > 0 - Positive Sharpe with high confidence"
                )
            elif ci_lower > -0.5:
                logging.info(f"    ⚠️  Lower bound near 0 - Some uncertainty")
            else:
                logging.warning(f"    🚨 Lower bound < -0.5 - High uncertainty")
    else:
        logging.info(
            f"\n  NOTE: Only {n_periods} periods - need ≥3 for statistical tests"
        )

    # ==========================================================================
    # DIVERSIFICATION ANALYSIS
    # ==========================================================================
    if portfolio_weights:
        logging.info(f"\n{'=' * 80}")
        logging.info(" DIVERSIFICATION ANALYSIS")
        logging.info(f"{'=' * 80}")

        concentration = calculate_concentration_top3(portfolio_weights)
        n_assets = len(portfolio_weights)

        logging.info(f"\n  Total Assets: {n_assets}")
        logging.info(f"  Top 3 Concentration: {concentration:.1%}")

        if concentration < 0.30:
            div_verdict = "✅ Well diversified (top 3 < 30%)"
        elif concentration < 0.50:
            div_verdict = "✅ Adequately diversified (top 3 < 50%)"
        elif concentration < 0.70:
            div_verdict = "⚠️  Moderately concentrated (top 3 < 70%)"
        else:
            div_verdict = "🚨 Highly concentrated (top 3 ≥ 70%)"

        logging.info(f"  {div_verdict}")

    # ==========================================================================
    # COMPOSITE SCORE
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" COMPOSITE SCORE")
    logging.info(f"{'=' * 80}")

    # Calculate composite score (0-100)
    score = 0
    max_score = 0

    # Sharpe contribution (0-30)
    if avg_sharpe >= 2.0:
        score += 30
    elif avg_sharpe >= 1.0:
        score += 20
    elif avg_sharpe >= 0.5:
        score += 10
    elif avg_sharpe > 0:
        score += 5
    max_score += 30

    # Stability contribution (0-20)
    if cv < 0.15:
        score += 20
    elif cv < 0.30:
        score += 15
    elif cv < 0.50:
        score += 8
    max_score += 20

    # Risk contribution (0-20)
    if worst_dd > -0.15:
        score += 20
    elif worst_dd > -0.25:
        score += 15
    elif worst_dd > -0.35:
        score += 8
    max_score += 20

    # Statistical significance contribution (0-15)
    if n_periods >= 3:
        if p_value < 0.01:
            score += 15
        elif p_value < 0.05:
            score += 10
        elif p_value < 0.10:
            score += 5
    max_score += 15

    # PSR contribution (0-15)
    if "psr" in metrics_summary:
        if avg_psr >= 0.95:
            score += 15
        elif avg_psr >= 0.85:
            score += 10
        elif avg_psr >= 0.70:
            score += 5
    max_score += 15

    pct_score = (score / max_score * 100) if max_score > 0 else 0

    logging.info(f"\n  Composite Score: {score}/{max_score} ({pct_score:.0f}%)")

    if pct_score >= 80:
        grade = "A - EXCELLENT"
    elif pct_score >= 65:
        grade = "B - GOOD"
    elif pct_score >= 50:
        grade = "C - ACCEPTABLE"
    elif pct_score >= 35:
        grade = "D - MARGINAL"
    else:
        grade = "F - POOR"

    logging.info(f"  Grade: {grade}")

    # ==========================================================================
    # FINAL RECOMMENDATION
    # ==========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info(" RECOMMENDATION")
    logging.info(f"{'=' * 80}")

    if pct_score >= 65 and avg_sharpe > 0.5 and cv < 0.50:
        logging.info(f"\n✅ PORTFOLIO IS ROBUST")
        logging.info(
            f"   Score: {pct_score:.0f}% | Sharpe: {avg_sharpe:.2f} | CV: {cv:.2f}"
        )
        logging.info(
            f"   Avg Return: {avg_return:.1%} | Avg Vol: {avg_vol:.1%} | Worst DD: {worst_dd:.1%}"
        )
        logging.info(f"\n   → DEPLOY with confidence for buy-and-hold strategy")
    elif pct_score >= 50 and avg_sharpe > 0.3:
        logging.info(f"\n⚠️  PORTFOLIO IS ACCEPTABLE")
        logging.info(
            f"   Score: {pct_score:.0f}% | Sharpe: {avg_sharpe:.2f} | CV: {cv:.2f}"
        )
        logging.info(
            f"   Avg Return: {avg_return:.1%} | Avg Vol: {avg_vol:.1%} | Worst DD: {worst_dd:.1%}"
        )
        logging.info(f"\n   → DEPLOY with caution, monitor closely")
        logging.info(f"   → Consider reoptimization if better alternatives exist")
    else:
        logging.error(f"\n🚨 PORTFOLIO IS NOT RECOMMENDED")
        logging.error(
            f"   Score: {pct_score:.0f}% | Sharpe: {avg_sharpe:.2f} | CV: {cv:.2f}"
        )
        logging.error(
            f"   Avg Return: {avg_return:.1%} | Avg Vol: {avg_vol:.1%} | Worst DD: {worst_dd:.1%}"
        )
        logging.error(f"\n   → DO NOT deploy for buy-and-hold")
        logging.error(f"   → Reoptimize or consider regime-adaptive strategy")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description="Test fixed portfolio weights across multiple time periods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test portfolio from log file (yearly periods)
  python %(prog)s --log logs/portfolio_exploration_20260108_145404.log \\
      --portfolio max_sharpe \\
      --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
      --start 2020-01-01 --end 2026-01-01 \\
      --test-period-months 12

  # Test HRP stability (quarterly)
  python %(prog)s --log logs/portfolio_exploration_20260108_145404.log \\
      --portfolio hrp \\
      --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
      --start 2020-01-01 --end 2026-01-01 \\
      --test-period-months 3

  # Manual weights (no log)
  python %(prog)s --weights "IAU:0.15,NVDA:0.12,WELL:0.10,WMT:0.08" \\
      --csv ../../data/your_data.csv \\
      --start 2020-01-01 --end 2026-01-01
        """,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--log", type=str, help="Path to portfolio_exploration log file"
    )
    input_group.add_argument(
        "--weights",
        type=str,
        help="Manual weights (format: ASSET1:WEIGHT1,ASSET2:WEIGHT2,...)",
    )

    # Data source parameters
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to CSV with price data"
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Read data from parquet market data store (no --csv needed)",
    )

    # Required parameters
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1wk", "1mo"],
        help="Data interval (default: 1d). Affects annualization: 1d=252, 1wk=52, 1mo=12",
    )

    # Optional parameters
    parser.add_argument(
        "--portfolio",
        type=str,
        default="max_sharpe",
        choices=["max_sharpe", "hrp", "min_volatility"],
        help="Portfolio name (if using --log, default: max_sharpe)",
    )
    parser.add_argument(
        "--test-period-months",
        type=int,
        default=12,
        help="Test period size in months (default: 12)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.01,
        help="Risk-free rate (default: 0.01)",
    )

    args = parser.parse_args()

    # Extract weights
    if args.log:
        logging.info(f"\nExtracting {args.portfolio} weights from log file...")
        weights = extract_portfolio_from_log(args.log, args.portfolio)

        if not weights:
            logging.error(f"Failed to extract {args.portfolio} from {args.log}")
            logging.error(
                f"Make sure the log contains '{args.portfolio.upper()}:' section"
            )
            return

        logging.info(f"✅ Extracted {len(weights)} assets")
    else:
        logging.info(f"\nParsing manual weights...")
        weights = parse_manual_weights(args.weights)

        if not weights:
            logging.error(f"Failed to parse weights: {args.weights}")
            logging.error(f"Format: ASSET1:WEIGHT1,ASSET2:WEIGHT2,...")
            return

        logging.info(f"✅ Parsed {len(weights)} assets")

    # Resolve data source: --from-store, --csv, or error
    data_path = args.csv
    if args.from_store:
        import tempfile
        from pathlib import Path

        try:
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from algos.common.market_data_store import MarketDataStore
            from algos.common.update_market_data import load_ticker_universe

            store = MarketDataStore()

            # Build reverse map: column_name -> yfinance_ticker
            # The log file uses column names (e.g., "Rolls_Royce") from
            # ticker_universe.json, but the store indexes by yfinance
            # tickers (e.g., "RR.L").  Resolve names before lookup.
            universe = load_ticker_universe()
            reverse_map = {v: k for k, v in universe.items()}

            tickers_map = {}
            for asset in weights.keys():
                yf_ticker = reverse_map.get(asset, asset)
                tickers_map[yf_ticker] = asset  # store key -> output col name

            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, prefix="fixed_portfolio_data_"
            )
            data_path = tmp.name
            tmp.close()

            export_df = store.get_multi_ticker_prices(
                list(tickers_map.keys()),
                args.start,
                args.end,
                "adj_close",
                column_names=tickers_map,
                min_coverage=0.5,
            )
            if export_df.empty:
                logging.error(
                    "Parquet store returned no data for fixed portfolio assets. "
                    "Run 'python -m algos.common.update_market_data --init' first."
                )
                return

            # Forward-fill cross-exchange NaN gaps so that tickers
            # from exchanges with different trading calendars (e.g.
            # TASE Sun-Thu vs NYSE Mon-Fri) are not penalised by
            # the downstream 80% coverage filter.
            export_df = export_df.ffill()

            export_df = export_df.reset_index().rename(columns={"index": "Date"})
            export_df.to_csv(data_path, encoding="utf-8", index=False)
            logging.info(
                f"Exported {export_df.shape} from parquet store to {data_path}"
            )
        except ImportError:
            logging.error("MarketDataStore not available. Use --csv instead.")
            return
    elif data_path is None:
        parser.error("Either --csv or --from-store is required.")

    # Load returns data
    logging.info(f"\nLoading price data...")
    returns_df = load_and_preprocess_data(data_path, args.start, args.end)

    if returns_df.empty:
        logging.error("Failed to load data")
        return

    logging.info(f"✅ Loaded {len(returns_df)} days, {len(returns_df.columns)} assets")

    # Run validation
    results = validate_fixed_portfolio(
        weights, returns_df, args.test_period_months, args.risk_free_rate, args.interval
    )

    if results:
        # Summary (pass weights for diversification analysis)
        summarize_fixed_portfolio_results(results, portfolio_weights=weights)

        # Save to CSV
        csv_output = os.path.join(
            BASE_LOG_DIR, f"fixed_portfolio_validation_{TIMESTAMP}.csv"
        )
        df = pd.DataFrame(results)
        df.to_csv(csv_output, index=False)

        logging.info(f"\n{'=' * 80}")
        logging.info(f" VALIDATION COMPLETE")
        logging.info(f"{'=' * 80}")
        logging.info(f" Log:     {LOG_FILE}")
        logging.info(f" Results: {csv_output}")
        logging.info(f"{'=' * 80}\n")
    else:
        logging.error("\n❌ Validation failed - no results")


if __name__ == "__main__":
    main()
