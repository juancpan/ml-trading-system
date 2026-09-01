"""
Out-of-Sample Portfolio Validation using Walk-Forward Testing

Tests whether portfolio optimization strategy holds up on unseen data.
Critical validation step before live deployment.

Walk-Forward Methodology:
    Expanding Window (Recommended):
        Train: 2020-2022 (2y) → Optimize → Test: 2023 (1y) → Record OOS
        Train: 2020-2023 (3y) → Optimize → Test: 2024 (1y) → Record OOS
        Train: 2020-2024 (4y) → Optimize → Test: 2025 (1y) → Record OOS

    Rolling Window (Regime Detection):
        Train: 2020-01-01 to 2023-01-01 (3y) → Test: 2023-01-05 to 2023-04-05 (3m)
        Train: 2020-04-01 to 2023-04-01 (3y) → Test: 2023-04-06 to 2023-07-06 (3m)
        Train: 2020-07-01 to 2023-07-01 (3y) → Test: 2023-07-06 to 2023-10-06 (3m)
        (steps forward by test_months, with embargo gap between train/test)

Key Metrics:
    - In-Sample Sharpe: Performance on training data
    - Out-of-Sample Sharpe: Performance on test data
    - Degradation: (IS - OOS) / IS (should be <20%)
    - Consistency: Std dev of OOS Sharpe across windows

Usage Examples:
    # Example 1: Standard validation (2y train, 1y test, expanding)
    python validate_portfolio_oos.py \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --portfolio max_sharpe \\
        --train-years 2 --test-years 1 --mode expanding

    # Example 2: Compare max_sharpe vs HRP
    python validate_portfolio_oos.py \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --portfolio both \\
        --train-years 2 --test-years 1

    # Example 3: Conservative (3y train, 1y test)
    python validate_portfolio_oos.py \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --train-years 3 --test-years 1

    # Example 4: Quarterly rebalancing (rolling window, 3y train, 3m test, 5d embargo)
    # Windows roll forward by test_months (3 months) each iteration:
    #   Window 1: Train 2020-01-01 to 2023-01-01, Test 2023-01-06 to 2023-04-06
    #   Window 2: Train 2020-04-01 to 2023-04-01, Test 2023-04-06 to 2023-07-06
    #   Window 3: Train 2020-07-01 to 2023-07-01, Test 2023-07-06 to 2023-10-06
    python validate_portfolio_oos.py \\
        --csv ../../data/financial_data_combined_prices_2020-01-01_2026-01-01_1d.csv \\
        --start 2020-01-01 --end 2026-01-01 \\
        --train-years 3 --test-months 3 --mode rolling --embargo-days 5

Author: Algorithmic Trading System
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats

# Import from portfolio_exploration_global
from portfolio_exploration_global import (
    stage1_multi_criteria_screening,
    stage2_direct_selection,
    stage3_global_optimization,
    DEFAULT_CONFIG,
    BASE_LOG_DIR,
)

from portimization import load_and_preprocess_data


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


# Import from WFOV (existing Monte Carlo infrastructure)
# Add parent directory to path
wfov_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wfov"))
if wfov_path not in sys.path:
    sys.path.insert(0, wfov_path)

from window_generator import generate_random_windows

# Logging setup
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(BASE_LOG_DIR, f"portfolio_oos_validation_{TIMESTAMP}.log")

logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_sortino_ratio(
    returns, risk_free_rate=0.01, annualize=True, periods_per_year=252
):
    """
    Calculate Sortino ratio (Sharpe using downside deviation only).

    Sortino = (Return - Rf) / Downside Deviation

    Threshold Guidelines:
        < 0: Poor (losing money on risk-adjusted basis)
        0-1: Below average
        1-2: Good
        > 2: Excellent
    """
    if len(returns) < 2:
        return 0.0

    mean_ret = returns.mean()

    # Downside returns (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside, perfect performance

    downside_std = np.sqrt((downside_returns**2).mean())

    if annualize:
        mean_ret *= periods_per_year
        downside_std *= np.sqrt(periods_per_year)

    if downside_std == 0:
        return np.inf

    return (mean_ret - risk_free_rate) / downside_std


def calculate_calmar_ratio(returns, annualize=True, periods_per_year=252):
    """
    Calculate Calmar ratio (Return / Max Drawdown).

    Measures return per unit of maximum drawdown risk.

    Threshold Guidelines:
        < 0.5: Poor
        0.5-1: Average
        1-2: Good
        > 2: Excellent
    """
    if len(returns) < 2:
        return 0.0

    mean_ret = returns.mean()
    if annualize:
        mean_ret *= periods_per_year

    # Calculate max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(drawdown.min())

    if max_dd == 0:
        return np.inf

    return mean_ret / max_dd


def calculate_omega_ratio(returns, threshold=0.0):
    """
    Calculate Omega ratio (probability-weighted gains/losses).

    Omega = Sum(returns > threshold) / Sum(returns < threshold)

    Threshold Guidelines:
        < 1: More losses than gains (poor)
        1-1.5: Average
        1.5-2: Good
        > 2: Excellent
    """
    if len(returns) < 2:
        return 1.0

    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns <= threshold].sum())

    if losses == 0:
        return np.inf

    return gains / losses


def calculate_var(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR) at given confidence level.

    Historical VaR: percentile of returns distribution.

    Returns daily VaR (negative number = potential loss).

    Threshold Guidelines (daily, as % of portfolio):
        > -1%: Low risk
        -1% to -2%: Moderate risk
        -2% to -3%: High risk
        < -3%: Very high risk
    """
    if len(returns) < 10:
        return 0.0

    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns, confidence=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Average loss when VaR is exceeded (tail risk measure).

    Threshold Guidelines (daily, as % of portfolio):
        > -1.5%: Low tail risk
        -1.5% to -3%: Moderate tail risk
        -3% to -5%: High tail risk
        < -5%: Extreme tail risk
    """
    if len(returns) < 10:
        return 0.0

    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_psr(sharpe_ratio, n_observations, skewness=0, kurtosis=3):
    """
    Calculate Probabilistic Sharpe Ratio (PSR).

    Probability that true Sharpe > 0 given observed Sharpe and sample size.
    Uses Bailey & López de Prado (2012) methodology.

    PSR = Phi((SR * sqrt(n-1)) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2))

    Threshold Guidelines:
        < 0.5: Not statistically significant (50% chance of luck)
        0.5-0.85: Weak evidence
        0.85-0.95: Moderate evidence
        > 0.95: Strong evidence (statistically significant)
    """
    from scipy.stats import norm

    if n_observations < 3:
        return 0.5  # Insufficient data

    if sharpe_ratio == 0:
        return 0.5

    # Adjust for non-normality
    sr_std = np.sqrt(
        (1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2)
        / (n_observations - 1)
    )

    if sr_std == 0:
        return 0.5

    psr = norm.cdf(sharpe_ratio / sr_std)
    return psr


def calculate_diversification_ratio(weights, top_n=3):
    """
    Calculate concentration in top N assets.

    Returns percentage of portfolio in top N positions.
    Lower is more diversified.

    Threshold Guidelines:
        < 30%: Well diversified
        30-50%: Moderately concentrated
        50-70%: Concentrated
        > 70%: Highly concentrated (risky)
    """
    if not weights:
        return 0.0

    sorted_weights = sorted(weights.values(), reverse=True)
    top_n_weight = sum(sorted_weights[: min(top_n, len(sorted_weights))])

    return top_n_weight


def calculate_all_metrics(returns, weights, risk_free_rate=0.01, periods_per_year=252):
    """
    Calculate all performance metrics for a returns series.

    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    weights : dict
        Portfolio weights {asset: weight}
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Annualization factor (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    dict with all metrics
    """
    if len(returns) < 5:
        return None

    # Basic metrics
    mean_ret = returns.mean() * periods_per_year
    std_ret = returns.std() * np.sqrt(periods_per_year)
    sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0

    # Max drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Higher moments for PSR
    skewness = returns.skew() if len(returns) > 3 else 0
    kurtosis = (
        returns.kurtosis() + 3 if len(returns) > 4 else 3
    )  # scipy returns excess kurtosis

    return {
        "sharpe": sharpe,
        "annual_return": mean_ret,
        "annual_volatility": std_ret,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "sortino": calculate_sortino_ratio(
            returns, risk_free_rate, periods_per_year=periods_per_year
        ),
        "calmar": calculate_calmar_ratio(returns, periods_per_year=periods_per_year),
        "omega": calculate_omega_ratio(returns),
        "var_95": calculate_var(returns, 0.95),
        "cvar_95": calculate_cvar(returns, 0.95),
        "psr": calculate_psr(sharpe, len(returns), skewness, kurtosis),
        "top3_concentration": calculate_diversification_ratio(weights, top_n=3),
        "n_periods": len(returns),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


# =============================================================================
# METRIC THRESHOLDS AND INTERPRETATION
# =============================================================================

METRIC_THRESHOLDS = {
    "sharpe": {
        "excellent": 1.5,
        "good": 0.8,
        "acceptable": 0.3,
        "description": "Risk-adjusted return (higher better)",
        "unit": "ratio",
    },
    "sortino": {
        "excellent": 2.0,
        "good": 1.0,
        "acceptable": 0.5,
        "description": "Downside risk-adjusted return (higher better)",
        "unit": "ratio",
    },
    "calmar": {
        "excellent": 2.0,
        "good": 1.0,
        "acceptable": 0.5,
        "description": "Return per max drawdown (higher better)",
        "unit": "ratio",
    },
    "omega": {
        "excellent": 2.0,
        "good": 1.5,
        "acceptable": 1.1,
        "description": "Gains vs losses ratio (higher better)",
        "unit": "ratio",
    },
    "var_95": {
        "excellent": -0.01,  # Better than -1%
        "good": -0.02,  # Better than -2%
        "acceptable": -0.03,  # Better than -3%
        "description": "Daily VaR at 95% (closer to 0 better)",
        "unit": "daily %",
    },
    "cvar_95": {
        "excellent": -0.015,  # Better than -1.5%
        "good": -0.03,  # Better than -3%
        "acceptable": -0.05,  # Better than -5%
        "description": "Expected shortfall (closer to 0 better)",
        "unit": "daily %",
    },
    "max_drawdown": {
        "excellent": -0.10,  # Better than -10%
        "good": -0.20,  # Better than -20%
        "acceptable": -0.30,  # Better than -30%
        "description": "Maximum peak-to-trough decline (closer to 0 better)",
        "unit": "%",
    },
    "psr": {
        "excellent": 0.95,
        "good": 0.85,
        "acceptable": 0.70,
        "description": "Probability Sharpe > 0 (higher better)",
        "unit": "probability",
    },
    "top3_concentration": {
        "excellent": 0.30,  # Less than 30%
        "good": 0.50,  # Less than 50%
        "acceptable": 0.70,  # Less than 70%
        "description": "Concentration in top 3 (lower better)",
        "unit": "%",
    },
    "annual_return": {
        "excellent": 0.20,  # > 20%
        "good": 0.10,  # > 10%
        "acceptable": 0.05,  # > 5%
        "description": "Annualized return (higher better)",
        "unit": "%",
    },
    "annual_volatility": {
        "excellent": 0.10,  # < 10%
        "good": 0.20,  # < 20%
        "acceptable": 0.30,  # < 30%
        "description": "Annualized volatility (lower better)",
        "unit": "%",
    },
}


def interpret_metric(metric_name, value, thresholds=METRIC_THRESHOLDS):
    """
    Interpret a metric value against thresholds.

    Returns: (rating, emoji, interpretation)
    """
    if metric_name not in thresholds:
        return ("unknown", "❓", "No threshold defined")

    thresh = thresholds[metric_name]

    # Handle metrics where lower is better
    lower_is_better = metric_name in [
        "var_95",
        "cvar_95",
        "max_drawdown",
        "annual_volatility",
        "top3_concentration",
    ]

    if value is None or np.isnan(value) or np.isinf(value):
        return ("unknown", "❓", "Invalid value")

    if lower_is_better:
        # For these metrics, value should be GREATER (closer to 0) than threshold
        if value >= thresh["excellent"]:
            return ("excellent", "🌟", f"Excellent (<{thresh['excellent']:.1%})")
        elif value >= thresh["good"]:
            return ("good", "✅", f"Good (<{thresh['good']:.1%})")
        elif value >= thresh["acceptable"]:
            return ("acceptable", "ℹ️", f"Acceptable (<{thresh['acceptable']:.1%})")
        else:
            return ("poor", "⚠️", f"Poor (>{thresh['acceptable']:.1%})")
    else:
        # For these metrics, value should be GREATER than threshold
        if value >= thresh["excellent"]:
            return ("excellent", "🌟", f"Excellent (>{thresh['excellent']:.2f})")
        elif value >= thresh["good"]:
            return ("good", "✅", f"Good (>{thresh['good']:.2f})")
        elif value >= thresh["acceptable"]:
            return ("acceptable", "ℹ️", f"Acceptable (>{thresh['acceptable']:.2f})")
        else:
            return ("poor", "⚠️", f"Poor (<{thresh['acceptable']:.2f})")


def calculate_degradation(is_value, oos_value, lower_is_better=False):
    """
    Calculate performance degradation from IS to OOS.

    Returns degradation percentage (positive = worse OOS performance).
    """
    if is_value is None or oos_value is None:
        return None
    if np.isnan(is_value) or np.isnan(oos_value):
        return None
    if np.isinf(is_value) or np.isinf(oos_value):
        return None

    if lower_is_better:
        # For var, cvar, max_dd, volatility: OOS worse if more negative
        if is_value == 0:
            return 0
        return (
            (oos_value - is_value) / abs(is_value)
        ) * 100  # More negative = positive degradation
    else:
        # For sharpe, sortino, etc.: OOS worse if lower
        if is_value == 0:
            return 0 if oos_value >= 0 else 100
        return ((is_value - oos_value) / abs(is_value)) * 100


def calculate_oos_metrics(
    portfolio_weights, test_returns, risk_free_rate=0.01, periods_per_year=252
):
    """
    Calculate comprehensive out-of-sample performance metrics.

    Parameters:
    -----------
    portfolio_weights : dict
        {asset: weight} from in-sample optimization
    test_returns : pd.DataFrame
        Log returns for test period (unseen data)
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Annualization factor (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    dict with all OOS metrics or None if failed
    """
    # Build weight array aligned with test_returns columns
    weight_array = np.zeros(len(test_returns.columns))
    assets_found = 0

    for i, asset in enumerate(test_returns.columns):
        if asset in portfolio_weights:
            weight_array[i] = portfolio_weights[asset]
            assets_found += 1

    # Normalize weights (handle missing assets in test period)
    if weight_array.sum() > 0:
        weight_array = weight_array / weight_array.sum()
    else:
        return None  # No overlapping assets

    # Portfolio returns time series
    portfolio_returns = (test_returns * weight_array).sum(axis=1)

    if len(portfolio_returns) < 5:
        return None

    # Basic metrics
    mean_ret = portfolio_returns.mean() * periods_per_year
    std_ret = portfolio_returns.std() * np.sqrt(periods_per_year)
    sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0

    # Cumulative returns
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

    # Higher moments for PSR
    skewness = portfolio_returns.skew() if len(portfolio_returns) > 3 else 0
    kurtosis = portfolio_returns.kurtosis() + 3 if len(portfolio_returns) > 4 else 3

    # Calculate all new metrics
    sortino = calculate_sortino_ratio(
        portfolio_returns, risk_free_rate, periods_per_year=periods_per_year
    )
    calmar = calculate_calmar_ratio(
        portfolio_returns, periods_per_year=periods_per_year
    )
    omega = calculate_omega_ratio(portfolio_returns)
    var_95 = calculate_var(portfolio_returns, 0.95)
    cvar_95 = calculate_cvar(portfolio_returns, 0.95)
    psr = calculate_psr(sharpe, len(portfolio_returns), skewness, kurtosis)
    top3_concentration = calculate_diversification_ratio(portfolio_weights, top_n=3)

    return {
        # Core metrics
        "sharpe": sharpe,
        "annual_return": mean_ret,
        "annual_volatility": std_ret,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        # New comprehensive metrics
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "psr": psr,
        "top3_concentration": top3_concentration,
        # Metadata
        "n_periods": len(test_returns),
        "n_assets_trained": len(portfolio_weights),
        "n_assets_available": assets_found,
        "coverage": assets_found / len(portfolio_weights)
        if len(portfolio_weights) > 0
        else 0,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def calculate_is_metrics(
    portfolio_weights, train_returns, risk_free_rate=0.01, periods_per_year=252
):
    """
    Calculate comprehensive in-sample performance metrics.

    Same as calculate_oos_metrics but for training data.

    Parameters:
    -----------
    portfolio_weights : dict
        {asset: weight} from optimization
    train_returns : pd.DataFrame
        Log returns for training period
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Annualization factor (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    dict with all IS metrics or None if failed
    """
    return calculate_oos_metrics(
        portfolio_weights, train_returns, risk_free_rate, periods_per_year
    )


def monte_carlo_validation(
    csv_path,
    start_date,
    end_date,
    portfolio_choice="max_sharpe",
    train_years=2,
    test_years=1,
    iterations=100,
    embargo_pct=0.02,
    master_seed=42,
    interval="1d",
    config=None,
):
    """
    Monte Carlo validation with random window sampling.

    HIGH STATISTICAL POWER mode - generates many random train/test splits.

    Key Differences from Sequential:
    - Sequential: 2-3 windows (low stat power)
    - Monte Carlo: 50-200 windows (high stat power)
    - Sequential: No overlap (independent)
    - Monte Carlo: Some overlap (correlated, but acceptable)

    Parameters:
    -----------
    csv_path : str
        Path to CSV
    start_date, end_date : str
        Overall date range
    portfolio_choice : str
        'max_sharpe', 'hrp', 'min_volatility', or 'both'
    train_years : int
        Training window size
    test_years : int
        Test window size
    iterations : int
        Number of random samples (50-200 recommended)
    embargo_pct : float
        Gap between train and test (default: 0.02 = 2%, López de Prado methodology)
    master_seed : int
        Random seed for reproducibility
    interval : str
        Data interval ('1d', '1wk', '1mo')
    config : dict
        Portfolio config

    Returns:
    --------
    list of result dicts
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Get interval-specific settings
    periods_per_year = get_periods_per_year(interval)
    min_periods = get_min_periods(interval)

    # Pass periods_per_year to config for annualization in stage functions
    config["periods_per_year"] = periods_per_year

    # Adjust min_trading_days for interval (default 500 is for daily data)
    # For weekly: ~100 periods = ~2 years, for monthly: ~24 periods = ~2 years
    if "min_trading_days" not in config or config["min_trading_days"] == 500:
        if interval == "1wk":
            config["min_trading_days"] = 100  # ~2 years of weekly data
        elif interval == "1mo":
            config["min_trading_days"] = 24  # ~2 years of monthly data

    # IMPORTANT: Lower stage1_min_assets for Monte Carlo (short windows have fewer quality assets)
    config["stage1_min_assets"] = max(20, config["stage2_target_n"] // 2)

    logging.info("\n" + "=" * 80)
    logging.info(" MONTE CARLO OUT-OF-SAMPLE VALIDATION (High Statistical Power)")
    logging.info("=" * 80)
    logging.info(f"\nConfiguration:")
    logging.info(f"  Portfolio:     {portfolio_choice}")
    logging.info(f"  Mode:          Monte Carlo (random stratified windows)")
    logging.info(f"  Interval:      {interval} ({periods_per_year} periods/year)")
    logging.info(f"  Iterations:    {iterations}")
    logging.info(
        f"  Train Window:  {train_years} years (~{train_years * periods_per_year} periods)"
    )
    logging.info(
        f"  Test Window:   {test_years} years (~{test_years * periods_per_year} periods)"
    )
    logging.info(
        f"  Embargo:       {embargo_pct * 100:.1f}% (~{int(embargo_pct * train_years * periods_per_year)} periods)"
    )
    logging.info(f"  Date Range:    {start_date} to {end_date}")
    logging.info(f"  Random Seed:   {master_seed} (reproducible)")
    min_return_str = (
        f"{config['min_annual_return']:.1%}"
        if config.get("min_annual_return") is not None
        else "disabled"
    )
    max_weight_str = (
        f"{config['max_weight']:.1%}"
        if config.get("max_weight") is not None
        else "no cap"
    )
    logging.info(
        f"  Filters:       min Sharpe {config['min_sharpe']}, min annual return {min_return_str}, max weight {max_weight_str}"
    )

    # Load full dataset
    seed = config.get("seed", master_seed) if config else master_seed
    logging.info(f"\nLoading data... (seed={seed})")
    full_returns = load_and_preprocess_data(csv_path, start_date, end_date, seed=seed)

    if full_returns.empty:
        logging.error("Failed to load data")
        return None

    logging.info(
        f"✅ Loaded {len(full_returns)} days, {len(full_returns.columns)} assets"
    )

    if full_returns.empty:
        logging.error("Failed to load data")
        return None

    logging.info(
        f"✅ Loaded {len(full_returns)} days, {len(full_returns.columns)} assets"
    )
    logging.info(
        f"   Date range: {full_returns.index[0].date()} to {full_returns.index[-1].date()}"
    )

    # Generate random windows using WFOV infrastructure
    min_train_days = int(train_years * periods_per_year)
    max_train_days = int(train_years * periods_per_year * 1.2)  # Allow 20% variation

    logging.info(f"\nGenerating {iterations} random windows (quartile-stratified)...")

    windows = generate_random_windows(
        full_start_date=start_date,
        full_end_date=end_date,
        min_lookback_days=min_train_days,
        max_lookback_days=max_train_days,
        num_iterations=iterations,
        master_seed=master_seed,
        stratified=True,
    )

    logging.info(f"✅ Generated {len(windows)} windows")
    logging.info(f"   Stratification: 25% per quartile (ensures regime coverage)")

    # Calculate embargo
    embargo_days = int(embargo_pct * min_train_days)
    logging.info(f"   Embargo: {embargo_days} days (prevents temporal leakage)")

    # Run validation
    results = []
    failed_windows = 0

    logging.info(f"\nRunning Monte Carlo validation ({iterations} iterations)...")

    for i, window in enumerate(windows, 1):
        train_start_dt = pd.to_datetime(window["start_date"])
        train_end_dt = pd.to_datetime(window["end_date"])

        # Test window with embargo
        embargo_end = train_end_dt + pd.Timedelta(days=embargo_days)
        test_start_dt = embargo_end
        test_end_dt = test_start_dt + pd.DateOffset(years=test_years)

        # Check bounds
        if test_end_dt > full_returns.index[-1]:
            if i <= 3:
                logging.warning(
                    f"  Iteration {i}: Test end {test_end_dt.date()} > data end {full_returns.index[-1].date()}"
                )
            failed_windows += 1
            continue

        # Extract data
        train_returns = full_returns.loc[
            (full_returns.index >= train_start_dt) & (full_returns.index < train_end_dt)
        ].copy()

        test_returns = full_returns.loc[
            (full_returns.index >= test_start_dt) & (full_returns.index < test_end_dt)
        ].copy()

        # Lenient check for Monte Carlo (shorter windows acceptable)
        # For weekly: 150 days/year → ~30 weeks; for monthly: ~5 months
        min_train_required = int(
            train_years * periods_per_year * 0.6
        )  # 60% of expected (very lenient)
        min_test_required = int(
            test_years * periods_per_year * 0.2
        )  # 20% of expected minimum

        if (
            len(train_returns) < min_train_required
            or len(test_returns) < min_test_required
        ):
            if i <= 3:
                logging.warning(
                    f"  Iteration {i}: Insufficient data - train {len(train_returns)}/{min_train_required} days, test {len(test_returns)}/{min_test_required} days"
                )
            failed_windows += 1
            continue

        # Progress
        if i % 10 == 0 or i == 1:
            logging.info(f"  Iteration {i}/{len(windows)}...")

        # Optimize (suppress logs)
        try:
            logging.getLogger().setLevel(logging.WARNING)

            stage1_out = stage1_multi_criteria_screening(train_returns, config)

            # For Monte Carlo: Accept fallback portfolios (equal-weight is valid strategy)
            # Unlike deployment, we're testing strategy robustness, not requiring perfect data

            stage2_out = stage2_direct_selection(stage1_out, config)
            stage3_out = stage3_global_optimization(stage2_out, config)

            logging.getLogger().setLevel(logging.INFO)

        except Exception as e:
            logging.getLogger().setLevel(logging.INFO)
            if i <= 3:  # Debug first 3 failures
                logging.error(f"  Iteration {i}: Optimization failed - {e}")
            failed_windows += 1
            continue

        # Test portfolios
        portfolios_to_test = {}

        if portfolio_choice in ["max_sharpe", "both"]:
            if stage3_out.get("max_sharpe", {}).get("status") == "success":
                portfolios_to_test["max_sharpe"] = stage3_out["max_sharpe"]

        if portfolio_choice in ["hrp", "both"]:
            if stage3_out.get("hrp", {}).get("status") == "success":
                portfolios_to_test["hrp"] = stage3_out["hrp"]

        if portfolio_choice == "min_volatility":
            if stage3_out.get("min_volatility", {}).get("status") == "success":
                portfolios_to_test["min_volatility"] = stage3_out["min_volatility"]

        # Calculate IS and OOS metrics
        for portfolio_name, portfolio_data in portfolios_to_test.items():
            weights = portfolio_data["weights"]

            # Calculate comprehensive IS metrics
            is_metrics = calculate_is_metrics(
                weights, train_returns, config["risk_free_rate"], periods_per_year
            )
            if is_metrics is None:
                continue

            # Calculate comprehensive OOS metrics
            oos_metrics = calculate_oos_metrics(
                weights, test_returns, config["risk_free_rate"], periods_per_year
            )
            if oos_metrics is None:
                continue

            # Calculate degradation for all metrics
            deg_sharpe = calculate_degradation(
                is_metrics["sharpe"], oos_metrics["sharpe"]
            )
            deg_return = calculate_degradation(
                is_metrics["annual_return"], oos_metrics["annual_return"]
            )
            deg_sortino = calculate_degradation(
                is_metrics["sortino"], oos_metrics["sortino"]
            )
            deg_calmar = calculate_degradation(
                is_metrics["calmar"], oos_metrics["calmar"]
            )
            deg_omega = calculate_degradation(is_metrics["omega"], oos_metrics["omega"])
            deg_var = calculate_degradation(
                is_metrics["var_95"], oos_metrics["var_95"], lower_is_better=True
            )
            deg_cvar = calculate_degradation(
                is_metrics["cvar_95"], oos_metrics["cvar_95"], lower_is_better=True
            )
            deg_maxdd = calculate_degradation(
                is_metrics["max_drawdown"],
                oos_metrics["max_drawdown"],
                lower_is_better=True,
            )
            deg_vol = calculate_degradation(
                is_metrics["annual_volatility"],
                oos_metrics["annual_volatility"],
                lower_is_better=True,
            )
            deg_psr = calculate_degradation(is_metrics["psr"], oos_metrics["psr"])

            results.append(
                {
                    # Metadata
                    "iteration": i,
                    "mode": "monte_carlo",
                    "portfolio": portfolio_name,
                    "train_start": train_start_dt.strftime("%Y-%m-%d"),
                    "train_end": train_end_dt.strftime("%Y-%m-%d"),
                    "test_start": test_start_dt.strftime("%Y-%m-%d"),
                    "test_end": test_end_dt.strftime("%Y-%m-%d"),
                    "train_days": len(train_returns),
                    "test_days": len(test_returns),
                    "embargo_days": embargo_days,
                    "quartile": window.get("quartile", 0),
                    "asset_coverage": oos_metrics["coverage"],
                    # IS metrics
                    "is_sharpe": is_metrics["sharpe"],
                    "is_return": is_metrics["annual_return"],
                    "is_vol": is_metrics["annual_volatility"],
                    "is_sortino": is_metrics["sortino"],
                    "is_calmar": is_metrics["calmar"],
                    "is_omega": is_metrics["omega"],
                    "is_var_95": is_metrics["var_95"],
                    "is_cvar_95": is_metrics["cvar_95"],
                    "is_max_dd": is_metrics["max_drawdown"],
                    "is_psr": is_metrics["psr"],
                    "is_top3_conc": is_metrics["top3_concentration"],
                    # OOS metrics
                    "oos_sharpe": oos_metrics["sharpe"],
                    "oos_return": oos_metrics["annual_return"],
                    "oos_vol": oos_metrics["annual_volatility"],
                    "oos_sortino": oos_metrics["sortino"],
                    "oos_calmar": oos_metrics["calmar"],
                    "oos_omega": oos_metrics["omega"],
                    "oos_var_95": oos_metrics["var_95"],
                    "oos_cvar_95": oos_metrics["cvar_95"],
                    "oos_max_dd": oos_metrics["max_drawdown"],
                    "oos_psr": oos_metrics["psr"],
                    "oos_top3_conc": oos_metrics["top3_concentration"],
                    # Degradation metrics (%)
                    "deg_sharpe": deg_sharpe,
                    "deg_return": deg_return,
                    "deg_sortino": deg_sortino,
                    "deg_calmar": deg_calmar,
                    "deg_omega": deg_omega,
                    "deg_var": deg_var,
                    "deg_cvar": deg_cvar,
                    "deg_maxdd": deg_maxdd,
                    "deg_vol": deg_vol,
                    "deg_psr": deg_psr,
                }
            )

    logging.info(
        f"\n✅ Monte Carlo complete: {len(results)} successful / {iterations} attempted ({failed_windows} failed)"
    )

    if len(results) < iterations * 0.5:
        logging.warning(
            f"   ⚠️  High failure rate ({failed_windows}/{iterations}) - check data quality or relax filters"
        )

    return results


def walk_forward_validation(
    csv_path,
    start_date,
    end_date,
    portfolio_choice="max_sharpe",
    train_years=2,
    test_months=None,
    test_years=None,
    mode="expanding",
    iterations=100,
    embargo_pct=0.02,
    embargo_days=5,
    master_seed=42,
    interval="1d",
    config=None,
):
    """
    Portfolio validation with multiple modes.

    Modes:
    ------
    1. 'expanding': Sequential expanding window (low stat power, deployment-realistic)
    2. 'rolling': Sequential rolling window with test_months stepping (regime detection)
    3. 'monte_carlo': Random window sampling (HIGH stat power, strategy validation)

    Rolling Mode Details:
    ---------------------
    - Windows step forward by test_months (not train_years)
    - Embargo gap added between train end and test start
    - Example (train_years=3, test_months=3, embargo_days=5):
        Window 1: Train 2020-01-01 to 2023-01-01, Test 2023-01-06 to 2023-04-06
        Window 2: Train 2020-04-01 to 2023-04-01, Test 2023-04-06 to 2023-07-06
        Window 3: Train 2020-07-01 to 2023-07-01, Test 2023-07-06 to 2023-10-06

    Parameters:
    -----------
    csv_path : str
        Path to CSV
    start_date, end_date : str
        Date range
    portfolio_choice : str
        'max_sharpe', 'hrp', 'min_volatility', or 'both'
    train_years : int
        Training window in years
    test_months : int or None
        Test window in months (for sequential modes)
    test_years : int or None
        Test window in years (for Monte Carlo mode)
    mode : str
        'expanding', 'rolling', or 'monte_carlo'
    iterations : int
        Number of iterations for Monte Carlo mode
    embargo_pct : float
        Embargo percentage (Monte Carlo only, default: 0.02)
    embargo_days : int
        Embargo days between train/test for sequential modes (default: 5)
    master_seed : int
        Random seed (Monte Carlo only)
    interval : str
        Data interval ('1d', '1wk', '1mo')
    config : dict
        Config

    Returns:
    --------
    list of result dicts
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Get interval-specific settings
    periods_per_year = get_periods_per_year(interval)
    min_periods = get_min_periods(interval)

    # Pass periods_per_year to config for annualization in stage functions
    config["periods_per_year"] = periods_per_year

    # Adjust min_trading_days for interval (default 500 is for daily data)
    if "min_trading_days" not in config or config["min_trading_days"] == 500:
        if interval == "1wk":
            config["min_trading_days"] = 100  # ~2 years of weekly data
        elif interval == "1mo":
            config["min_trading_days"] = 24  # ~2 years of monthly data

    # Dispatch to Monte Carlo if requested
    if mode == "monte_carlo":
        test_years_mc = test_years if test_years else 1
        return monte_carlo_validation(
            csv_path,
            start_date,
            end_date,
            portfolio_choice=portfolio_choice,
            train_years=train_years,
            test_years=test_years_mc,
            iterations=iterations,
            embargo_pct=embargo_pct,
            master_seed=master_seed,
            interval=interval,
            config=config,
        )

    # Sequential modes (expanding or rolling)
    if test_months is None:
        test_months = 12  # Default: 1 year

    logging.info("\n" + "=" * 80)
    logging.info(" WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    logging.info("=" * 80)
    logging.info(f"\nConfiguration:")
    logging.info(f"  Portfolio:     {portfolio_choice}")
    logging.info(f"  Mode:          {mode}")
    logging.info(f"  Interval:      {interval} ({periods_per_year} periods/year)")
    logging.info(f"  Train Window:  {train_years} years")
    logging.info(f"  Test Window:   {test_months} months")
    logging.info(f"  Embargo Gap:   {embargo_days} days (between train/test)")
    if mode == "rolling":
        logging.info(f"  Step Size:     {test_months} months (rolling by test window)")
    logging.info(f"  Date Range:    {start_date} to {end_date}")
    min_return_str = (
        f"{config['min_annual_return']:.1%}"
        if config.get("min_annual_return") is not None
        else "disabled"
    )
    max_weight_str = (
        f"{config['max_weight']:.1%}"
        if config.get("max_weight") is not None
        else "no cap"
    )
    logging.info(
        f"  Stage 1:       Top {config['stage1_top_n']} (min Sharpe {config['min_sharpe']}, min annual return {min_return_str})"
    )
    logging.info(f"  Stage 2:       Top {config['stage2_target_n']}")
    logging.info(f"  Stage 3:       Max weight {max_weight_str}")

    # Load full dataset
    seed = config.get("seed", master_seed) if config else master_seed
    logging.info(f"\nLoading full dataset... (seed={seed})")
    full_returns = load_and_preprocess_data(csv_path, start_date, end_date, seed=seed)

    if full_returns.empty:
        logging.error("Failed to load data")
        return None

    logging.info(
        f"✅ Loaded {len(full_returns)} days, {len(full_returns.columns)} assets"
    )
    logging.info(
        f"   Date range: {full_returns.index[0].date()} to {full_returns.index[-1].date()}"
    )

    # Generate walk-forward windows
    results = []
    train_start = pd.to_datetime(start_date)
    overall_end = pd.to_datetime(end_date)

    window_num = 1

    while True:
        # Calculate window dates
        if mode == "expanding":
            # Train window grows from start_date
            train_window_start = train_start  # Fixed at overall start
            train_window_end = (
                train_window_start
                + relativedelta(years=train_years)
                + relativedelta(months=test_months * (window_num - 1))
            )
        else:  # rolling
            # Train window slides forward by test_months each iteration
            train_window_start = train_start + relativedelta(
                months=test_months * (window_num - 1)
            )
            train_window_end = train_window_start + relativedelta(years=train_years)

        # Add embargo gap between train and test
        test_window_start = train_window_end + pd.Timedelta(days=embargo_days)
        test_window_end = test_window_start + relativedelta(months=test_months)

        # Check if we have enough data for this window
        if test_window_end > overall_end:
            break

        logging.info(f"\n{'=' * 80}")
        logging.info(f"WINDOW {window_num} ({mode.upper()})")
        logging.info(f"{'=' * 80}")
        logging.info(
            f"  Train:   {train_window_start.date()} to {train_window_end.date()} ({(train_window_end - train_window_start).days} days)"
        )
        logging.info(f"  Embargo: {embargo_days} days gap")
        logging.info(
            f"  Test:    {test_window_start.date()} to {test_window_end.date()} ({test_months} months)"
        )

        # Split data
        train_returns = full_returns.loc[
            (full_returns.index >= train_window_start)
            & (full_returns.index < train_window_end)
        ].copy()

        test_returns = full_returns.loc[
            (full_returns.index >= test_window_start)
            & (full_returns.index < test_window_end)
        ].copy()

        logging.info(
            f"  Train: {len(train_returns)} days, {len(train_returns.columns)} assets"
        )
        logging.info(
            f"  Test:  {len(test_returns)} days, {len(test_returns.columns)} assets"
        )

        # Validation checks (interval-aware)
        if len(train_returns) < min_periods["min_train_periods"]:
            logging.warning(
                f"  ⚠️  Insufficient training data ({len(train_returns)} periods, need {min_periods['min_train_periods']}), skipping"
            )
            break

        if len(test_returns) < min_periods["min_test_periods"]:
            logging.warning(
                f"  ⚠️  Insufficient test data ({len(test_returns)} periods, need {min_periods['min_test_periods']}), skipping"
            )
            break

        # Run portfolio optimization on TRAIN data only
        logging.info(f"\n  Optimizing on in-sample data...")

        try:
            # Suppress detailed logging during optimization
            logging.getLogger().setLevel(logging.WARNING)

            stage1_out = stage1_multi_criteria_screening(train_returns, config)
            stage2_out = stage2_direct_selection(stage1_out, config)
            stage3_out = stage3_global_optimization(stage2_out, config)

            # Restore logging
            logging.getLogger().setLevel(logging.INFO)

            logging.info(f"  ✅ Optimization complete")

        except Exception as e:
            logging.getLogger().setLevel(logging.INFO)
            logging.error(f"  ❌ Optimization failed: {e}")
            break

        # Determine which portfolios to test
        portfolios_to_test = {}

        if portfolio_choice in ["max_sharpe", "both"]:
            if stage3_out.get("max_sharpe", {}).get("status") == "success":
                portfolios_to_test["max_sharpe"] = stage3_out["max_sharpe"]

        if portfolio_choice in ["hrp", "both"]:
            if stage3_out.get("hrp", {}).get("status") == "success":
                portfolios_to_test["hrp"] = stage3_out["hrp"]

        if portfolio_choice == "min_volatility":
            if stage3_out.get("min_volatility", {}).get("status") == "success":
                portfolios_to_test["min_volatility"] = stage3_out["min_volatility"]

        # Test each portfolio on OOS data
        for portfolio_name, portfolio_data in portfolios_to_test.items():
            weights = portfolio_data["weights"]
            is_positions = portfolio_data["n_positions"]

            # Calculate comprehensive IS metrics
            is_metrics = calculate_is_metrics(
                weights, train_returns, config["risk_free_rate"], periods_per_year
            )
            if is_metrics is None:
                logging.warning(
                    f"  ⚠️  {portfolio_name}: Failed to calculate IS metrics, skipping"
                )
                continue

            # Calculate comprehensive OOS metrics
            oos_metrics = calculate_oos_metrics(
                weights, test_returns, config["risk_free_rate"], periods_per_year
            )
            if oos_metrics is None:
                logging.warning(
                    f"  ⚠️  {portfolio_name}: No overlapping assets, skipping"
                )
                continue

            # Calculate degradation for all metrics
            deg_sharpe = calculate_degradation(
                is_metrics["sharpe"], oos_metrics["sharpe"]
            )
            deg_return = calculate_degradation(
                is_metrics["annual_return"], oos_metrics["annual_return"]
            )
            deg_sortino = calculate_degradation(
                is_metrics["sortino"], oos_metrics["sortino"]
            )
            deg_calmar = calculate_degradation(
                is_metrics["calmar"], oos_metrics["calmar"]
            )
            deg_omega = calculate_degradation(is_metrics["omega"], oos_metrics["omega"])
            deg_var = calculate_degradation(
                is_metrics["var_95"], oos_metrics["var_95"], lower_is_better=True
            )
            deg_cvar = calculate_degradation(
                is_metrics["cvar_95"], oos_metrics["cvar_95"], lower_is_better=True
            )
            deg_maxdd = calculate_degradation(
                is_metrics["max_drawdown"],
                oos_metrics["max_drawdown"],
                lower_is_better=True,
            )
            deg_vol = calculate_degradation(
                is_metrics["annual_volatility"],
                oos_metrics["annual_volatility"],
                lower_is_better=True,
            )
            deg_psr = calculate_degradation(is_metrics["psr"], oos_metrics["psr"])

            # Log summary
            logging.info(f"\n  {portfolio_name.upper()} (Positions: {is_positions}):")
            logging.info(f"    {'Metric':<12} {'IS':>10} {'OOS':>10} {'Degrad%':>10}")
            logging.info(f"    {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")
            logging.info(
                f"    {'Sharpe':<12} {is_metrics['sharpe']:>10.3f} {oos_metrics['sharpe']:>10.3f} {deg_sharpe:>+10.1f}%"
            )
            logging.info(
                f"    {'Return':<12} {is_metrics['annual_return']:>10.1%} {oos_metrics['annual_return']:>10.1%} {deg_return:>+10.1f}%"
            )
            logging.info(
                f"    {'Volatility':<12} {is_metrics['annual_volatility']:>10.1%} {oos_metrics['annual_volatility']:>10.1%} {deg_vol:>+10.1f}%"
            )
            logging.info(
                f"    {'Sortino':<12} {is_metrics['sortino']:>10.3f} {oos_metrics['sortino']:>10.3f} {deg_sortino:>+10.1f}%"
            )
            logging.info(
                f"    {'Calmar':<12} {is_metrics['calmar']:>10.3f} {oos_metrics['calmar']:>10.3f} {deg_calmar:>+10.1f}%"
            )
            logging.info(
                f"    {'Omega':<12} {is_metrics['omega']:>10.3f} {oos_metrics['omega']:>10.3f} {deg_omega:>+10.1f}%"
            )
            logging.info(
                f"    {'VaR 95%':<12} {is_metrics['var_95']:>10.2%} {oos_metrics['var_95']:>10.2%} {deg_var:>+10.1f}%"
            )
            logging.info(
                f"    {'CVaR 95%':<12} {is_metrics['cvar_95']:>10.2%} {oos_metrics['cvar_95']:>10.2%} {deg_cvar:>+10.1f}%"
            )
            logging.info(
                f"    {'Max DD':<12} {is_metrics['max_drawdown']:>10.1%} {oos_metrics['max_drawdown']:>10.1%} {deg_maxdd:>+10.1f}%"
            )
            logging.info(
                f"    {'PSR':<12} {is_metrics['psr']:>10.3f} {oos_metrics['psr']:>10.3f} {deg_psr:>+10.1f}%"
            )
            logging.info(
                f"    {'Top3 Conc.':<12} {is_metrics['top3_concentration']:>10.1%} {oos_metrics['top3_concentration']:>10.1%} {'N/A':>10}"
            )
            logging.info(
                f"    Coverage: {oos_metrics['n_assets_available']}/{oos_metrics['n_assets_trained']} assets ({oos_metrics['coverage']:.1%})"
            )

            # Verdict based on average degradation of key metrics
            avg_degradation = np.mean(
                [
                    d
                    for d in [deg_sharpe, deg_sortino, deg_calmar]
                    if d is not None and not np.isnan(d)
                ]
            )
            if avg_degradation > 30:
                logging.error(
                    f"    Verdict: 🚨 SEVERE overfitting (>30% avg degradation)"
                )
            elif avg_degradation > 20:
                logging.warning(
                    f"    Verdict: ⚠️  SIGNIFICANT overfitting (>20% avg degradation)"
                )
            elif avg_degradation > 10:
                logging.info(f"    Verdict: ℹ️  MODERATE degradation (10-20% avg)")
            elif avg_degradation > 0:
                logging.info(f"    Verdict: ✅ ACCEPTABLE (<10% avg degradation)")
            else:
                logging.info(f"    Verdict: ✅ EXCELLENT (OOS ≥ IS on average)")

            # Store results
            results.append(
                {
                    # Metadata
                    "window": window_num,
                    "mode": mode,
                    "portfolio": portfolio_name,
                    "train_start": train_window_start.strftime("%Y-%m-%d"),
                    "train_end": train_window_end.strftime("%Y-%m-%d"),
                    "test_start": test_window_start.strftime("%Y-%m-%d"),
                    "test_end": test_window_end.strftime("%Y-%m-%d"),
                    "train_days": len(train_returns),
                    "test_days": len(test_returns),
                    "embargo_days": embargo_days,
                    "n_positions": is_positions,
                    "asset_coverage": oos_metrics["coverage"],
                    # IS metrics
                    "is_sharpe": is_metrics["sharpe"],
                    "is_return": is_metrics["annual_return"],
                    "is_vol": is_metrics["annual_volatility"],
                    "is_sortino": is_metrics["sortino"],
                    "is_calmar": is_metrics["calmar"],
                    "is_omega": is_metrics["omega"],
                    "is_var_95": is_metrics["var_95"],
                    "is_cvar_95": is_metrics["cvar_95"],
                    "is_max_dd": is_metrics["max_drawdown"],
                    "is_psr": is_metrics["psr"],
                    "is_top3_conc": is_metrics["top3_concentration"],
                    # OOS metrics
                    "oos_sharpe": oos_metrics["sharpe"],
                    "oos_return": oos_metrics["annual_return"],
                    "oos_vol": oos_metrics["annual_volatility"],
                    "oos_sortino": oos_metrics["sortino"],
                    "oos_calmar": oos_metrics["calmar"],
                    "oos_omega": oos_metrics["omega"],
                    "oos_var_95": oos_metrics["var_95"],
                    "oos_cvar_95": oos_metrics["cvar_95"],
                    "oos_max_dd": oos_metrics["max_drawdown"],
                    "oos_psr": oos_metrics["psr"],
                    "oos_top3_conc": oos_metrics["top3_concentration"],
                    "oos_win_rate": oos_metrics["win_rate"],
                    # Degradation metrics (%)
                    "deg_sharpe": deg_sharpe,
                    "deg_return": deg_return,
                    "deg_vol": deg_vol,
                    "deg_sortino": deg_sortino,
                    "deg_calmar": deg_calmar,
                    "deg_omega": deg_omega,
                    "deg_var": deg_var,
                    "deg_cvar": deg_cvar,
                    "deg_maxdd": deg_maxdd,
                    "deg_psr": deg_psr,
                }
            )

        # Move to next window (window calculations handled by window_num at loop start)
        window_num += 1

    return results


def summarize_validation_results(results):
    """
    Aggregate and summarize walk-forward validation results with comprehensive metrics.

    Provides insights based on:
    - Multiple performance metrics (Sharpe, Sortino, Calmar, Omega)
    - Risk metrics (VaR, CVaR, Max Drawdown, Volatility)
    - Statistical significance (PSR, t-tests, confidence intervals)
    - Diversification analysis (Top 3 concentration)
    """
    if not results:
        logging.error("\n❌ No validation results to summarize")
        return

    df = pd.DataFrame(results)

    logging.info(f"\n{'=' * 100}")
    logging.info(" COMPREHENSIVE OUT-OF-SAMPLE VALIDATION SUMMARY")
    logging.info(f"{'=' * 100}")

    # Summary by portfolio
    portfolios = df["portfolio"].unique()

    # Define metrics to analyze
    metrics_config = [
        # (column_prefix, display_name, is_lower_better, is_pct)
        ("sharpe", "Sharpe Ratio", False, False),
        ("return", "Annual Return", False, True),
        ("vol", "Volatility", True, True),
        ("sortino", "Sortino Ratio", False, False),
        ("calmar", "Calmar Ratio", False, False),
        ("omega", "Omega Ratio", False, False),
        ("var_95", "VaR 95%", True, True),
        ("cvar_95", "CVaR 95%", True, True),
        ("max_dd", "Max Drawdown", True, True),
        ("psr", "PSR", False, False),
        ("top3_conc", "Top3 Concentration", True, True),
    ]

    summary_data = []
    all_insights = []

    for portfolio_name in portfolios:
        portfolio_df = df[df["portfolio"] == portfolio_name]
        n_samples = len(portfolio_df)

        logging.info(f"\n{'=' * 100}")
        logging.info(f" {portfolio_name.upper()} - DETAILED METRICS ANALYSIS")
        logging.info(f"{'=' * 100}")
        logging.info(f"\n  Windows Tested: {n_samples}")
        logging.info(
            f"  Avg Asset Coverage: {portfolio_df['asset_coverage'].mean():.1%}"
        )

        # Create comprehensive metrics table
        logging.info(
            f"\n  {'Metric':<18} {'Avg IS':>12} {'Avg OOS':>12} {'Std OOS':>10} {'Avg Deg%':>10} {'Rating':>12}"
        )
        logging.info(
            f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 12}"
        )

        portfolio_insights = []
        metric_ratings = {}

        for metric_key, metric_name, lower_is_better, is_pct in metrics_config:
            is_col = f"is_{metric_key}"
            oos_col = f"oos_{metric_key}"
            deg_col = f"deg_{metric_key}" if metric_key != "top3_conc" else None

            # Skip if columns don't exist
            if (
                is_col not in portfolio_df.columns
                or oos_col not in portfolio_df.columns
            ):
                continue

            avg_is = portfolio_df[is_col].mean()
            avg_oos = portfolio_df[oos_col].mean()
            std_oos = portfolio_df[oos_col].std()
            avg_deg = (
                portfolio_df[deg_col].mean()
                if deg_col and deg_col in portfolio_df.columns
                else 0
            )

            # Get rating
            rating, emoji, _ = interpret_metric(metric_key, avg_oos)
            metric_ratings[metric_key] = rating

            # Format values
            if is_pct:
                is_str = f"{avg_is:>11.1%}"
                oos_str = f"{avg_oos:>11.1%}"
                std_str = f"{std_oos:>9.1%}"
            else:
                is_str = f"{avg_is:>11.3f}"
                oos_str = f"{avg_oos:>11.3f}"
                std_str = f"{std_oos:>9.3f}"

            deg_str = (
                f"{avg_deg:>+9.1f}%"
                if avg_deg is not None and not np.isnan(avg_deg)
                else f"{'N/A':>10}"
            )

            logging.info(
                f"  {metric_name:<18} {is_str:>12} {oos_str:>12} {std_str:>10} {deg_str:>10} {emoji + ' ' + rating:<12}"
            )

            # Generate insights for this metric
            if avg_deg is not None and not np.isnan(avg_deg):
                if abs(avg_deg) > 50:
                    insight = f"🚨 {metric_name}: High degradation ({avg_deg:+.1f}%) - potential overfitting"
                    portfolio_insights.append(insight)
                elif avg_deg < -20:
                    insight = f"✅ {metric_name}: OOS outperforms IS ({avg_deg:+.1f}%) - conservative in-sample"
                    portfolio_insights.append(insight)

        # Statistical significance analysis
        logging.info(f"\n  STATISTICAL SIGNIFICANCE ANALYSIS:")
        logging.info(f"  {'-' * 60}")

        if n_samples >= 5:
            # T-test for key metrics
            for metric_key, metric_name in [
                ("sharpe", "Sharpe"),
                ("sortino", "Sortino"),
                ("return", "Return"),
            ]:
                oos_col = f"oos_{metric_key}"
                if oos_col not in portfolio_df.columns:
                    continue

                oos_values = portfolio_df[oos_col].dropna().values
                if len(oos_values) < 3:
                    continue

                t_stat, p_value = stats.ttest_1samp(oos_values, 0)

                # 95% CI
                try:
                    ci_lower, ci_upper = stats.t.interval(
                        0.95,
                        len(oos_values) - 1,
                        loc=np.mean(oos_values),
                        scale=stats.sem(oos_values),
                    )
                except:
                    ci_lower, ci_upper = np.nan, np.nan

                sig_emoji = "✅" if p_value < 0.05 else "⚠️"
                logging.info(
                    f"    {metric_name:<12} t={t_stat:>6.2f}  p={p_value:>6.4f} {sig_emoji}  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                )

                if p_value >= 0.05:
                    portfolio_insights.append(
                        f"⚠️ {metric_name} not statistically significant (p={p_value:.3f})"
                    )
                elif ci_lower > 0:
                    portfolio_insights.append(
                        f"✅ {metric_name} statistically positive (95% CI > 0)"
                    )

        # PSR Analysis
        if "oos_psr" in portfolio_df.columns:
            avg_psr = portfolio_df["oos_psr"].mean()
            logging.info(f"\n    Probabilistic Sharpe Ratio (PSR):")
            logging.info(f"      Average OOS PSR: {avg_psr:.3f}")

            if avg_psr >= 0.95:
                logging.info(f"      ✅ Strong evidence (>95% probability Sharpe > 0)")
            elif avg_psr >= 0.85:
                logging.info(f"      ✅ Moderate evidence (>85% probability)")
            elif avg_psr >= 0.70:
                logging.info(f"      ℹ️  Weak evidence (70-85% probability)")
            else:
                logging.warning(f"      ⚠️  Insufficient evidence (<70% probability)")
                portfolio_insights.append(
                    f"⚠️ Low PSR ({avg_psr:.2f}) - strategy may be noise"
                )

        # Risk Analysis
        logging.info(f"\n  RISK ANALYSIS:")
        logging.info(f"  {'-' * 60}")

        if (
            "oos_var_95" in portfolio_df.columns
            and "oos_cvar_95" in portfolio_df.columns
        ):
            avg_var = portfolio_df["oos_var_95"].mean()
            avg_cvar = portfolio_df["oos_cvar_95"].mean()
            avg_maxdd = (
                portfolio_df["oos_max_dd"].mean()
                if "oos_max_dd" in portfolio_df.columns
                else np.nan
            )

            logging.info(
                f"    Average Daily VaR (95%):    {avg_var:>8.2%} (expect to lose this on 1/20 days)"
            )
            logging.info(
                f"    Average Daily CVaR (95%):   {avg_cvar:>8.2%} (average loss when VaR exceeded)"
            )
            if not np.isnan(avg_maxdd):
                logging.info(f"    Average Max Drawdown:       {avg_maxdd:>8.1%}")

            # Risk insights
            if avg_var < -0.03:
                portfolio_insights.append(
                    f"🚨 High daily risk: VaR = {avg_var:.1%} (expect significant daily losses)"
                )
            if avg_cvar < -0.05:
                portfolio_insights.append(
                    f"🚨 High tail risk: CVaR = {avg_cvar:.1%} (extreme losses when bad)"
                )
            if not np.isnan(avg_maxdd) and avg_maxdd < -0.30:
                portfolio_insights.append(
                    f"⚠️ Deep drawdowns: Avg Max DD = {avg_maxdd:.1%}"
                )

        # Concentration Analysis
        if "oos_top3_conc" in portfolio_df.columns:
            avg_conc = portfolio_df["oos_top3_conc"].mean()
            logging.info(f"\n  DIVERSIFICATION ANALYSIS:")
            logging.info(f"  {'-' * 60}")
            logging.info(f"    Average Top 3 Concentration: {avg_conc:.1%}")

            if avg_conc > 0.70:
                logging.warning(f"    ⚠️  Highly concentrated portfolio (>70% in top 3)")
                portfolio_insights.append(
                    f"⚠️ High concentration: {avg_conc:.0%} in top 3 assets"
                )
            elif avg_conc > 0.50:
                logging.info(f"    ℹ️  Moderately concentrated (50-70% in top 3)")
            else:
                logging.info(f"    ✅ Well diversified (<50% in top 3)")

        # Overall verdict
        logging.info(f"\n  OVERALL VERDICT:")
        logging.info(f"  {'-' * 60}")

        # Count ratings
        excellent_count = sum(1 for r in metric_ratings.values() if r == "excellent")
        good_count = sum(1 for r in metric_ratings.values() if r == "good")
        acceptable_count = sum(1 for r in metric_ratings.values() if r == "acceptable")
        poor_count = sum(1 for r in metric_ratings.values() if r == "poor")

        logging.info(
            f"    Ratings: {excellent_count} Excellent, {good_count} Good, {acceptable_count} Acceptable, {poor_count} Poor"
        )

        # Calculate composite score
        avg_deg_sharpe = (
            portfolio_df["deg_sharpe"].mean()
            if "deg_sharpe" in portfolio_df.columns
            else 0
        )
        avg_oos_sharpe = (
            portfolio_df["oos_sharpe"].mean()
            if "oos_sharpe" in portfolio_df.columns
            else 0
        )
        avg_oos_psr = (
            portfolio_df["oos_psr"].mean() if "oos_psr" in portfolio_df.columns else 0.5
        )

        composite_score = (
            avg_oos_sharpe * 0.4
            + (1 - abs(avg_deg_sharpe) / 100) * 0.3
            + avg_oos_psr * 0.3
        )

        if avg_deg_sharpe > 30 or poor_count > 3:
            verdict = "🚨 REJECT"
            reason = "Severe overfitting or multiple poor metrics"
            logging.error(f"    Verdict: {verdict} - {reason}")
        elif avg_deg_sharpe > 20 or poor_count > 1:
            verdict = "⚠️ CAUTION"
            reason = "Significant overfitting risk"
            logging.warning(f"    Verdict: {verdict} - {reason}")
        elif excellent_count >= 3 and poor_count == 0:
            verdict = "🌟 EXCELLENT"
            reason = "Strong OOS performance across metrics"
            logging.info(f"    Verdict: {verdict} - {reason}")
        elif good_count + excellent_count >= 5:
            verdict = "✅ GOOD"
            reason = "Solid OOS performance"
            logging.info(f"    Verdict: {verdict} - {reason}")
        else:
            verdict = "ℹ️ ACCEPTABLE"
            reason = "Moderate OOS performance"
            logging.info(f"    Verdict: {verdict} - {reason}")

        logging.info(f"    Composite Score: {composite_score:.3f}")

        # Show insights
        if portfolio_insights:
            logging.info(f"\n  KEY INSIGHTS:")
            for insight in portfolio_insights[:5]:  # Limit to 5
                logging.info(f"    {insight}")

        all_insights.extend(
            [(portfolio_name, insight) for insight in portfolio_insights]
        )

        # Store summary
        summary_data.append(
            {
                "Portfolio": portfolio_name,
                "Windows": n_samples,
                "OOS Sharpe": f"{avg_oos_sharpe:.3f}",
                "OOS Sortino": f"{portfolio_df['oos_sortino'].mean():.3f}"
                if "oos_sortino" in portfolio_df.columns
                else "N/A",
                "OOS PSR": f"{avg_oos_psr:.2f}",
                "Deg Sharpe%": f"{avg_deg_sharpe:+.1f}",
                "Score": f"{composite_score:.3f}",
                "Verdict": verdict,
            }
        )

    # Comparison table for multiple portfolios
    if len(portfolios) > 1:
        logging.info(f"\n{'=' * 100}")
        logging.info(" PORTFOLIO COMPARISON")
        logging.info(f"{'=' * 100}")

        summary_df = pd.DataFrame(summary_data)
        logging.info(f"\n{summary_df.to_string(index=False)}")

        # Find winners by different criteria
        logging.info(f"\n  WINNERS BY CRITERIA:")

        if "oos_sharpe" in df.columns:
            best_sharpe = df.groupby("portfolio")["oos_sharpe"].mean().idxmax()
            logging.info(f"    Best OOS Sharpe:     {best_sharpe}")

        if "oos_sortino" in df.columns:
            best_sortino = df.groupby("portfolio")["oos_sortino"].mean().idxmax()
            logging.info(f"    Best OOS Sortino:    {best_sortino}")

        if "oos_psr" in df.columns:
            best_psr = df.groupby("portfolio")["oos_psr"].mean().idxmax()
            logging.info(f"    Best OOS PSR:        {best_psr}")

        if "deg_sharpe" in df.columns:
            lowest_deg = df.groupby("portfolio")["deg_sharpe"].mean().abs().idxmin()
            logging.info(f"    Lowest Degradation:  {lowest_deg}")

        if "oos_max_dd" in df.columns:
            best_dd = (
                df.groupby("portfolio")["oos_max_dd"].mean().idxmax()
            )  # Less negative is better
            logging.info(f"    Best Max Drawdown:   {best_dd}")

    # Final recommendations
    logging.info(f"\n{'=' * 100}")
    logging.info(" DEPLOYMENT RECOMMENDATIONS")
    logging.info(f"{'=' * 100}")

    for portfolio_name in portfolios:
        portfolio_df = df[df["portfolio"] == portfolio_name]

        # Get key metrics
        avg_oos_sharpe = (
            portfolio_df["oos_sharpe"].mean()
            if "oos_sharpe" in portfolio_df.columns
            else 0
        )
        avg_oos_psr = (
            portfolio_df["oos_psr"].mean() if "oos_psr" in portfolio_df.columns else 0.5
        )
        avg_deg = (
            portfolio_df["deg_sharpe"].mean()
            if "deg_sharpe" in portfolio_df.columns
            else 0
        )

        logging.info(f"\n  {portfolio_name.upper()}:")

        if avg_oos_sharpe > 0.8 and avg_oos_psr > 0.85 and abs(avg_deg) < 20:
            logging.info(f"    ✅ RECOMMENDED for live deployment")
            logging.info(f"       - Strong risk-adjusted returns (Sharpe > 0.8)")
            logging.info(f"       - Statistically significant (PSR > 85%)")
            logging.info(f"       - Stable IS→OOS transfer (<20% degradation)")
        elif avg_oos_sharpe > 0.3 and avg_oos_psr > 0.70:
            logging.info(f"    ℹ️  ACCEPTABLE for deployment with monitoring")
            logging.info(f"       - Consider paper trading first")
            logging.info(f"       - Monitor for continued degradation")
        else:
            logging.info(f"    ⚠️  NOT RECOMMENDED for deployment")
            logging.info(f"       - Consider refining strategy or increasing data")

    return results


def save_results_csv(results, output_path):
    """Save validation results to CSV."""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logging.info(f"\n📊 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio out-of-sample validation (Sequential & Monte Carlo modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SEQUENTIAL MODE (deployment validation, low stat power)
  # Standard: 2y train, 1y test, expanding
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 --portfolio max_sharpe

  # Rolling window (regime detection)
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 --mode rolling

  # MONTE CARLO MODE (strategy validation, HIGH stat power)
  # 100 random windows, 2y train, 1y test
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 \\
      --mode monte_carlo --iterations 100 --train-years 2 --test-years 1

  # With minimum annual return filter (e.g., 5%% minimum)
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 \\
      --mode monte_carlo --min-annual-return 0.05 --min-sharpe 0.5

  # Compare max_sharpe vs HRP (Monte Carlo)
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 \\
      --mode monte_carlo --portfolio both --iterations 100

  # Conservative (3y train, 200 iterations for strong confidence)
  python %(prog)s --csv data.csv --start 2015-01-01 --end 2026-01-01 \\
      --mode monte_carlo --train-years 3 --iterations 200

  # High growth filter (15%% minimum annual return)
  python %(prog)s --csv data.csv --start 2020-01-01 --end 2026-01-01 \\
      --mode expanding --min-annual-return 0.15
        """,
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV or parquet file with price data",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Read data from the parquet market data store (no CSV needed). "
        "Uses all tickers in data/market_data/ticker_universe.json.",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Required unless --lookback-days is used.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today). Required unless --lookback-days is used.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Lookback period in calendar days from today. Overrides --start and --end. "
        "E.g., --lookback-days 1825 sets start to ~5 years ago and end to today.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1wk", "1mo"],
        help="Data interval (default: 1d). Affects annualization: 1d=252, 1wk=52, 1mo=12",
    )

    parser.add_argument(
        "--portfolio",
        type=str,
        default="max_sharpe",
        choices=["max_sharpe", "hrp", "min_volatility", "both"],
        help='Portfolio to test (default: max_sharpe). "both" = max_sharpe vs hrp',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="expanding",
        choices=["expanding", "rolling", "monte_carlo"],
        help="Validation mode: expanding, rolling, or monte_carlo (default: expanding)",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=2,
        help="Training window in years (default: 2)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=12,
        help="Test window in months for sequential modes (default: 12)",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=None,
        help="Test window in years for Monte Carlo mode (default: 1)",
    )

    # Monte Carlo specific parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for Monte Carlo mode (default: 100)",
    )
    parser.add_argument(
        "--embargo-pct",
        type=float,
        default=0.02,
        help="Embargo percentage for Monte Carlo (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for Monte Carlo (default: 42)"
    )

    # Sequential mode embargo parameter
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="Embargo days between train/test for sequential modes (default: 5)",
    )

    # Portfolio optimization parameters
    parser.add_argument(
        "--min-sharpe", type=float, default=0.5, help="Min Sharpe filter (default: 0.5)"
    )
    parser.add_argument(
        "--min-annual-return",
        type=float,
        default=None,
        help="Min annual return filter (e.g., 0.05 for 5%%, 0.15 for 15%%). Default: None (disabled)",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=500,
        help="Min trading days for data quality (default: 500)",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.95,
        help="Max correlation for duplicate removal (default: 0.95)",
    )
    parser.add_argument(
        "--stage1-top-n", type=int, default=100, help="Stage 1 target (default: 100)"
    )
    parser.add_argument(
        "--stage2-target", type=int, default=40, help="Stage 2 target (default: 40)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.01,
        help="Risk-free rate (default: 0.01)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Max weight per position (e.g., 0.10 for 10%%, 0.15 for 15%%). Default: None (no cap)",
    )
    parser.add_argument(
        "--force-tickers",
        type=str,
        nargs="*",
        default=None,
        help="Tickers to force-include in Stage 3 optimization (bypass Stage 1/2 "
        "filtering). They must pass data quality checks but are exempt from "
        "Sharpe/score/rank thresholds. Space-separated.",
    )

    args = parser.parse_args()

    # Compute dates from --lookback-days if provided
    if args.lookback_days:
        from datetime import datetime, timedelta

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=args.lookback_days)
        args.end = end_dt.strftime("%Y-%m-%d")
        args.start = start_dt.strftime("%Y-%m-%d")
    elif not args.start:
        parser.error("--start is required (or use --lookback-days)")

    if not args.end:
        from datetime import datetime

        args.end = datetime.now().strftime("%Y-%m-%d")

    # Resolve data source: --from-store, --csv, or error
    data_path = args.csv
    if args.from_store:
        import tempfile

        try:
            import sys as _sys
            from pathlib import Path as _Path

            _project_root = str(_Path(__file__).resolve().parent.parent.parent)
            if _project_root not in _sys.path:
                _sys.path.insert(0, _project_root)

            from algos.common.market_data_store import MarketDataStore
            from algos.common.update_market_data import load_ticker_universe

            store = MarketDataStore()
            tickers_map = load_ticker_universe()
            if not tickers_map:
                logging.error("No tickers found in ticker_universe.json")
                return None

            # Export from parquet store to a temp CSV
            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, prefix="oos_data_"
            )
            data_path = tmp.name
            tmp.close()
            export_df = store.export_portfolio_csv(
                tickers_map,
                args.start,
                args.end,
                data_path,
                min_coverage=0.8,
                include_calendar_gaps=False,
            )
            if export_df.empty:
                logging.error(
                    "Parquet store returned no data. "
                    "Run 'python -m algos.common.update_market_data --init' first."
                )
                return None
            logging.info(
                f"Exported {export_df.shape} from parquet store to {data_path}"
            )
        except ImportError:
            logging.error("MarketDataStore not available. Use --csv instead.")
            return None
    elif data_path is None:
        parser.error("Either --csv or --from-store is required.")

    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "min_sharpe": args.min_sharpe,
            "min_annual_return": args.min_annual_return,
            "min_trading_days": args.min_trading_days,
            "max_correlation": args.max_correlation,
            "stage1_top_n": args.stage1_top_n,
            "stage2_target_n": args.stage2_target,
            "risk_free_rate": args.risk_free_rate,
            "max_weight": args.max_weight,
            "force_tickers": args.force_tickers or [],
            "seed": args.seed,
        }
    )

    # Run validation
    results = walk_forward_validation(
        data_path,
        args.start,
        args.end,
        portfolio_choice=args.portfolio,
        train_years=args.train_years,
        test_months=args.test_months,
        test_years=args.test_years,
        mode=args.mode,
        iterations=args.iterations,
        embargo_pct=args.embargo_pct,
        embargo_days=args.embargo_days,
        master_seed=args.seed,
        interval=args.interval,
        config=config,
    )

    if results:
        # Summary
        summarize_validation_results(results)

        # Save to CSV
        csv_output = os.path.join(BASE_LOG_DIR, f"oos_validation_{TIMESTAMP}.csv")
        save_results_csv(results, csv_output)

        logging.info(f"\n{'=' * 80}")
        logging.info(f" VALIDATION COMPLETE")
        logging.info(f"{'=' * 80}")
        logging.info(f" Log:     {LOG_FILE}")
        logging.info(f" Results: {csv_output}")
        logging.info(f"{'=' * 80}\n")

        return results
    else:
        logging.error("\n❌ Validation failed - no results")
        return None


if __name__ == "__main__":
    main()
