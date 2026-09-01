"""
Portfolio Exploration Workflow with Global Optimization

Complete 4-stage pipeline: 300+ assets → 20-30 final portfolio
Reuses battle-tested components from portimization.py.

Workflow:
    Stage 1: Multi-criteria screening (350 → 100)
    Stage 2: Diversification clustering (100 → 40)
    Stage 3: Global optimization (SLSQP/DE/HRP)
    Stage 4: Analysis & reporting

Critical Fixes Applied:
    - Ledoit-Wolf covariance (prevents singularity)
    - Volatility floor (prevents division by zero)
    - Data snapshot freeze (prevents staleness)
    - Dynamic thresholds (adaptive filtering)
    - Fallback portfolios (handles edge cases)

Usage:
    # Fast mode (SLSQP only, ~35 seconds)
    python portfolio_exploration_global.py \\
        --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \\
        --start 2022-12-01 \\
        --end 2025-12-01

    # Global search mode (differential_evolution, ~90 seconds)
    python portfolio_exploration_global.py \\
        --csv ../../data/financial_data_combined_prices_2022-12-01_2025-12-01_1d.csv \\
        --start 2022-12-01 \\
        --end 2025-12-01 \\
        --use-global-search

Author: Algorithmic Trading System
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import logging
import os
import sys
import time
from datetime import datetime
import warnings

# Import battle-tested functions from portimization.py
from portimization import (
    load_and_preprocess_data,
    calculate_leveraged_portfolio,
    statistics,
    BASE_DATA_DIR,
    BASE_IMAGE_DIR,
    BASE_LOG_DIR,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # Suppress sklearn numerical warnings
np.seterr(all="ignore")  # Suppress numpy warnings (divide, overflow, invalid)

# Generate timestamp for outputs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# FIX: Logging conflict with portimization.py
# portimization.py calls basicConfig at import, which runs first
# Solution: Get root logger and add our own handlers
LOG_FILE_PATH = os.path.join(BASE_LOG_DIR, f"portfolio_exploration_{TIMESTAMP}.log")

# Remove any existing handlers to avoid conflicts
logger = logging.getLogger()
logger.handlers.clear()

# Set up our own handlers
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(message)s"))  # Clean format for file

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(message)s")
)  # Clean format for console

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Stage 1: Screening
    "min_trading_days": 500,  # Data quality filter (adjust for interval: 500 daily, ~100 weekly, ~24 monthly)
    "min_sharpe": 0.5,  # Only keep strong performers (changed from -0.5)
    "min_annual_return": None,  # Minimum annual return filter (e.g., 0.05 for 5%, 0.15 for 15%)
    "max_correlation": 0.95,  # Remove duplicates
    "stage1_top_n": 100,  # Target output
    "stage1_min_assets": 50,  # Fallback minimum
    # Stage 2: Direct Selection (simplified - no clustering)
    "stage2_target_n": 40,  # Exact output (deterministic)
    # Stage 3: Optimization
    "risk_free_rate": 0.01,  # 1% annual
    "min_weight_threshold": 0.001,  # Dust filter (0.1%)
    "max_weight": None,  # Max weight per position (e.g., 0.10 for 10%, 0.15 for 15%). None = no cap
    # Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
    "periods_per_year": 252,
    # Removed parameters (deleted features):
    # - use_global_search: Exhaustive search removed (0% improvement over SLSQP)
    # - n_clusters, min_per_cluster, max_per_cluster: Clustering removed (caused non-monotonic behavior)
}


def get_periods_per_year(interval: str = "1d") -> int:
    """Get annualization factor based on data interval."""
    interval_map = {
        "1d": 252,  # Daily trading days
        "1wk": 52,  # Weekly
        "1mo": 12,  # Monthly
        "1h": 252 * 7,  # Hourly (approximate)
    }
    return interval_map.get(interval, 252)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _cap_weights(weights, max_weight, max_iterations=100):
    """
    Cap portfolio weights at max_weight and redistribute excess proportionally.

    This is used for algorithms like HRP that don't use optimization bounds.
    The algorithm iteratively:
    1. Cap weights exceeding max_weight
    2. Redistribute excess to weights below max_weight (proportionally)
    3. Repeat until no weight exceeds max_weight or max iterations reached

    Parameters:
    -----------
    weights : np.ndarray
        Original portfolio weights (must sum to 1.0)
    max_weight : float
        Maximum allowed weight per position (e.g., 0.10 for 10%)
    max_iterations : int
        Maximum iterations to prevent infinite loops

    Returns:
    --------
    tuple (np.ndarray, bool)
        - Capped weights (sum = 1.0, all weights <= max_weight if possible)
        - success: True if constraint was achievable, False if mathematically impossible
    """
    n_assets = len(weights)
    min_possible_weight = 1.0 / n_assets

    # Check if constraint is mathematically achievable
    # With n assets summing to 1.0, minimum max weight is 1/n
    if max_weight < min_possible_weight - 1e-9:
        logging.warning(
            f"     ⚠️  Max weight {max_weight:.1%} is impossible with {n_assets} assets "
            f"(minimum achievable: {min_possible_weight:.1%})"
        )
        return weights.copy(), False

    weights = weights.copy()

    for _ in range(max_iterations):
        # Find weights exceeding cap
        excess_mask = weights > max_weight
        if not excess_mask.any():
            break  # All weights within limit

        # Calculate excess to redistribute
        excess = weights[excess_mask] - max_weight
        total_excess = excess.sum()

        # Cap the exceeding weights
        weights[excess_mask] = max_weight

        # Redistribute excess to weights below cap (proportionally)
        below_cap_mask = weights < max_weight
        if below_cap_mask.any():
            # Proportional redistribution based on current weights
            below_weights = weights[below_cap_mask]
            redistribution = total_excess * (below_weights / below_weights.sum())
            weights[below_cap_mask] += redistribution

    # Final normalization to ensure sum = 1.0
    weights = weights / weights.sum()

    return weights, True


def _fallback_equal_weight(returns_df, n=20, periods_per_year=252):
    """
    Emergency fallback: Equal weight portfolio on top N assets by Sharpe.

    Used when screening is too aggressive and eliminates most assets.
    """
    logging.warning(
        f"FALLBACK: Creating equal-weight portfolio with {min(n, len(returns_df.columns))} assets"
    )

    MIN_VOL = 0.001
    sharpe_scores = {}

    for col in returns_df.columns:
        ret = returns_df[col].dropna()
        if len(ret) == 0:
            continue

        mean_ret = ret.mean() * periods_per_year
        std_ret = max(ret.std() * np.sqrt(periods_per_year), MIN_VOL)
        sharpe_scores[col] = mean_ret / std_ret

    # Select top N by Sharpe
    top_assets = sorted(sharpe_scores, key=sharpe_scores.get, reverse=True)[
        : min(n, len(sharpe_scores))
    ]

    n_selected = len(top_assets)

    fallback_metrics = pd.DataFrame(
        {
            "composite_score": [1.0 / n_selected] * n_selected,
            "sharpe": [sharpe_scores[asset] for asset in top_assets],
        },
        index=top_assets,
    )

    return {
        "metrics": fallback_metrics,
        "returns": returns_df[top_assets],
        "fallback": True,
        "n_removed_quality": 0,
        "n_removed_sharpe": 0,
        "n_removed_correlation": 0,
    }


# =============================================================================
# STAGE 1: Multi-Criteria Screening
# =============================================================================


def stage1_multi_criteria_screening(returns_df, config):
    """
    Stage 1: Filter assets using multiple quantitative metrics.

    Process:
        1. Data quality filter (sufficient trading days)
        2. Calculate 10 performance metrics per asset
        3. Remove extreme underperformers (Sharpe threshold)
        4. Remove highly correlated duplicates
        5. Rank by composite score, select top N

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Log returns (from load_and_preprocess_data)
    config : dict
        Configuration parameters

    Returns:
    --------
    dict with keys:
        'metrics': pd.DataFrame with composite scores and metrics
        'returns': pd.DataFrame with filtered assets
        'n_removed_quality': int
        'n_removed_sharpe': int
        'n_removed_correlation': int
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 1: Multi-Criteria Screening (350+ → 100 assets)")
    logging.info("=" * 80)

    # FIX #3: Freeze data snapshot (prevent staleness between stages)
    returns_snapshot = returns_df.copy()

    # Get annualization factor from config (252 for daily, 52 for weekly, 12 for monthly)
    periods_per_year = config.get("periods_per_year", 252)

    logging.info(f"Starting with {len(returns_snapshot.columns)} assets")

    # === Filter 1: Data Quality ===
    # Force-include tickers bypass the normal MIN_TRADING_PERIODS threshold but
    # still require FORCE_MIN_PERIODS non-NaN periods for stable covariance /
    # Sharpe estimates.  Below that floor we refuse to admit them and log a
    # loud warning so the user can widen the date range or drop the ticker.
    FORCE_MIN_PERIODS = 90  # ~3 months daily; floor for forced-ticker admission
    MIN_TRADING_PERIODS = config.get("min_trading_days", 500)
    force_tickers_set = set(config.get("force_tickers") or [])

    valid_assets = []
    forced_under_quality = []  # (ticker, n_periods) admitted below MIN_TRADING_PERIODS
    forced_below_floor = []  # (ticker, n_periods) rejected: < FORCE_MIN_PERIODS

    for col in returns_snapshot.columns:
        non_na_count = int(returns_snapshot[col].notna().sum())
        if non_na_count >= MIN_TRADING_PERIODS:
            valid_assets.append(col)
        elif col in force_tickers_set:
            if non_na_count >= FORCE_MIN_PERIODS:
                valid_assets.append(col)
                forced_under_quality.append((col, non_na_count))
            else:
                forced_below_floor.append((col, non_na_count))

    n_removed_quality = len(returns_snapshot.columns) - len(valid_assets)

    logging.info(
        f"  Data quality filter (>= {MIN_TRADING_PERIODS} periods): {len(returns_snapshot.columns)} → {len(valid_assets)} assets"
    )
    if forced_under_quality:
        logging.warning(
            f"  ⚠️  Force-include: {len(forced_under_quality)} ticker(s) admitted under "
            f"normal {MIN_TRADING_PERIODS}-period threshold (>= {FORCE_MIN_PERIODS}-period "
            f"floor for forced tickers): {forced_under_quality}"
        )
    if forced_below_floor:
        logging.warning(
            f"  ⚠️  Force-include: {len(forced_below_floor)} ticker(s) SKIPPED — fewer "
            f"than {FORCE_MIN_PERIODS} non-NaN periods (insufficient for stable estimates): "
            f"{forced_below_floor}"
        )

    # FIX #4: Fallback if too few assets
    if len(valid_assets) < config.get("stage1_min_assets", 50):
        logging.warning(
            f"Too few assets after quality filter ({len(valid_assets)} < {config['stage1_min_assets']})"
        )
        return _fallback_equal_weight(
            returns_snapshot[valid_assets] if valid_assets else returns_snapshot,
            periods_per_year=periods_per_year,
        )

    returns_filtered = returns_snapshot[valid_assets].copy()

    # === Calculate Performance Metrics ===
    # FIX #2: Volatility floor to prevent division by zero
    MIN_VOL = 0.001  # 0.1% annualized

    metrics = {}

    for col in returns_filtered.columns:
        ret = returns_filtered[col].dropna()

        if len(ret) == 0:
            continue

        # Basic statistics
        mean_ret = ret.mean() * periods_per_year
        std_ret = max(ret.std() * np.sqrt(periods_per_year), MIN_VOL)  # FLOOR APPLIED
        sharpe = mean_ret / std_ret

        # Downside risk (semi-deviation)
        downside_ret = ret[ret < 0]
        if len(downside_ret) > 0:
            downside_std = max(downside_ret.std() * np.sqrt(periods_per_year), MIN_VOL)
        else:
            downside_std = std_ret
        sortino = mean_ret / downside_std

        # Win metrics
        win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0
        avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0
        avg_loss = abs(ret[ret < 0].mean()) if (ret < 0).any() else 1e-6
        profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss + 1e-10)

        # Drawdown
        cum_returns = (1 + ret).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar = abs(mean_ret / (max_drawdown + 1e-10))

        # Tail risk
        var_95 = ret.quantile(0.05)
        cvar_95 = ret[ret <= var_95].mean() if (ret <= var_95).any() else var_95

        metrics[col] = {
            "mean_return": mean_ret,
            "volatility": std_ret,
            "sharpe": sharpe,
            "sortino": sortino,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "cvar_95": cvar_95,
            "downside_std": downside_std,
        }

    metrics_df = pd.DataFrame(metrics).T

    # Snapshot all metrics before filtering (needed for force-include recovery)
    all_metrics_pre_filter = metrics_df.copy()

    # === Filter 2: Remove Underperformers ===
    MIN_SHARPE = config.get("min_sharpe", -0.5)
    initial_count = len(metrics_df)
    metrics_df = metrics_df[metrics_df["sharpe"] >= MIN_SHARPE]
    n_removed_sharpe = initial_count - len(metrics_df)

    logging.info(
        f"  Sharpe filter (>= {MIN_SHARPE}): {initial_count} → {len(metrics_df)} assets"
    )

    # === Filter 2b: Remove Low Annual Return Assets ===
    MIN_ANNUAL_RETURN = config.get("min_annual_return", None)
    n_removed_return = 0

    if MIN_ANNUAL_RETURN is not None:
        initial_count_return = len(metrics_df)
        metrics_df = metrics_df[metrics_df["mean_return"] >= MIN_ANNUAL_RETURN]
        n_removed_return = initial_count_return - len(metrics_df)

        logging.info(
            f"  Annual return filter (>= {MIN_ANNUAL_RETURN:.1%}): {initial_count_return} → {len(metrics_df)} assets"
        )

    # Fallback check
    if len(metrics_df) < config.get("stage1_min_assets", 50):
        logging.warning(
            f"Too few assets after Sharpe/Return filters ({len(metrics_df)} < {config['stage1_min_assets']})"
        )
        return _fallback_equal_weight(
            returns_filtered[metrics_df.index], periods_per_year=periods_per_year
        )

    # === Filter 3: Remove High Correlations ===
    MAX_CORR = config.get("max_correlation", 0.95)
    # force_tickers_set already defined above for data-quality filter exemption
    corr_matrix = returns_filtered[metrics_df.index].corr()

    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            asset_i = corr_matrix.columns[i]
            asset_j = corr_matrix.columns[j]

            if abs(corr_matrix.iloc[i, j]) > MAX_CORR:
                # Never remove a force-included ticker from a correlated pair
                i_forced = asset_i in force_tickers_set
                j_forced = asset_j in force_tickers_set
                if i_forced and j_forced:
                    continue  # Both forced -- keep both
                elif i_forced:
                    to_remove.add(asset_j)
                elif j_forced:
                    to_remove.add(asset_i)
                # Neither forced -- remove the one with lower Sharpe
                elif (
                    metrics_df.loc[asset_i, "sharpe"]
                    < metrics_df.loc[asset_j, "sharpe"]
                ):
                    to_remove.add(asset_i)
                else:
                    to_remove.add(asset_j)

    metrics_df = metrics_df.drop(index=to_remove, errors="ignore")
    n_removed_correlation = len(to_remove)

    logging.info(
        f"  Correlation filter (< {MAX_CORR}): Removed {n_removed_correlation} duplicates → {len(metrics_df)} assets"
    )

    # === Composite Score Ranking ===
    scaler = StandardScaler()

    score_components = metrics_df[
        ["sharpe", "sortino", "calmar", "win_rate", "profit_factor"]
    ].copy()
    score_components["max_drawdown_inv"] = -metrics_df["max_drawdown"]
    score_components["cvar_inv"] = -metrics_df["cvar_95"]

    # Normalize to [0, 1]
    score_components_norm = pd.DataFrame(
        scaler.fit_transform(score_components),
        index=score_components.index,
        columns=score_components.columns,
    )

    # Weighted composite (configurable)
    score_weights = {
        "sharpe": 0.30,
        "sortino": 0.20,
        "calmar": 0.15,
        "win_rate": 0.10,
        "profit_factor": 0.10,
        "max_drawdown_inv": 0.10,
        "cvar_inv": 0.05,
    }

    metrics_df["composite_score"] = sum(
        score_components_norm[metric] * weight
        for metric, weight in score_weights.items()
    )

    # === Select Top N ===
    TOP_N = config.get("stage1_top_n", 100)
    top_assets = metrics_df.nlargest(min(TOP_N, len(metrics_df)), "composite_score")

    # === Force-include tickers that were eliminated by Sharpe/correlation/rank ===
    # Forced tickers that passed data quality (exist in returns_filtered) but were
    # dropped by subsequent filters are re-added here.  Their metrics are computed
    # from the pre-filter snapshot so HRP receives them in its covariance matrix.
    n_force_included = 0
    force_tickers = config.get("force_tickers") or []
    if force_tickers:
        # Tickers available after data quality filter but missing from top_assets
        recoverable = [
            t
            for t in force_tickers
            if t in all_metrics_pre_filter.index and t not in top_assets.index
        ]
        # Tickers that failed data quality (not in returns_filtered at all)
        not_in_data = [t for t in force_tickers if t not in returns_filtered.columns]
        # Tickers already in top_assets (no action needed)
        already_in = [t for t in force_tickers if t in top_assets.index]

        if already_in:
            logging.info(
                f"  Force-include: {len(already_in)} ticker(s) already passed filters: {already_in}"
            )

        if not_in_data:
            logging.warning(
                f"  Force-include: {len(not_in_data)} ticker(s) not in data or failed "
                f"data quality filter (< {MIN_TRADING_PERIODS} periods), cannot include: {not_in_data}"
            )

        if recoverable:
            # Compute composite scores for recoverable tickers using the same
            # normalization parameters (refit on union of top_assets + recoverable)
            recovery_metrics = all_metrics_pre_filter.loc[recoverable].copy()

            # Build score components for recoverable tickers
            recovery_components = recovery_metrics[
                ["sharpe", "sortino", "calmar", "win_rate", "profit_factor"]
            ].copy()
            recovery_components["max_drawdown_inv"] = -recovery_metrics["max_drawdown"]
            recovery_components["cvar_inv"] = -recovery_metrics["cvar_95"]

            # Use the already-fitted scaler to transform (consistent scale)
            recovery_norm = pd.DataFrame(
                scaler.transform(recovery_components),
                index=recovery_components.index,
                columns=recovery_components.columns,
            )

            recovery_metrics["composite_score"] = sum(
                recovery_norm[metric] * weight
                for metric, weight in score_weights.items()
            )

            top_assets = pd.concat([top_assets, recovery_metrics])
            n_force_included = len(recoverable)
            logging.info(
                f"  Force-included {n_force_included} ticker(s) into Stage 1 output: {recoverable}"
            )

    logging.info(
        f"\nStage 1 Complete: {len(returns_snapshot.columns)} → {len(top_assets)} assets"
        + (
            f" (including {n_force_included} force-included)"
            if n_force_included
            else ""
        )
    )
    logging.info(
        f"  Removed: {n_removed_quality} (quality) + {n_removed_sharpe} (Sharpe) + {n_removed_return} (annual return) + {n_removed_correlation} (correlation)"
    )
    logging.info(
        f"  Score range: [{top_assets['composite_score'].min():.3f}, {top_assets['composite_score'].max():.3f}]"
    )
    logging.info(
        f"  Sharpe range: [{top_assets['sharpe'].min():.3f}, {top_assets['sharpe'].max():.3f}]"
    )
    logging.info(
        f"  Annual return range: [{top_assets['mean_return'].min():.1%}, {top_assets['mean_return'].max():.1%}]"
    )

    return {
        "metrics": top_assets,
        "returns": returns_filtered[top_assets.index],
        "n_removed_quality": n_removed_quality,
        "n_removed_sharpe": n_removed_sharpe,
        "n_removed_return": n_removed_return,
        "n_removed_correlation": n_removed_correlation,
        "n_force_included": n_force_included,
        "fallback": False,
    }


# =============================================================================
# STAGE 2: Diversification Selection
# =============================================================================


def stage2_direct_selection(stage1_output, config):
    """
    Stage 2: Direct top-N selection by composite score.

    Rationale:
    - Clustering caused non-monotonic behavior (stage2-target 40 beat 80)
    - Quality-aware allocation was non-deterministic (request 80, get 46)
    - Composite score already includes diversity via correlation filter in Stage 1
    - Simpler = more maintainable, predictable, monotonic

    Returns:
    --------
    dict with keys:
        'metrics': DataFrame (exactly target_n rows)
        'returns': DataFrame
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 2: Direct Selection (Top-N by Composite Score)")
    logging.info("=" * 80)

    returns_df = stage1_output["returns"]
    metrics_df = stage1_output["metrics"]

    TARGET_N = config.get("stage2_target_n", 40)

    logging.info(
        f"Selecting top {TARGET_N} assets from {len(returns_df.columns)} by composite score"
    )

    # Simple: Take top N by composite score (no clustering)
    top_n_metrics = metrics_df.nlargest(
        min(TARGET_N, len(metrics_df)), "composite_score"
    )

    # === Force-include tickers that survived Stage 1 but missed the rank cut ===
    n_force_included_s2 = 0
    force_tickers = config.get("force_tickers") or []
    if force_tickers:
        missing = [
            t
            for t in force_tickers
            if t in metrics_df.index and t not in top_n_metrics.index
        ]
        if missing:
            top_n_metrics = pd.concat([top_n_metrics, metrics_df.loc[missing]])
            n_force_included_s2 = len(missing)
            logging.info(
                f"  Force-included {n_force_included_s2} ticker(s) into Stage 2 output: {missing}"
            )

    logging.info(
        f"\nStage 2 Complete: {len(returns_df.columns)} → {len(top_n_metrics)} assets"
        + (
            f" (including {n_force_included_s2} force-included)"
            if n_force_included_s2
            else ""
        )
    )
    logging.info(
        f"  Selected score range: [{top_n_metrics['composite_score'].min():.3f}, {top_n_metrics['composite_score'].max():.3f}]"
    )
    logging.info(
        f"  Selected Sharpe range: [{top_n_metrics['sharpe'].min():.3f}, {top_n_metrics['sharpe'].max():.3f}]"
    )

    return {"metrics": top_n_metrics, "returns": returns_df[top_n_metrics.index]}


# =============================================================================
# STAGE 3: Global Optimization
# =============================================================================


def hierarchical_risk_parity(cov_matrix):
    """
    Hierarchical Risk Parity allocation (Lopez de Prado 2016).

    Algorithm:
        1. Convert covariance to correlation distance
        2. Hierarchical clustering
        3. Recursive bisection with inverse variance weighting

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Annualized covariance matrix

    Returns:
    --------
    np.ndarray
        Portfolio weights (sum = 1.0)
    """
    # Convert to correlation
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)

    # Force perfect symmetry (fix numerical errors)
    corr = (corr + corr.T) / 2

    # Clip correlation to [-1, 1] (fix floating point errors)
    corr = np.clip(corr, -1, 1)

    # Distance matrix (angular distance)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))  # Clip to [0, 1] before sqrt

    # Force symmetry in distance matrix
    dist = (dist + dist.T) / 2

    # Force diagonal to zero (distance from asset to itself must be 0)
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    link = sch.linkage(squareform(dist), method="single")
    sort_ix = sch.dendrogram(link, no_plot=True)["leaves"]

    def _cluster_variance(cov, indices):
        """Compute cluster variance using inverse-variance portfolio weights."""
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)

        # Guard against zero/negative numerical artifacts in the diagonal
        diag = np.clip(diag, 1e-12, None)

        inv_diag = 1.0 / diag
        ivp = inv_diag / np.sum(inv_diag)
        return ivp @ sub_cov @ ivp

    def _recursive_bisection(cov, indices):
        """Recursively allocate weights via bisection."""
        if len(indices) == 1:
            return np.array([1.0])

        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]

        # Cluster variances using inverse-variance portfolio within each cluster
        left_var = _cluster_variance(cov, left_idx)
        right_var = _cluster_variance(cov, right_idx)

        # Allocate by inverse variance
        alpha = 1 - left_var / (left_var + right_var)

        # Recurse
        left_weights = alpha * _recursive_bisection(cov, left_idx)
        right_weights = (1 - alpha) * _recursive_bisection(cov, right_idx)

        return np.concatenate([left_weights, right_weights])

    # Sort covariance matrix by cluster order
    cov_sorted = cov_matrix[sort_ix, :][:, sort_ix]
    weights = _recursive_bisection(cov_sorted, list(range(len(sort_ix))))

    # Restore original asset order
    weights_orig = np.zeros(len(weights))
    weights_orig[sort_ix] = weights

    return weights_orig


def stage3_global_optimization(stage2_output, config):
    """
    Stage 3: Optimize portfolios with global search options.

    Generates 4+ portfolio alternatives:
        1. Max Sharpe (SLSQP - fast, local optimum)
        2. Max Sharpe (DE - slow, global search, optional)
        3. Min Volatility (SLSQP - convex, always global)
        4. HRP (deterministic, robust to estimation error)
        5. Efficient Frontier (5 target volatility points)

    Parameters:
    -----------
    stage2_output : dict
        Output from Stage 2
    config : dict
        Configuration parameters

    Returns:
    --------
    dict with keys for each portfolio type
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 3: Portfolio Optimization (40 assets → Final Portfolios)")
    logging.info("=" * 80)

    returns_df = stage2_output["returns"]

    # Get annualization factor from config (252 for daily, 52 for weekly, 12 for monthly)
    periods_per_year = config.get("periods_per_year", 252)

    logging.info(f"Optimizing {len(returns_df.columns)} assets")

    # FIX #1: Use Ledoit-Wolf for covariance (CRITICAL - prevents singularity)
    lw = LedoitWolf()
    lw.fit(returns_df)
    cov_matrix = lw.covariance_ * periods_per_year  # Annualized

    logging.info(f"  Ledoit-Wolf shrinkage intensity: {lw.shrinkage_:.4f}")
    logging.info(f"  Covariance condition number: {np.linalg.cond(cov_matrix):.2f}")

    mean_returns = returns_df.mean().values * periods_per_year
    n_assets = len(returns_df.columns)

    # Force-include tickers must survive the dust filter even if the optimizer
    # assigns them effectively zero weight.  Their weight is still whatever the
    # optimizer chose (bounded by --max-weight); we just guarantee they appear
    # in the exported weights dict.
    force_tickers_set = set(config.get("force_tickers") or [])
    forced_mask = np.array(
        [c in force_tickers_set for c in returns_df.columns], dtype=bool
    )

    # FIX #2: Volatility floor (prevents division by zero)
    MIN_VOL = 0.001

    def portfolio_stats(weights):
        """Calculate portfolio metrics with safeguards."""
        ret = mean_returns @ weights
        vol = max(np.sqrt(weights @ cov_matrix @ weights), MIN_VOL)
        sharpe = (ret - config["risk_free_rate"]) / vol
        return ret, vol, sharpe

    def neg_sharpe(weights):
        """Objective function for Sharpe maximization."""
        return -portfolio_stats(weights)[2]

    # Constraints and bounds
    max_weight_cap = config.get("max_weight", None)
    if max_weight_cap is not None:
        # Cap max weight per position (e.g., 0.10 for 10%)
        bounds = [(0, max_weight_cap) for _ in range(n_assets)]
        logging.info(f"     Max weight constraint: {max_weight_cap:.1%} per position")
    else:
        bounds = [(0, 1) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    initial_weights = np.ones(n_assets) / n_assets

    results = {}

    # === 1. Maximum Sharpe (SLSQP - Fast) ===
    logging.info("\n  1. Maximum Sharpe Ratio (SLSQP - fast)")

    opt_slsqp = sco.minimize(
        neg_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if opt_slsqp.success:
        weights_slsqp = opt_slsqp.x
        ret_slsqp, vol_slsqp, sharpe_slsqp = portfolio_stats(weights_slsqp)

        # Filter dust positions
        significant = (weights_slsqp > config["min_weight_threshold"]) | forced_mask
        clean_weights = weights_slsqp[significant]
        clean_assets = returns_df.columns[significant]

        results["max_sharpe"] = {
            "weights": dict(zip(clean_assets, clean_weights)),
            "n_positions": len(clean_assets),
            "return": ret_slsqp,
            "volatility": vol_slsqp,
            "sharpe": sharpe_slsqp,
            "max_weight": clean_weights.max(),
            "min_weight": clean_weights.min(),
            "concentration": (clean_weights**2).sum(),
            "method": "SLSQP (fast, local optimum)",
            "convergence_iters": opt_slsqp.nit,
            "status": "success",
        }

        logging.info(
            f"     ✅ Sharpe={sharpe_slsqp:.4f}, Vol={vol_slsqp:.4f}, Positions={len(clean_assets)}"
        )
        logging.info(
            f"     Weight range: [{clean_weights.min():.4f}, {clean_weights.max():.4f}], Iters={opt_slsqp.nit}"
        )
    else:
        logging.warning(f"     ❌ SLSQP failed: {opt_slsqp.message}")
        results["max_sharpe"] = {"status": "failed", "message": opt_slsqp.message}

    # NOTE: Exhaustive global search (DE, dual_annealing, basin_hopping) removed
    # Reason: 10 methods, 11.5 minutes → converged to IDENTICAL solution as SLSQP
    # Evidence: Weight distance = 0.0 between all methods (same local optimum)
    # Conclusion: SLSQP finds strong optimum for this quasi-convex problem
    # For validation: Use out-of-sample testing instead of exhaustive search

    # === 2. Minimum Volatility (Always Global for Convex Problems) ===
    logging.info("\n  3. Minimum Volatility (SLSQP - convex problem)")

    def portfolio_vol(weights):
        return portfolio_stats(weights)[1]

    opt_minvol = sco.minimize(
        portfolio_vol,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if opt_minvol.success:
        weights_minvol = opt_minvol.x
        ret_minvol, vol_minvol, sharpe_minvol = portfolio_stats(weights_minvol)

        significant = (weights_minvol > config["min_weight_threshold"]) | forced_mask
        clean_weights = weights_minvol[significant]
        clean_assets = returns_df.columns[significant]

        results["min_volatility"] = {
            "weights": dict(zip(clean_assets, clean_weights)),
            "n_positions": len(clean_assets),
            "return": ret_minvol,
            "volatility": vol_minvol,
            "sharpe": sharpe_minvol,
            "max_weight": clean_weights.max(),
            "concentration": (clean_weights**2).sum(),
            "method": "SLSQP (convex problem, guaranteed global)",
            "status": "success",
        }

        logging.info(
            f"     ✅ Sharpe={sharpe_minvol:.4f}, Vol={vol_minvol:.4f}, Positions={len(clean_assets)}"
        )
    else:
        logging.warning(f"     ❌ Min Vol optimization failed: {opt_minvol.message}")
        results["min_volatility"] = {"status": "failed", "message": opt_minvol.message}

    # === 4. Hierarchical Risk Parity ===
    logging.info("\n  4. Hierarchical Risk Parity (deterministic)")

    weights_hrp = hierarchical_risk_parity(cov_matrix)

    # Apply max_weight constraint if specified (HRP doesn't use optimization bounds)
    if max_weight_cap is not None:
        weights_hrp, cap_success = _cap_weights(weights_hrp, max_weight_cap)
        if cap_success:
            logging.info(f"     Applied max weight cap: {max_weight_cap:.1%}")
        else:
            logging.warning(
                f"     Max weight cap could not be applied (need more assets)"
            )

    ret_hrp, vol_hrp, sharpe_hrp = portfolio_stats(weights_hrp)

    significant = (weights_hrp > config["min_weight_threshold"]) | forced_mask
    clean_weights = weights_hrp[significant]
    clean_assets = returns_df.columns[significant]

    results["hrp"] = {
        "weights": dict(zip(clean_assets, clean_weights)),
        "n_positions": len(clean_assets),
        "return": ret_hrp,
        "volatility": vol_hrp,
        "sharpe": sharpe_hrp,
        "max_weight": clean_weights.max(),
        "concentration": (clean_weights**2).sum(),
        "method": "HRP (deterministic, robust to estimation error)",
        "status": "success",
    }

    logging.info(
        f"     ✅ Sharpe={sharpe_hrp:.4f}, Vol={vol_hrp:.4f}, Positions={len(clean_assets)}"
    )

    # === 5. Efficient Frontier (5 Target Volatilities) ===
    logging.info("\n  5. Efficient Frontier")

    # Check prerequisites
    min_vol_ok = results.get("min_volatility", {}).get("status") == "success"
    max_sharpe_ok = results.get("max_sharpe", {}).get("status") == "success"

    logging.info(
        f"     Prerequisites: Min Vol = {'✅' if min_vol_ok else '❌'}, Max Sharpe = {'✅' if max_sharpe_ok else '❌'}"
    )

    if min_vol_ok and max_sharpe_ok:
        vol_min = results["min_volatility"]["volatility"]
        vol_max = results["max_sharpe"]["volatility"] * 1.3

        target_vols = np.linspace(vol_min, vol_max, 5)

        logging.info(
            f"     Computing frontier from vol={vol_min:.4f} to vol={vol_max:.4f} (5 points)..."
        )

        results["efficient_frontier"] = []
        failed_count = 0

        # Import constraint classes
        from scipy.optimize import LinearConstraint, NonlinearConstraint

        # Use min_vol portfolio as starting point for first frontier point
        ef_start_weights = results["min_volatility"]["weights"]
        ef_start_array = np.zeros(n_assets)
        for asset_name, weight in ef_start_weights.items():
            idx = list(returns_df.columns).index(asset_name)
            ef_start_array[idx] = weight

        for i, target_vol in enumerate(target_vols):

            def neg_return(weights):
                return -portfolio_stats(weights)[0]

            # Sum constraint: sum(weights) = 1
            sum_constraint = LinearConstraint(np.ones(n_assets), 1, 1)

            # Volatility constraint: portfolio_vol = target_vol
            def vol_func(w):
                return portfolio_stats(w)[1]

            vol_constraint = NonlinearConstraint(vol_func, target_vol, target_vol)

            # Warm start: Use previous frontier point (helps convergence)
            start_point = (
                ef_start_array
                if i == 0
                else opt_ef.x
                if opt_ef.success
                else initial_weights
            )

            opt_ef = sco.minimize(
                neg_return,
                start_point,
                method="trust-constr",  # More robust than SLSQP
                bounds=sco.Bounds(0, max_weight_cap if max_weight_cap else 1),
                constraints=[sum_constraint, vol_constraint],
                options={
                    "maxiter": 5000,
                    "verbose": 0,
                },  # Increased to 5000 for difficult points
            )

            # Fallback: If trust-constr fails, try SLSQP as backup
            if not opt_ef.success:
                vol_constraint_dict = {
                    "type": "eq",
                    "fun": lambda w, tv=target_vol: portfolio_stats(w)[1] - tv,
                }
                all_constraints = (constraints, vol_constraint_dict)

                opt_ef = sco.minimize(
                    neg_return,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=all_constraints,
                    options={"maxiter": 1000},
                )

            if opt_ef.success:
                weights_ef = opt_ef.x
                ret_ef, vol_ef, sharpe_ef = portfolio_stats(weights_ef)

                significant = (weights_ef > config["min_weight_threshold"]) | forced_mask
                clean_weights = weights_ef[significant]
                clean_assets = returns_df.columns[significant]

                results["efficient_frontier"].append(
                    {
                        "target_volatility": target_vol,
                        "weights": dict(zip(clean_assets, clean_weights)),
                        "n_positions": len(clean_assets),
                        "return": ret_ef,
                        "volatility": vol_ef,
                        "sharpe": sharpe_ef,
                        "max_weight": clean_weights.max(),
                    }
                )
            else:
                # Log why this frontier point failed
                failed_count += 1
                logging.warning(
                    f"     ⚠️  EF point #{failed_count} failed (target_vol={target_vol:.4f}): {opt_ef.message}"
                )

        logging.info(
            f"     ✅ Generated {len(results['efficient_frontier'])} frontier portfolios (0 failures)"
            if failed_count == 0
            else f"     ⚠️  Generated {len(results['efficient_frontier'])} frontier portfolios ({failed_count} failed)"
        )

        # Show efficient frontier details
        if results["efficient_frontier"]:
            logging.info(f"\n     Efficient Frontier Points:")
            for i, ef_point in enumerate(results["efficient_frontier"], 1):
                logging.info(
                    f"       {i}. Vol={ef_point['volatility']:.4f}, Sharpe={ef_point['sharpe']:.4f}, Positions={ef_point['n_positions']}"
                )
    else:
        logging.warning(
            f"     ⚠️  Skipping efficient frontier (min_vol or max_sharpe failed)"
        )
        results["efficient_frontier"] = []  # Initialize as empty to prevent KeyError

    logging.info(
        f"\nStage 3 Complete: Generated {sum(1 for k, v in results.items() if k != 'efficient_frontier' and v.get('status') == 'success')} portfolios"
    )

    return results


# =============================================================================
# STAGE 4: Analysis & Reporting
# =============================================================================


def stage4_analysis_and_reporting(stage3_results, stage2_output, config):
    """
    Stage 4: Present results for decision-making.

    Outputs:
        1. Comparison table (all portfolios side-by-side)
        2. Complete holdings (ALL weights per portfolio)
        3. Asset overlap analysis
        4. Decision recommendations

    Parameters:
    -----------
    stage3_results : dict
        Output from Stage 3
    stage2_output : dict
        Output from Stage 2
    config : dict
        Configuration

    Returns:
    --------
    dict with keys:
        'comparison_df': pd.DataFrame
        'common_assets': list
        'recommendations': list
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 4: Portfolio Analysis & Complete Results")
    logging.info("=" * 80)

    returns_df = stage2_output["returns"]

    # === Comparison Table ===
    comparison = []

    for portfolio_name, portfolio_data in stage3_results.items():
        if portfolio_name == "efficient_frontier":
            continue

        if portfolio_data.get("status") != "success":
            continue

        comparison.append(
            {
                "Portfolio": portfolio_name,
                "Method": portfolio_data["method"],
                "Positions": portfolio_data["n_positions"],
                "Return": f"{portfolio_data['return']:.4f}",
                "Volatility": f"{portfolio_data['volatility']:.4f}",
                "Sharpe": f"{portfolio_data['sharpe']:.4f}",
                "Max Weight": f"{portfolio_data['max_weight']:.2%}",
                "HHI": f"{portfolio_data.get('concentration', 0):.4f}",
            }
        )

    comparison_df = pd.DataFrame(comparison)

    logging.info("\nPORTFOLIO COMPARISON TABLE")
    logging.info("-" * 80)
    logging.info("\n" + comparison_df.to_string(index=False))

    # === Complete Portfolio Holdings (ALL weights, not just top 10) ===
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETE PORTFOLIO HOLDINGS (ALL POSITIONS)")
    logging.info("=" * 80)

    for portfolio_name in ["max_sharpe", "max_sharpe", "min_volatility", "hrp"]:
        if (
            portfolio_name not in stage3_results
            or stage3_results[portfolio_name].get("status") != "success"
        ):
            continue

        portfolio_data = stage3_results[portfolio_name]
        weights = portfolio_data["weights"]
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        # Header with portfolio metrics
        logging.info(f"\n{portfolio_name.upper().replace('_', ' ')}:")
        logging.info(f"  Return:      {portfolio_data['return']:.4f}")
        logging.info(f"  Volatility:  {portfolio_data['volatility']:.4f}")
        logging.info(f"  Sharpe:      {portfolio_data['sharpe']:.4f}")
        logging.info(f"  Positions:   {portfolio_data['n_positions']}")
        logging.info(f"  Max Weight:  {portfolio_data['max_weight']:.4f}")
        logging.info(f"  HHI:         {portfolio_data.get('concentration', 0):.4f}")
        logging.info(f"\n  All Holdings:")

        # Show ALL weights (not just top 10)
        for i, (asset, weight) in enumerate(sorted_weights, 1):
            logging.info(
                f"    {i:3d}. {asset:30s} {weight:8.4f} ({weight * 100:6.2f}%)"
            )

    # === Asset Overlap Analysis ===
    logging.info("\n" + "=" * 80)
    logging.info("ASSET OVERLAP ANALYSIS")
    logging.info("=" * 80)

    portfolio_sets = {
        name: set(data["weights"].keys())
        for name, data in stage3_results.items()
        if name != "efficient_frontier" and data.get("status") == "success"
    }

    common_assets = []
    if len(portfolio_sets) >= 2:
        common_assets = list(set.intersection(*portfolio_sets.values()))

        logging.info(f"\nAssets in ALL portfolios ({len(common_assets)}):")
        for asset in sorted(common_assets):
            weights_str = " | ".join(
                [
                    f"{name}: {stage3_results[name]['weights'][asset]:6.2%}"
                    for name in portfolio_sets.keys()
                    if asset in stage3_results[name]["weights"]
                ]
            )
            logging.info(f"  {asset:20s} {weights_str}")

    # === Decision Insights ===
    logging.info("\n" + "=" * 80)
    logging.info("DECISION INSIGHTS & RECOMMENDATIONS")
    logging.info("=" * 80)

    recommendations = []

    # Get primary portfolio (max_sharpe or max_sharpe if available)
    primary = stage3_results.get("max_sharpe", stage3_results.get("max_sharpe", {}))

    if primary.get("status") == "success":
        logging.info(f"\n1. Position Counts (what optimizer naturally chose):")
        for name in ["max_sharpe", "max_sharpe", "min_volatility", "hrp"]:
            if (
                name in stage3_results
                and stage3_results[name].get("status") == "success"
            ):
                logging.info(
                    f"   {name:20s}: {stage3_results[name]['n_positions']} positions"
                )

        logging.info(f"\n2. Concentration Analysis (HHI - lower = more diversified):")
        for name in ["max_sharpe", "max_sharpe", "min_volatility", "hrp"]:
            if (
                name in stage3_results
                and stage3_results[name].get("status") == "success"
            ):
                data = stage3_results[name]
                logging.info(
                    f"   {name:20s}: HHI={data.get('concentration', 0):.4f}, Max={data.get('max_weight', 0):.2%}"
                )

        logging.info(f"\n3. Recommendations:")

        # Position count recommendation
        if primary["n_positions"] <= 30:
            logging.info(
                f"   ✅ Position count ({primary['n_positions']}) is manageable - portfolio ready"
            )
            recommendations.append(
                {
                    "type": "success",
                    "message": f"Portfolio ready with {primary['n_positions']} positions",
                }
            )
        else:
            logging.info(
                f"   ⚠️  Many positions ({primary['n_positions']}) - consider cardinality constraint"
            )
            recommendations.append(
                {
                    "type": "action",
                    "message": f"Re-run with target_positions={int(primary['n_positions'] * 0.75)}",
                }
            )

        # Concentration recommendation
        if primary["max_weight"] > 0.15:
            logging.info(
                f"   ⚠️  High concentration ({primary['max_weight']:.1%} max) - consider max weight cap"
            )
            recommendations.append(
                {"type": "action", "message": f"Add max_weight=0.10 constraint"}
            )
        else:
            logging.info(
                f"   ✅ Concentration acceptable ({primary['max_weight']:.1%} max)"
            )

        # Global vs fast comparison
        if (
            "max_sharpe" in stage3_results
            and stage3_results["max_sharpe"].get("status") == "success"
        ):
            improvement = (
                stage3_results["max_sharpe"]["sharpe"]
                - stage3_results["max_sharpe"]["sharpe"]
            )
            if improvement > 0.05:
                logging.info(
                    f"   🎯 Global search significantly better (+{improvement:.4f} Sharpe) - use global result"
                )
                recommendations.append(
                    {"type": "use", "message": "Use max_sharpe (superior)"}
                )
            elif improvement > 0:
                logging.info(
                    f"   ℹ️  Global search marginally better (+{improvement:.4f} Sharpe)"
                )
            else:
                logging.info(
                    f"   ✅ Fast search sufficient (diff: {improvement:+.4f} Sharpe)"
                )

        # HRP robustness note
        if "hrp" in stage3_results and stage3_results["hrp"].get("status") == "success":
            hrp_sharpe = stage3_results["hrp"]["sharpe"]
            fast_sharpe = stage3_results["max_sharpe"]["sharpe"]

            if hrp_sharpe > fast_sharpe * 0.95:
                logging.info(
                    f"   💡 HRP is competitive ({hrp_sharpe:.4f} vs {fast_sharpe:.4f}) - consider for out-of-sample robustness"
                )
                recommendations.append(
                    {
                        "type": "consider",
                        "message": "HRP may be more robust for live trading",
                    }
                )

    # === Decision Recommender (Single Best Portfolio) ===
    logging.info("\n" + "=" * 80)
    logging.info("🎯 RECOMMENDED PORTFOLIO FOR PRODUCTION")
    logging.info("=" * 80)

    # Decision logic
    max_sharpe = stage3_results.get("max_sharpe", {})
    min_vol = stage3_results.get("min_volatility", {})
    hrp = stage3_results.get("hrp", {})

    if max_sharpe.get("status") == "success":
        sharpe_ms = max_sharpe["sharpe"]
        sharpe_hrp = hrp.get("sharpe", 0)
        sharpe_mv = min_vol.get("sharpe", 0)
        max_weight_ms = max_sharpe.get("max_weight", 0)

        # Decision tree
        if sharpe_hrp > sharpe_ms * 0.95:
            choice = "hrp"
            reason = f"HRP Sharpe ({sharpe_hrp:.2f}) within 5% of Max Sharpe ({sharpe_ms:.2f})"
            detail = "HRP is more robust out-of-sample (no mean estimation error)"
            confidence = "High"
        elif max_weight_ms > 0.15:
            choice = "min_volatility"
            reason = f"Max Sharpe too concentrated (max weight {max_weight_ms:.1%})"
            detail = "Min Volatility provides better risk management"
            confidence = "Medium"
        else:
            choice = "max_sharpe"
            reason = f"Highest risk-adjusted return (Sharpe {sharpe_ms:.2f})"
            detail = f"Acceptable concentration (max {max_weight_ms:.1%})"
            confidence = "High"

        logging.info(f"\nRecommended: {choice.upper()}")
        logging.info(f"  Reason:      {reason}")
        logging.info(f"  Details:     {detail}")
        logging.info(f"  Confidence:  {confidence}")

        logging.info(f"\nAlternatives:")
        for portfolio_name in ["max_sharpe", "min_volatility", "hrp"]:
            if portfolio_name != choice and portfolio_name in stage3_results:
                p = stage3_results[portfolio_name]
                if p.get("status") == "success":
                    logging.info(
                        f"  - {portfolio_name}: Sharpe {p['sharpe']:.2f}, Vol {p['volatility']:.3f}"
                    )

        logging.info(f"\n⚠️  IMPORTANT: Validate out-of-sample before live deployment")
        logging.info(
            f"    Recommended: Walk-forward backtest (2020-2025) to confirm robustness"
        )
    else:
        logging.warning(
            f"\n⚠️  Cannot provide recommendation (max_sharpe optimization failed)"
        )

    # === Force-ticker verification ===
    # For every successfully optimized portfolio, confirm that the user's
    # --force-tickers are present in the exported weights.  Regression guard:
    # if a forced ticker silently drops out of the final portfolio, the issue
    # is surfaced here immediately rather than discovered downstream.
    force_tickers = config.get("force_tickers") or []
    if force_tickers:
        logging.info("\n" + "=" * 80)
        logging.info("Force-ticker verification (requested via --force-tickers)")
        logging.info("=" * 80)
        for portfolio_name, label in [
            ("max_sharpe", "Max Sharpe"),
            ("min_volatility", "Min Volatility"),
            ("hrp", "HRP"),
        ]:
            port = stage3_results.get(portfolio_name, {})
            if port.get("status") != "success":
                logging.info(f"  {label}: skipped (optimization not successful)")
                continue
            weights = port.get("weights", {})
            present = {t: weights[t] for t in force_tickers if t in weights}
            missing = [t for t in force_tickers if t not in weights]
            if not missing:
                wstr = ", ".join(f"{t}={w:.4f}" for t, w in present.items())
                logging.info(
                    f"  ✅ {label}: all {len(present)} forced ticker(s) present ({wstr})"
                )
            else:
                present_str = (
                    ", ".join(f"{t}={w:.4f}" for t, w in present.items()) or "none"
                )
                logging.error(
                    f"  ❌ {label}: missing forced ticker(s) {missing} "
                    f"(present: {present_str})"
                )

    logging.info("\n" + "=" * 80)

    return {
        "comparison_df": comparison_df,
        "common_assets": common_assets,
        "recommendations": recommendations,
    }


# =============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# =============================================================================


def run_complete_workflow(csv_path, start_date, end_date, config):
    """
    Execute complete 4-stage workflow with error handling.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with price data
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    config : dict
        Configuration parameters

    Returns:
    --------
    dict with keys:
        'stage1': dict
        'stage2': dict
        'stage3': dict
        'stage4': dict
        'execution_time': float
        'status': str
    """
    start_time = time.time()

    logging.info("\n" + "=" * 80)
    logging.info(" PORTFOLIO EXPLORATION WORKFLOW - 4-Stage Global Optimization")
    logging.info("=" * 80)
    logging.info(f"\nInput:")
    logging.info(f"  CSV File:    {csv_path}")
    logging.info(f"  Date Range:  {start_date} to {end_date}")
    logging.info(f"\nConfiguration:")
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
        f"  Stage 1 Target:      {config['stage1_top_n']} assets (min Sharpe: {config['min_sharpe']}, min annual return: {min_return_str})"
    )
    logging.info(
        f"  Stage 2 Target:      {config['stage2_target_n']} assets (direct selection)"
    )
    logging.info(
        f"  Stage 3 Constraints: Risk-free rate: {config['risk_free_rate']}, max weight: {max_weight_str}"
    )
    # Global search removed (was 10x slower, 0% improvement)

    # Load data (uses portimization.py's function with synthetic fallback)
    seed = config.get("seed", 42)
    logging.info(f"\nLoading data... (seed={seed})")
    log_returns = load_and_preprocess_data(csv_path, start_date, end_date, seed=seed)

    if log_returns.empty:
        logging.error("\n" + "=" * 80)
        logging.error("FATAL ERROR: Data loading failed")
        logging.error("=" * 80)
        return {"status": "failed", "error": "Data loading failed"}

    logging.info(
        f"✅ Loaded {len(log_returns.columns)} assets with {len(log_returns)} trading days"
    )

    # Apply ticker filter if specified (from filter_investable_universe.py)
    ticker_filter = config.get("ticker_filter")
    if ticker_filter:
        available_tickers = set(log_returns.columns)
        filter_tickers = set(ticker_filter)
        valid_tickers = list(available_tickers & filter_tickers)
        missing_tickers = filter_tickers - available_tickers

        if missing_tickers:
            logging.warning(f"  Tickers not found in CSV: {sorted(missing_tickers)}")

        # Force-include tickers bypass the --tickers-file filter so long as they
        # exist in the underlying market data (i.e. in log_returns columns).
        # Tickers absent from the data store are reported and skipped.
        force_tickers = config.get("force_tickers") or []
        forced_available = [t for t in force_tickers if t in available_tickers]
        forced_missing_store = [t for t in force_tickers if t not in available_tickers]
        re_added = [t for t in forced_available if t not in set(valid_tickers)]

        if forced_missing_store:
            logging.warning(
                f"  ⚠️  Force-include: {len(forced_missing_store)} ticker(s) absent from "
                f"market data, cannot include: {forced_missing_store}"
            )
        if re_added:
            logging.info(
                f"  Force-include: re-adding {len(re_added)} ticker(s) excluded by "
                f"--tickers-file: {re_added}"
            )

        keep = sorted(set(valid_tickers) | set(forced_available))

        if not keep:
            logging.error("FATAL ERROR: No valid tickers after filtering")
            return {"status": "failed", "error": "No valid tickers after filtering"}

        log_returns = log_returns[keep]
        logging.info(
            f"✅ Filtered to {len(keep)} tickers from investable universe"
            + (f" (+{len(re_added)} force-included)" if re_added else "")
        )

    try:
        # Stage 1: Multi-Criteria Screening
        stage1_output = stage1_multi_criteria_screening(log_returns, config)

        # Stage 2: Direct Selection
        stage2_output = stage2_direct_selection(stage1_output, config)

        # Stage 3: Global Optimization
        stage3_results = stage3_global_optimization(stage2_output, config)

        # Stage 4: Analysis & Reporting
        stage4_output = stage4_analysis_and_reporting(
            stage3_results, stage2_output, config
        )

        elapsed = time.time() - start_time

        logging.info("\n" + "=" * 80)
        logging.info(" WORKFLOW COMPLETE - SUCCESS")
        logging.info("=" * 80)
        logging.info(f"\nExecution Summary:")
        logging.info(f"  Total Time:        {elapsed:.1f} seconds")
        logging.info(f"\nPipeline Flow:")
        logging.info(f"  Initial Assets:    {len(log_returns.columns)}")
        logging.info(f"  → Stage 1:         {len(stage1_output['returns'].columns)}")
        logging.info(f"  → Stage 2:         {len(stage2_output['returns'].columns)}")

        if stage3_results.get("max_sharpe", {}).get("status") == "success":
            logging.info(
                f"  → Final Portfolio: {stage3_results['max_sharpe']['n_positions']} positions"
            )

        logging.info(f"\nPortfolios Generated:")
        count = sum(
            1
            for k, v in stage3_results.items()
            if k != "efficient_frontier" and v.get("status") == "success"
        )
        logging.info(
            f"  {count} optimized portfolios + {len(stage3_results.get('efficient_frontier', []))} efficient frontier points"
        )

        logging.info(f"\n{'=' * 80}")
        logging.info(f" COMPLETE LOG SAVED TO:")
        logging.info(f" {LOG_FILE_PATH}")
        logging.info(f"{'=' * 80}\n")

        return {
            "stage1": stage1_output,
            "stage2": stage2_output,
            "stage3": stage3_results,
            "stage4": stage4_output,
            "execution_time": elapsed,
            "status": "success",
        }

    except Exception as e:
        logging.error(f"WORKFLOW FAILED: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simplified 3-stage portfolio optimization workflow (fast, predictable, monotonic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (~2 seconds)
  python %(prog)s --csv data/prices.csv --start 2022-01-01 --end 2025-01-01

  # Custom parameters
  python %(prog)s --csv data/prices.csv --start 2022-01-01 --end 2025-01-01 \\
      --stage1-top-n 120 --stage2-target 50 --min-sharpe 0.6

  # With minimum annual return filter (e.g., 5%% minimum)
  python %(prog)s --csv data/prices.csv --start 2022-01-01 --end 2025-01-01 \\
      --min-annual-return 0.05 --min-sharpe 0.5

  # High growth filter (15%% minimum annual return)
  python %(prog)s --csv data/prices.csv --start 2022-01-01 --end 2025-01-01 \\
      --min-annual-return 0.15

  # Cap max weight per position (10%% max per position for diversification)
  python %(prog)s --csv data/prices.csv --start 2022-01-01 --end 2025-01-01 \\
      --max-weight 0.10

Note: Exhaustive global search removed (10x slower, 0%% improvement)
      Use out-of-sample validation instead for confidence
        """,
    )

    # Data source arguments
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with price data (Date column + asset columns)",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Read data from parquet market data store (no --csv needed). "
        "Uses tickers from data/market_data/ticker_universe.json.",
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

    # Stage 1 parameters
    parser.add_argument(
        "--stage1-top-n",
        type=int,
        default=100,
        help="Stage 1: Number of assets after screening (default: 100)",
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.5,
        help="Stage 1: Minimum Sharpe ratio filter (default: 0.5, stricter)",
    )
    parser.add_argument(
        "--min-annual-return",
        type=float,
        default=None,
        help="Stage 1: Minimum annual return filter (e.g., 0.05 for 5%%, 0.15 for 15%%). Default: None (disabled)",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=500,
        help="Stage 1: Minimum trading days for data quality (default: 500)",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.95,
        help="Stage 1: Maximum correlation for duplicate removal (default: 0.95)",
    )

    # Stage 2 parameters
    parser.add_argument(
        "--stage2-target",
        type=int,
        default=40,
        help="Stage 2: Exact number of assets to select (default: 40)",
    )

    # Stage 3 parameters
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.01,
        help="Risk-free rate for Sharpe calculation (default: 0.01)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Max weight per position (e.g., 0.10 for 10%%, 0.15 for 15%%). Default: None (no cap)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d). Affects annualization: 1d=252, 1wk=52, 1mo=12",
    )

    # Ticker filtering (from filter_investable_universe.py)
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        help="File with one ticker per line (output from filter_investable_universe.py)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=None,
        help="Specific tickers to include (space-separated)",
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

    parser.add_argument(
        "--output-weights",
        type=str,
        default=None,
        help="Export portfolio weights to JSON file (e.g., data/weights.json). "
        "Use --output-weights-type to select which portfolio (default: hrp). "
        "Creates a file compatible with validate_gated_portfolio_oos.py --hrp-weights.",
    )
    parser.add_argument(
        "--output-weights-type",
        type=str,
        default="hrp",
        choices=["max_sharpe", "min_vol", "hrp"],
        help="Portfolio type to export when --output-weights is set. "
        "Choices: max_sharpe, min_vol, hrp (default: hrp).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). Passed to downstream "
        "data loading and optimization functions.",
    )

    # NOTE: Removed parameters (deleted features):
    # --n-clusters, --min-per-cluster, --max-per-cluster (clustering removed)
    # --use-global-search (exhaustive search removed - 0% improvement over SLSQP)

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
    csv_path = args.csv
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
            tickers_map = load_ticker_universe()
            if not tickers_map:
                logging.error("No tickers found in ticker_universe.json")
                return

            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, prefix="portfolio_explore_"
            )
            csv_path = tmp.name
            tmp.close()

            export_df = store.export_portfolio_csv(
                tickers_map,
                args.start,
                args.end,
                csv_path,
                min_coverage=0.8,
            )
            if export_df.empty:
                logging.error(
                    "Parquet store returned no data. "
                    "Run 'python -m algos.common.update_market_data --init' first."
                )
                return
            logging.info(f"Exported {export_df.shape} from parquet store to {csv_path}")
        except ImportError:
            logging.error("MarketDataStore not available. Use --csv instead.")
            return
    elif csv_path is None:
        parser.error("Either --csv or --from-store is required.")

    # Load ticker filter if specified
    ticker_filter = None
    if args.tickers_file:
        if not os.path.exists(args.tickers_file):
            logging.error(f"Tickers file not found: {args.tickers_file}")
            return
        with open(args.tickers_file, "r") as f:
            ticker_filter = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(ticker_filter)} tickers from {args.tickers_file}")
    elif args.tickers:
        ticker_filter = args.tickers
        logging.info(f"Using {len(ticker_filter)} tickers from command line")

    # Build config from arguments
    config = DEFAULT_CONFIG.copy()

    # Get periods_per_year based on interval
    periods_per_year = get_periods_per_year(args.interval)

    # Adjust min_trading_days for interval if user didn't override
    min_trading_days = args.min_trading_days
    if min_trading_days == 500:  # Default value, auto-adjust for interval
        if args.interval == "1wk":
            min_trading_days = 100  # ~2 years of weekly data
        elif args.interval == "1mo":
            min_trading_days = 24  # ~2 years of monthly data

    config.update(
        {
            "stage1_top_n": args.stage1_top_n,
            "min_sharpe": args.min_sharpe,
            "min_annual_return": args.min_annual_return,
            "min_trading_days": min_trading_days,
            "max_correlation": args.max_correlation,
            "stage2_target_n": args.stage2_target,
            "risk_free_rate": args.risk_free_rate,
            "max_weight": args.max_weight,
            "periods_per_year": periods_per_year,
            "ticker_filter": ticker_filter,  # Pass ticker filter to workflow
            "force_tickers": args.force_tickers or [],  # Force-include in Stage 1/2
            "seed": args.seed,
        }
    )

    logging.info(
        f"Using interval '{args.interval}' ({periods_per_year} periods/year, min_trading_days={min_trading_days})"
    )

    # Run workflow
    result = run_complete_workflow(csv_path, args.start, args.end, config)

    if result["status"] == "success":
        logging.info("\n✅ Workflow completed successfully")

        # Export portfolio weights to JSON if requested
        if args.output_weights:
            import json

            # Map CLI choice to internal result key and display label
            type_map = {
                "max_sharpe": ("max_sharpe", "Max Sharpe"),
                "min_vol": ("min_volatility", "Min Volatility"),
                "hrp": ("hrp", "HRP"),
            }
            key, label = type_map[args.output_weights_type]

            port_data = result.get("stage3", {}).get(key, {})
            if port_data.get("status") == "success" and port_data.get("weights"):
                weights = port_data["weights"]
                os.makedirs(os.path.dirname(args.output_weights) or ".", exist_ok=True)
                with open(args.output_weights, "w") as f:
                    json.dump(weights, f, indent=2, sort_keys=True)
                logging.info(
                    f"\n  📦 {label} weights exported to {args.output_weights} "
                    f"({len(weights)} tickers, sum={sum(weights.values()):.4f})"
                )
            else:
                logging.warning(
                    f"  ⚠️ {label} optimization failed or produced no weights -- nothing to export"
                )
    else:
        logging.error(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
