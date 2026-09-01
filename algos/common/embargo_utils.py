"""
Embargo utilities for Lopez de Prado's purged cross-validation.

Implements embargoing (buffer zones) between train and test sets to prevent
temporal leakage in time series backtesting. Based on "Advances in Financial
Machine Learning" by Marcos López de Prado (2018).

References:
- https://en.wikipedia.org/wiki/Purged_cross-validation
- https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/

Author: jcp
Date: 2025-12-02
"""

import pandas as pd
from typing import Tuple


def get_interval_defaults(interval: str = '1d') -> dict:
    """
    Get interval-appropriate defaults for embargo and test size minimums.

    Args:
        interval: Data interval ('1d', '1wk', '1mo', '1h', etc.)

    Returns:
        Dict with 'min_embargo_samples' and 'min_test_samples'

    Notes:
        - Daily: 30 test samples = ~6 weeks, 5 embargo = ~1 week
        - Weekly: 4 test samples = ~1 month, 1 embargo = ~1 week
        - Monthly: 1 test sample = ~1 month, 1 embargo = ~1 month
    """
    interval_defaults = {
        '1d': {'min_embargo_samples': 5, 'min_test_samples': 30},
        '1wk': {'min_embargo_samples': 1, 'min_test_samples': 4},   # 4 weeks = ~1 month
        '1mo': {'min_embargo_samples': 1, 'min_test_samples': 1},   # 1 month minimum
        '1h': {'min_embargo_samples': 24, 'min_test_samples': 168}, # 1 day embargo, 1 week test
    }
    return interval_defaults.get(interval, interval_defaults['1d'])


def calculate_embargo_size(data_length: int,
                          embargo_pct: float = 0.02,
                          min_samples: int = 5,
                          interval: str = '1d') -> int:
    """
    Calculate embargo size (buffer zone samples) for train/test split.

    The embargo period prevents temporal leakage from:
    - Lagged features bleeding into test set
    - Market microstructure delays (information propagates over hours/days)
    - Serial correlation in returns

    Args:
        data_length: Total number of samples in dataset
        embargo_pct: Embargo as percentage of total data (default 2% per Lopez de Prado)
        min_samples: Minimum embargo size in samples (default 5 for daily, auto-adjusted for other intervals)
        interval: Data interval for auto-adjusting min_samples ('1d', '1wk', '1mo')

    Returns:
        Number of samples to use as embargo buffer

    Example:
        >>> calculate_embargo_size(1000, 0.02, 5)
        20  # 2% of 1000

        >>> calculate_embargo_size(100, 0.02, 5)
        5   # Enforces minimum of 5 samples

        >>> calculate_embargo_size(52, 0.02, interval='1wk')
        1   # Weekly data uses min_samples=1

    Notes:
        - For daily stock data: 2% ≈ 5-10 trading days
        - For daily crypto data: 3-5% ≈ 15-25 days (higher volatility)
        - For hourly data: 5-10% ≈ 10-20 hours
        - For weekly data: 2% ≈ 1 week (min 1 sample)
        - For monthly data: 2% ≈ 1 month (min 1 sample)
    """
    # Use interval-aware defaults if min_samples is the default value
    if min_samples == 5 and interval != '1d':
        defaults = get_interval_defaults(interval)
        min_samples = defaults['min_embargo_samples']

    embargo_samples = int(data_length * embargo_pct)
    return max(embargo_samples, min_samples)


def validate_embargo_split(train_size: int,
                          embargo_size: int,
                          test_size: int,
                          min_test_samples: int = 30,
                          interval: str = '1d') -> bool:
    """
    Validate that embargo doesn't reduce test set below minimum viable size.

    Args:
        train_size: Number of samples in training set
        embargo_size: Number of samples in embargo buffer
        test_size: Number of samples in test set (after embargo)
        min_test_samples: Minimum viable test samples (default 30 for daily, auto-adjusted for other intervals)
        interval: Data interval for auto-adjusting min_test_samples ('1d', '1wk', '1mo')

    Returns:
        True if validation passes

    Raises:
        ValueError: If test set is too small after embargo

    Example:
        >>> validate_embargo_split(train_size=500, embargo_size=20, test_size=480, min_test_samples=30)
        True  # Test set (480) is sufficient

        >>> validate_embargo_split(train_size=500, embargo_size=100, test_size=20, min_test_samples=30)
        ValueError: Embargo reduces test set to 20 samples (minimum: 30)

        >>> validate_embargo_split(train_size=26, embargo_size=1, test_size=5, interval='1wk')
        True  # Weekly data uses min_test_samples=4

    Notes:
        - For daily: minimum 30 test samples (~6 weeks)
        - For weekly: minimum 4 test samples (~1 month)
        - For monthly: minimum 1 test sample
        - If validation fails, user should either:
          1. Reduce embargo_pct
          2. Collect more data
          3. Reduce train_split_ratio
    """
    # Use interval-aware defaults if min_test_samples is the default value
    if min_test_samples == 30 and interval != '1d':
        defaults = get_interval_defaults(interval)
        min_test_samples = defaults['min_test_samples']

    if test_size < min_test_samples:
        total_used = train_size + embargo_size + test_size
        raise ValueError(
            f"Embargo reduces test set to {test_size} samples (minimum: {min_test_samples}).\n"
            f"Configuration:\n"
            f"  Train: {train_size} samples\n"
            f"  Embargo: {embargo_size} samples\n"
            f"  Test: {test_size} samples\n"
            f"  Total: {total_used} samples\n"
            f"\n"
            f"Solutions:\n"
            f"  1) Reduce embargo_pct (try --embargo_pct {0.5 * embargo_size / total_used:.3f})\n"
            f"  2) Get more data (need at least {train_size + embargo_size + min_test_samples} samples)\n"
            f"  3) Reduce train_split_ratio (try --train_split {(train_size - 10) / total_used:.2f})"
        )

    # Warn if embargo is unusually large (> 20% of total data)
    total_used = train_size + embargo_size + test_size
    embargo_ratio = embargo_size / total_used

    if embargo_ratio > 0.20:
        print(f"\n⚠️  WARNING: Embargo uses {embargo_ratio:.1%} of data (typically should be < 5%)")
        print(f"   Consider reducing embargo_pct to avoid wasting data")

    return True


def get_embargo_message(embargo_size: int,
                       data_length: int,
                       train_size: int,
                       test_size: int) -> str:
    """
    Generate human-readable embargo summary for logging.

    Args:
        embargo_size: Number of samples in embargo buffer
        data_length: Total number of samples in original dataset
        train_size: Number of samples in training set
        test_size: Number of samples in test set (after embargo)

    Returns:
        Formatted multi-line string describing embargo configuration

    Example:
        >>> msg = get_embargo_message(20, 1000, 500, 480)
        >>> print(msg)
        Embargo Configuration (Lopez de Prado):
          Total Data: 1000 samples
          Train: 500 samples (50.0%)
          Embargo Buffer: 20 samples (2.0%)
          Test: 480 samples (48.0%)
          Rationale: Embargo prevents temporal leakage from:
            - Lagged features bleeding into test set
            - Market microstructure delays
            - Serial correlation in returns
    """
    embargo_pct = 100 * embargo_size / data_length
    train_pct = 100 * train_size / data_length
    test_pct = 100 * test_size / data_length

    return (
        f"\nEmbargo Configuration (Lopez de Prado):\n"
        f"  Total Data: {data_length} samples\n"
        f"  Train: {train_size} samples ({train_pct:.1f}%)\n"
        f"  Embargo Buffer: {embargo_size} samples ({embargo_pct:.1f}%)\n"
        f"  Test: {test_size} samples ({test_pct:.1f}%)\n"
        f"  Rationale: Embargo prevents temporal leakage from:\n"
        f"    - Lagged features bleeding into test set\n"
        f"    - Market microstructure delays\n"
        f"    - Serial correlation in returns\n"
    )


def apply_embargo_to_walk_forward(train_data: pd.DataFrame,
                                 test_data: pd.DataFrame,
                                 embargo_pct: float = 0.02,
                                 interval: str = '1d') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply embargo to walk-forward validation split.

    For walk-forward models (ARIMA, VAR), the initial train/test split already exists.
    This function removes embargo samples from the BEGINNING of the test set to create
    a buffer zone after the training data.

    Args:
        train_data: Training DataFrame (remains unchanged)
        test_data: Test DataFrame (embargo samples removed from start)
        embargo_pct: Embargo percentage of total data (default 2%)
        interval: Data interval for auto-adjusting minimums ('1d', '1wk', '1mo')

    Returns:
        Tuple of (train_data_unchanged, test_data_embargoged)

    Example:
        With 1000 total samples, 50% split, 2% embargo:

        Original:
            train: samples 0-499 (500 samples)
            test:  samples 500-999 (500 samples)

        After embargo:
            train: samples 0-499 (500 samples, unchanged)
            embargo: samples 500-519 (20 samples, discarded)
            test:  samples 520-999 (480 samples, embargoged)

    Notes:
        - Walk-forward models already have natural temporal separation
        - Embargo adds additional protection against leakage
        - Typically use lower embargo_pct for walk-forward (1% vs 2%)
    """
    total_data_length = len(train_data) + len(test_data)
    embargo_size = calculate_embargo_size(
        total_data_length,
        embargo_pct=embargo_pct,
        interval=interval
    )

    # Remove embargo_size samples from START of test set
    test_embargoed = test_data.iloc[embargo_size:].copy()

    # Validate the split
    validate_embargo_split(
        train_size=len(train_data),
        embargo_size=embargo_size,
        test_size=len(test_embargoed),
        interval=interval
    )

    return train_data, test_embargoed
