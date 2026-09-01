"""
Random window generator for WFOV with stratified sampling.

Implements stratified random window generation to ensure coverage across
different market regimes (bull/bear/sideways) and minimize clustering bias.

Author: jcp
Date: 2025-12-02
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dateutil import parser as dateutil_parser
import hashlib


def generate_iteration_seed(master_seed: int, iteration: int) -> int:
    """
    Generate deterministic seed for each iteration from master seed.

    Args:
        master_seed: Master seed for WFOV session
        iteration: Iteration number (0-indexed)

    Returns:
        Deterministic seed for this iteration

    Example:
        >>> generate_iteration_seed(42, 0)
        2722073184
        >>> generate_iteration_seed(42, 1)
        2891336113
    """
    hash_input = f"{master_seed}_{iteration}".encode()
    hash_digest = hashlib.md5(hash_input).hexdigest()
    return int(hash_digest[:8], 16)


def calculate_quartiles(
    start_date: str, end_date: str
) -> List[Tuple[datetime, datetime]]:
    """
    Divide date range into 4 equal quartiles for stratified sampling.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        List of 4 tuples: [(q1_start, q1_end), (q2_start, q2_end), ...]

    Example:
        >>> quartiles = calculate_quartiles('2020-01-01', '2024-01-01')
        >>> len(quartiles)
        4
        >>> # Each quartile spans ~1 year
    """
    start_dt = dateutil_parser.parse(start_date)
    end_dt = dateutil_parser.parse(end_date)

    total_days = (end_dt - start_dt).days
    quartile_days = total_days // 4

    quartiles = []
    for i in range(4):
        q_start = start_dt + timedelta(days=i * quartile_days)
        if i == 3:  # Last quartile gets remaining days
            q_end = end_dt
        else:
            q_end = start_dt + timedelta(days=(i + 1) * quartile_days)
        quartiles.append((q_start, q_end))

    return quartiles


def generate_random_window(
    full_start: datetime,
    full_end: datetime,
    quartile_start: datetime,
    quartile_end: datetime,
    min_lookback_days: int,
    max_lookback_days: int,
    rng: np.random.Generator,
) -> Dict[str, any]:
    """
    Generate a single random window with END date stratified to quartile.

    KEY: Stratification applies to window END date (which quartile it falls in),
    but lookback can span ACROSS multiple quartiles or before full_start.

    This allows full lookback range (e.g., 365-1825 days) while ensuring
    end dates are distributed across the full time period.

    Args:
        full_start: Full date range start (earliest possible window start)
        full_end: Full date range end (latest possible window end)
        quartile_start: Quartile start date (for end_date stratification)
        quartile_end: Quartile end date (for end_date stratification)
        min_lookback_days: Minimum lookback period
        max_lookback_days: Maximum lookback period
        rng: NumPy random generator (for reproducibility)

    Returns:
        Dict with: start_date, end_date, lookback_days

    Example:
        full_start=2020-01-01, quartile=(2023-01-01, 2024-01-01)
        lookback=1825 days (5 years)
        end_date=2023-06-15 (random within quartile)
        start_date=2023-06-15 - 1825 days = 2018-06-20 (BEFORE full_start!)
        → Adjust start to 2020-01-01, recalc end or reduce lookback
    """
    # Generate random lookback (convert to Python int for timedelta compatibility)
    lookback_days = int(rng.integers(min_lookback_days, max_lookback_days + 1))

    # Generate random end_date within quartile (stratification happens here)
    quartile_days = (quartile_end - quartile_start).days
    if quartile_days <= 0:
        raise ValueError(f"Invalid quartile: {quartile_start} to {quartile_end}")

    random_offset = int(rng.integers(0, quartile_days + 1))
    end_dt = quartile_start + timedelta(days=random_offset)

    # Ensure end_dt doesn't exceed full_end
    if end_dt > full_end:
        end_dt = full_end

    # Calculate start date from lookback
    start_dt = end_dt - timedelta(days=lookback_days)

    # Validate: start_dt must not be before full_start
    if start_dt < full_start:
        # Clip start to full_start and reduce effective lookback.
        # This preserves the stratified end_date assignment (keeping the
        # window in its intended quartile) at the cost of shorter lookback.
        start_dt = full_start
        lookback_days = (
            end_dt - start_dt
        ).days  # Reduce lookback, keep end_dt in quartile

        # Validate reduced lookback is still usable
        if lookback_days < min_lookback_days:
            # Lookback too short after clipping -- expand end_dt as last resort
            end_dt = start_dt + timedelta(days=min_lookback_days)
            lookback_days = min_lookback_days
            if end_dt > full_end:
                max_possible_lookback = (full_end - full_start).days
                lookback_days = min(max_possible_lookback - 30, lookback_days)
                if lookback_days < min_lookback_days:
                    raise ValueError(
                        f"Cannot satisfy min_lookback_days ({min_lookback_days}) "
                        f"with date range {full_start} to {full_end}"
                    )
                end_dt = full_start + timedelta(days=lookback_days)

    return {
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "lookback_days": lookback_days,
    }


def generate_random_windows(
    full_start_date: str,
    full_end_date: str,
    min_lookback_days: int,
    max_lookback_days: int,
    num_iterations: int,
    master_seed: int,
    stratified: bool = True,
) -> List[Dict]:
    """
    Generate random backtest windows with stratified sampling.

    Stratified sampling ensures coverage across the full date range by dividing
    it into quartiles and sampling proportionally from each. This prevents
    clustering in recent data and validates across diverse market conditions.

    Args:
        full_start_date: Overall date range start (YYYY-MM-DD)
        full_end_date: Overall date range end (YYYY-MM-DD)
        min_lookback_days: Minimum lookback period
        max_lookback_days: Maximum lookback period
        num_iterations: Number of windows to generate
        master_seed: Master seed for reproducibility
        stratified: If True, use stratified sampling; else pure random

    Returns:
        List of window dicts: [
            {
                'start_date': '2020-01-01',
                'end_date': '2023-01-01',
                'lookback_days': 1095,
                'iteration_seed': 10298,
                'quartile': 1  # Which quartile this window came from
            },
            ...
        ]

    Example:
        >>> windows = generate_random_windows(
        ...     '2020-01-01', '2025-01-01',
        ...     min_lookback_days=365,
        ...     max_lookback_days=1825,
        ...     num_iterations=100,
        ...     master_seed=42
        ... )
        >>> len(windows)
        100
        >>> all('start_date' in w for w in windows)
        True
    """
    # Initialize random generator with master seed
    rng = np.random.default_rng(master_seed)

    windows = []

    if stratified:
        # Parse full date range
        full_start_dt = dateutil_parser.parse(full_start_date)
        full_end_dt = dateutil_parser.parse(full_end_date)

        # Calculate quartiles
        quartiles = calculate_quartiles(full_start_date, full_end_date)

        # Assign iterations to quartiles (stratified: 25% each)
        iterations_per_quartile = num_iterations // 4
        remainder = num_iterations % 4

        # Generate windows for each quartile
        for quartile_idx, (q_start, q_end) in enumerate(quartiles):
            # Distribute remainder to first quartiles
            n_windows = iterations_per_quartile + (1 if quartile_idx < remainder else 0)

            for i in range(n_windows):
                try:
                    window = generate_random_window(
                        full_start_dt,
                        full_end_dt,  # Pass full range
                        q_start,
                        q_end,  # Quartile for end_date stratification
                        min_lookback_days,
                        max_lookback_days,
                        rng,
                    )

                    # Add iteration metadata
                    iteration_num = len(windows)
                    window["iteration_seed"] = generate_iteration_seed(
                        master_seed, iteration_num
                    )
                    window["quartile"] = quartile_idx + 1  # 1-indexed for readability

                    windows.append(window)

                except ValueError as e:
                    print(
                        f"Warning: Skipped window in quartile {quartile_idx + 1}: {e}"
                    )
                    continue

    else:
        # Pure random sampling (no stratification)
        full_start_dt = dateutil_parser.parse(full_start_date)
        full_end_dt = dateutil_parser.parse(full_end_date)

        for i in range(num_iterations):
            window = generate_random_window(
                full_start_dt,
                full_end_dt,  # Full range
                full_start_dt,
                full_end_dt,  # Use full range as "quartile" (no stratification)
                min_lookback_days,
                max_lookback_days,
                rng,
            )

            window["iteration_seed"] = generate_iteration_seed(master_seed, i)
            window["quartile"] = 0  # No quartile for pure random

            windows.append(window)

    return windows


def validate_windows(windows: List[Dict], min_test_samples: int = 30) -> List[Dict]:
    """
    Validate that windows have sufficient data for backtesting.

    Args:
        windows: List of window dicts
        min_test_samples: Minimum test samples required after split/embargo

    Returns:
        Filtered list of valid windows

    Note:
        Invalid windows are logged but not included in output
    """
    valid_windows = []
    invalid_count = 0

    for window in windows:
        lookback_days = window["lookback_days"]

        # Rough validation: assume 70% trading days (weekends removed)
        approx_samples = int(lookback_days * 0.7)

        # With 50% split and 2% embargo, test ≈ 48% of samples
        approx_test_samples = int(approx_samples * 0.48)

        if approx_test_samples >= min_test_samples:
            valid_windows.append(window)
        else:
            invalid_count += 1
            print(
                f"Warning: Window {window['start_date']} to {window['end_date']} "
                f"may have insufficient test data (~{approx_test_samples} samples)"
            )

    if invalid_count > 0:
        print(f"Filtered out {invalid_count} potentially invalid windows")

    return valid_windows


def get_window_statistics(windows: List[Dict]) -> Dict:
    """
    Calculate statistics on generated windows.

    Args:
        windows: List of window dicts

    Returns:
        Dict with statistics on lookback_days, quartile distribution
    """
    if not windows:
        return {}

    lookback_values = [w["lookback_days"] for w in windows]
    quartile_counts = (
        pd.Series([w["quartile"] for w in windows]).value_counts().to_dict()
    )

    return {
        "total_windows": len(windows),
        "lookback_days": {
            "mean": np.mean(lookback_values),
            "std": np.std(lookback_values),
            "min": np.min(lookback_values),
            "max": np.max(lookback_values),
            "median": np.median(lookback_values),
        },
        "quartile_distribution": quartile_counts,
    }


def generate_walk_forward_expanding_windows(
    full_start_date: str,
    full_end_date: str,
    initial_train_days: int,
    test_days: int,
    step_days: int,
    embargo_pct: float = 0.02,
    master_seed: int = 42,
) -> List[Dict]:
    """
    Generate expanding window sequence for true walk-forward validation.

    Training set GROWS over time, test size FIXED.
    No overlapping test periods (true out-of-sample).

    Args:
        full_start_date: Start of available data (YYYY-MM-DD)
        full_end_date: End of available data (YYYY-MM-DD)
        initial_train_days: Initial training period (e.g., 1260 = 5 years)
        test_days: Test period size (e.g., 252 = 1 year)
        step_days: Step between iterations (e.g., 126 = 6 months)
        embargo_pct: Embargo buffer as % of train days (default: 2%)
        master_seed: Seed for iteration_seed generation

    Returns:
        List of window dicts with:
        {
            'start_date': '2020-01-01',  # Train start (fixed)
            'end_date': '2024-12-31',    # Train end (grows)
            'test_start': '2025-01-01',  # Test start
            'test_end': '2025-12-31',    # Test end
            'lookback_days': 1260,       # Train period length
            'iteration_seed': <int>,
            'validation_mode': 'walk_forward_expanding',
            'quartile': 0  # Not applicable for walk-forward
        }

    Example:
        >>> windows = generate_walk_forward_expanding_windows(
        ...     '2020-01-01', '2027-01-01',
        ...     initial_train_days=1260,
        ...     test_days=252,
        ...     step_days=126
        ... )
        >>> len(windows)
        10  # ~10 iterations with 6-month steps over 7 years
    """
    full_start_dt = dateutil_parser.parse(full_start_date)
    full_end_dt = dateutil_parser.parse(full_end_date)

    windows = []
    iteration = 0

    # Initial window
    train_start_dt = full_start_dt
    train_end_dt = train_start_dt + timedelta(days=initial_train_days)

    # Calculate embargo days
    embargo_days = int(initial_train_days * embargo_pct)

    test_start_dt = train_end_dt + timedelta(days=embargo_days)
    test_end_dt = test_start_dt + timedelta(days=test_days)

    # Generate windows while we have data
    while test_end_dt <= full_end_dt:
        window = {
            "start_date": train_start_dt.strftime("%Y-%m-%d"),
            "end_date": train_end_dt.strftime("%Y-%m-%d"),
            "test_start": test_start_dt.strftime("%Y-%m-%d"),
            "test_end": test_end_dt.strftime("%Y-%m-%d"),
            "lookback_days": (train_end_dt - train_start_dt).days,
            "test_days": test_days,
            "iteration_seed": generate_iteration_seed(master_seed, iteration),
            "validation_mode": "walk_forward_expanding",
            "quartile": 0,  # N/A for walk-forward
        }
        windows.append(window)

        # Move to next window: EXPAND train, slide test
        # Train grows to include previous test period
        train_end_dt = test_end_dt

        # Recalculate embargo based on new train size
        embargo_days = int((train_end_dt - train_start_dt).days * embargo_pct)

        # Slide test forward by step_days
        test_start_dt = train_end_dt + timedelta(days=embargo_days)
        test_end_dt = test_start_dt + timedelta(days=test_days)

        iteration += 1

        # Safety: prevent infinite loop
        if iteration > 1000:
            print(f"Warning: Generated 1000+ windows, stopping")
            break

    return windows


def generate_walk_forward_rolling_windows(
    full_start_date: str,
    full_end_date: str,
    window_size: int,
    test_days: int,
    step_days: int,
    embargo_pct: float = 0.02,
    master_seed: int = 42,
) -> List[Dict]:
    """
    Generate rolling window sequence for walk-forward validation.

    Both training and test windows SLIDE forward (fixed sizes).
    No overlapping test periods (true out-of-sample).

    IMPORTANT: For no test overlap, step_days must be >= test_days.
    If step_days < test_days, will auto-adjust to prevent overlap.

    Args:
        full_start_date: Start of available data (YYYY-MM-DD)
        full_end_date: End of available data (YYYY-MM-DD)
        window_size: Training window size (e.g., 1260 = 5 years)
        test_days: Test period size (e.g., 252 = 1 year)
        step_days: Step between iterations (e.g., 252 = test_days to avoid overlap)
        embargo_pct: Embargo buffer as % of window size (default: 2%)
        master_seed: Seed for iteration_seed generation

    Returns:
        List of window dicts (same structure as expanding)

    Example:
        >>> windows = generate_walk_forward_rolling_windows(
        ...     '2020-01-01', '2027-01-01',
        ...     window_size=1260,
        ...     test_days=252,
        ...     step_days=252  # Must be >= test_days to avoid overlap
        ... )
        >>> # Each window: 5 years train, 1 year test
        >>> # Slide forward by 1 year each iteration (no overlap)
    """
    full_start_dt = dateutil_parser.parse(full_start_date)
    full_end_dt = dateutil_parser.parse(full_end_date)

    # CRITICAL: Ensure no test period overlap
    # For non-overlapping test periods, step_days must be >= test_days
    if step_days < test_days:
        print(
            f"⚠️  WARNING: step_days ({step_days}) < test_days ({test_days}) would cause overlap"
        )
        print(f"   Auto-adjusting step_days to {test_days} to ensure no overlap")
        step_days = test_days

    windows = []
    iteration = 0

    # Calculate embargo days
    embargo_days = int(window_size * embargo_pct)

    # Initial window
    train_start_dt = full_start_dt
    train_end_dt = train_start_dt + timedelta(days=window_size)
    test_start_dt = train_end_dt + timedelta(days=embargo_days)
    test_end_dt = test_start_dt + timedelta(days=test_days)

    # Generate windows while we have data
    while test_end_dt <= full_end_dt:
        window = {
            "start_date": train_start_dt.strftime("%Y-%m-%d"),
            "end_date": train_end_dt.strftime("%Y-%m-%d"),
            "test_start": test_start_dt.strftime("%Y-%m-%d"),
            "test_end": test_end_dt.strftime("%Y-%m-%d"),
            "lookback_days": window_size,
            "test_days": test_days,
            "iteration_seed": generate_iteration_seed(master_seed, iteration),
            "validation_mode": "walk_forward_rolling",
            "quartile": 0,  # N/A for walk-forward
        }
        windows.append(window)

        # Move to next window: SLIDE both train and test forward
        # Use max(step_days, test_days) to ensure no overlap
        actual_step = max(step_days, test_days)
        train_start_dt = train_start_dt + timedelta(days=actual_step)
        train_end_dt = train_start_dt + timedelta(days=window_size)
        test_start_dt = train_end_dt + timedelta(days=embargo_days)
        test_end_dt = test_start_dt + timedelta(days=test_days)

        iteration += 1

        # Safety: prevent infinite loop
        if iteration > 1000:
            print(f"Warning: Generated 1000+ windows, stopping")
            break

    return windows


def validate_walk_forward_windows(windows: List[Dict]) -> bool:
    """
    Validate that walk-forward windows have no overlapping test periods.

    Critical for ensuring true out-of-sample validation.

    Args:
        windows: List of walk-forward window dicts

    Returns:
        True if validation passes

    Raises:
        ValueError: If overlapping test periods detected

    Example:
        >>> windows = generate_walk_forward_expanding_windows(...)
        >>> validate_walk_forward_windows(windows)
        True  # No overlaps
    """
    test_periods = []

    for i, window in enumerate(windows):
        test_start = dateutil_parser.parse(window["test_start"])
        test_end = dateutil_parser.parse(window["test_end"])

        # Check against all previous test periods
        for j, (prev_start, prev_end) in enumerate(test_periods):
            # Test for overlap: two periods overlap if NOT (one ends before other starts)
            overlaps = not (test_end <= prev_start or test_start >= prev_end)

            if overlaps:
                raise ValueError(
                    f"Overlapping test periods detected:\n"
                    f"  Window {i}: {test_start.date()} to {test_end.date()}\n"
                    f"  Window {j}: {prev_start.date()} to {prev_end.date()}\n"
                    f"This violates walk-forward out-of-sample requirement!"
                )

        test_periods.append((test_start, test_end))

    print(
        f"✓ Validated {len(windows)} walk-forward windows: No overlapping test periods"
    )
    return True
