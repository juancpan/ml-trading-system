"""
Simple unit tests for embargo utilities (no pytest required).

Tests embargo calculation, validation, and application.

Author: jcp
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
    apply_embargo_to_walk_forward
)


def test_calculate_embargo_size():
    """Test embargo size calculation."""
    print("\nTest 1: calculate_embargo_size()")
    print("-" * 60)

    # Normal case
    assert calculate_embargo_size(1000, 0.02, 5) == 20, "Failed: normal case"
    print("✓ Normal case: 2% of 1000 = 20 samples")

    # Minimum enforcement
    assert calculate_embargo_size(100, 0.02, 5) == 5, "Failed: minimum enforcement"
    print("✓ Minimum enforcement: 2% of 100 = 2 → enforced to 5")

    # Large dataset
    assert calculate_embargo_size(10000, 0.01, 5) == 100, "Failed: large dataset"
    print("✓ Large dataset: 1% of 10000 = 100 samples")

    # High embargo
    assert calculate_embargo_size(1000, 0.10, 5) == 100, "Failed: high embargo"
    print("✓ High embargo: 10% of 1000 = 100 samples")

    print("✅ All calculate_embargo_size tests passed!")


def test_validate_embargo_split():
    """Test embargo split validation."""
    print("\nTest 2: validate_embargo_split()")
    print("-" * 60)

    # Valid split
    assert validate_embargo_split(500, 20, 480, 30) == True, "Failed: valid split"
    print("✓ Valid split: 500 train, 20 embargo, 480 test (> 30 min)")

    # Exactly at minimum
    assert validate_embargo_split(500, 20, 30, 30) == True, "Failed: exactly at minimum"
    print("✓ Edge case: test exactly at minimum (30 samples)")

    # Insufficient test samples
    try:
        validate_embargo_split(500, 20, 20, 30)
        print("✗ Should have raised ValueError for insufficient test samples")
        sys.exit(1)
    except ValueError as e:
        assert "reduces test set to 20" in str(e)
        print("✓ Correctly rejected: test below minimum (20 < 30)")

    print("✅ All validate_embargo_split tests passed!")


def test_get_embargo_message():
    """Test embargo message generation."""
    print("\nTest 3: get_embargo_message()")
    print("-" * 60)

    msg = get_embargo_message(20, 1000, 500, 480)

    # Check key information is present
    assert "Embargo Configuration" in msg
    assert "1000 samples" in msg
    assert "500 samples" in msg
    assert "20 samples" in msg
    assert "480 samples" in msg
    assert "Lopez de Prado" in msg

    print("✓ Message contains all required information")
    print("✓ Sample message:\n")
    print(msg)

    print("✅ All get_embargo_message tests passed!")


def test_apply_embargo_to_walk_forward():
    """Test embargo application for walk-forward models."""
    print("\nTest 4: apply_embargo_to_walk_forward()")
    print("-" * 60)

    # Create sample data
    train = pd.DataFrame({'returns': np.random.randn(500)}, index=range(500))
    test = pd.DataFrame({'returns': np.random.randn(500)}, index=range(500, 1000))

    train_result, test_result = apply_embargo_to_walk_forward(
        train, test, embargo_pct=0.02
    )

    # Train should be unchanged
    assert len(train_result) == 500, "Train should be unchanged"
    assert train_result.index.tolist() == list(range(500)), "Train index should be unchanged"
    print("✓ Train data unchanged (500 samples)")

    # Test should have embargo removed (2% of 1000 = 20 samples)
    assert len(test_result) == 480, f"Test should be 480, got {len(test_result)}"
    assert test_result.index.tolist() == list(range(520, 1000)), "Test index should skip embargo"
    print("✓ Test data embargoged correctly (480 samples, skipped 20)")

    # Test with higher embargo
    train2 = pd.DataFrame({'returns': np.random.randn(500)})
    test2 = pd.DataFrame({'returns': np.random.randn(500)})

    train_result2, test_result2 = apply_embargo_to_walk_forward(
        train2, test2, embargo_pct=0.05  # 5% of 1000 = 50
    )

    assert len(test_result2) == 450, "Test should be 450 with 5% embargo"
    print("✓ Higher embargo (5%): test = 450 samples")

    print("✅ All apply_embargo_to_walk_forward tests passed!")


def test_integration_scenario():
    """Test realistic backtest scenario."""
    print("\nTest 5: Integration Scenario (Realistic Backtest)")
    print("-" * 60)

    # Simulate 5 years of daily stock data
    data_length = 1260  # ~5 years trading days
    train_samples = int(data_length * 0.5)  # 630
    embargo_size = calculate_embargo_size(data_length, 0.02, 5)  # 25 samples
    test_samples = data_length - train_samples - embargo_size  # 605

    print(f"5-year stock backtest (1260 samples):")
    print(f"  Train: {train_samples} samples (50%)")
    print(f"  Embargo: {embargo_size} samples (2%)")
    print(f"  Test: {test_samples} samples (48%)")

    # Validate
    assert validate_embargo_split(train_samples, embargo_size, test_samples, 30)

    # Calculate what this means in trading days
    embargo_weeks = embargo_size / 5
    print(f"  Embargo duration: ~{embargo_weeks:.1f} trading weeks")

    print("✅ Integration scenario passed!")


if __name__ == "__main__":
    print("=" * 80)
    print("EMBARGO UTILITIES UNIT TESTS")
    print("=" * 80)

    try:
        test_calculate_embargo_size()
        test_validate_embargo_split()
        test_get_embargo_message()
        test_apply_embargo_to_walk_forward()
        test_integration_scenario()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
