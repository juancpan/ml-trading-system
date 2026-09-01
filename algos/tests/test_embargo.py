"""
Unit tests for embargo utilities (Lopez de Prado purged cross-validation).

Tests embargo calculation, validation, and application for preventing
temporal leakage in backtesting.

Author: jcp
Date: 2025-12-02
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
    apply_embargo_to_walk_forward
)


class TestCalculateEmbargoSize:
    """Tests for calculate_embargo_size function."""

    def test_normal_case(self):
        """Test embargo calculation for normal dataset."""
        # 2% of 1000 = 20
        assert calculate_embargo_size(1000, 0.02, 5) == 20

    def test_minimum_enforcement(self):
        """Test minimum embargo size is enforced."""
        # 2% of 100 = 2, but minimum is 5
        assert calculate_embargo_size(100, 0.02, 5) == 5

    def test_very_small_dataset(self):
        """Test embargo on very small dataset."""
        # 2% of 10 = 0.2 → rounds to 0, but minimum is 5
        assert calculate_embargo_size(10, 0.02, 5) == 5

    def test_large_dataset(self):
        """Test embargo on large dataset."""
        # 1% of 10000 = 100
        assert calculate_embargo_size(10000, 0.01, 5) == 100

    def test_zero_embargo(self):
        """Test zero embargo percentage."""
        # 0% of 1000 = 0, but minimum is 5
        assert calculate_embargo_size(1000, 0.0, 5) == 5

    def test_high_embargo(self):
        """Test high embargo percentage."""
        # 10% of 1000 = 100
        assert calculate_embargo_size(1000, 0.10, 5) == 100

    def test_custom_minimum(self):
        """Test custom minimum samples."""
        # 2% of 100 = 2, but minimum is 10
        assert calculate_embargo_size(100, 0.02, 10) == 10


class TestValidateEmbargoSplit:
    """Tests for validate_embargo_split function."""

    def test_valid_split(self):
        """Test validation passes for valid configuration."""
        # Train: 500, Embargo: 20, Test: 480 (> 30 minimum)
        assert validate_embargo_split(500, 20, 480, 30) == True

    def test_sufficient_test_samples(self):
        """Test validation passes when test samples exactly at minimum."""
        assert validate_embargo_split(500, 20, 30, 30) == True

    def test_insufficient_test_samples(self):
        """Test validation fails for insufficient test samples."""
        with pytest.raises(ValueError, match="reduces test set to"):
            validate_embargo_split(500, 20, 20, 30)

    def test_very_small_test_set(self):
        """Test validation fails for very small test set."""
        with pytest.raises(ValueError, match="reduces test set to"):
            validate_embargo_split(900, 50, 10, 30)

    def test_zero_test_samples(self):
        """Test validation fails when embargo eliminates all test samples."""
        with pytest.raises(ValueError, match="reduces test set to 0"):
            validate_embargo_split(900, 100, 0, 30)

    def test_large_embargo_warning(self, capsys):
        """Test warning for excessively large embargo."""
        # Total: 1000, embargo: 250 → 25% (> 20% threshold)
        validate_embargo_split(500, 250, 250, 30)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "25.0%" in captured.out

    def test_normal_embargo_no_warning(self, capsys):
        """Test no warning for normal embargo size."""
        # Total: 1000, embargo: 20 → 2% (< 20% threshold)
        validate_embargo_split(500, 20, 480, 30)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


class TestGetEmbargoMessage:
    """Tests for get_embargo_message function."""

    def test_message_format(self):
        """Test embargo message contains all required information."""
        msg = get_embargo_message(20, 1000, 500, 480)

        # Check key components are present
        assert "Embargo Configuration" in msg
        assert "1000 samples" in msg
        assert "500 samples" in msg
        assert "20 samples" in msg
        assert "480 samples" in msg
        assert "Lopez de Prado" in msg

    def test_percentage_calculation(self):
        """Test percentages are calculated correctly."""
        msg = get_embargo_message(20, 1000, 500, 480)

        # Train: 500/1000 = 50%
        assert "50.0%" in msg or "50%" in msg

        # Embargo: 20/1000 = 2%
        assert "2.0%" in msg or "2%" in msg

        # Test: 480/1000 = 48%
        assert "48.0%" in msg or "48%" in msg

    def test_rationale_included(self):
        """Test rationale for embargo is included."""
        msg = get_embargo_message(20, 1000, 500, 480)

        assert "Rationale" in msg
        assert "Lagged features" in msg or "lagged" in msg.lower()
        assert "microstructure" in msg.lower()
        assert "correlation" in msg.lower()


class TestApplyEmbargoToWalkForward:
    """Tests for apply_embargo_to_walk_forward function."""

    def test_basic_application(self):
        """Test embargo is applied correctly to walk-forward split."""
        # Create sample data
        train = pd.DataFrame({'returns': np.random.randn(500)}, index=range(500))
        test = pd.DataFrame({'returns': np.random.randn(500)}, index=range(500, 1000))

        train_result, test_result = apply_embargo_to_walk_forward(
            train, test, embargo_pct=0.02
        )

        # Train should be unchanged
        assert len(train_result) == 500
        assert train_result.index.tolist() == list(range(500))

        # Test should have embargo samples removed (2% of 1000 = 20 samples)
        assert len(test_result) == 480
        assert test_result.index.tolist() == list(range(520, 1000))

    def test_minimum_embargo_enforcement(self):
        """Test minimum embargo size is enforced."""
        # Small dataset: 100 total samples
        train = pd.DataFrame({'returns': np.random.randn(50)}, index=range(50))
        test = pd.DataFrame({'returns': np.random.randn(50)}, index=range(50, 100))

        train_result, test_result = apply_embargo_to_walk_forward(
            train, test, embargo_pct=0.02  # 2% of 100 = 2, but minimum is 5
        )

        # Minimum 5 samples enforced
        assert len(test_result) == 45  # 50 - 5
        assert test_result.index.tolist() == list(range(55, 100))

    def test_higher_embargo(self):
        """Test higher embargo percentage."""
        train = pd.DataFrame({'returns': np.random.randn(500)})
        test = pd.DataFrame({'returns': np.random.randn(500)})

        train_result, test_result = apply_embargo_to_walk_forward(
            train, test, embargo_pct=0.05  # 5% of 1000 = 50 samples
        )

        assert len(train_result) == 500
        assert len(test_result) == 450  # 500 - 50

    def test_validation_error_insufficient_test(self):
        """Test error when embargo makes test set too small."""
        train = pd.DataFrame({'returns': np.random.randn(950)})
        test = pd.DataFrame({'returns': np.random.randn(50)})

        # 2% of 1000 = 20 embargo → test becomes 30, exactly at limit
        train_result, test_result = apply_embargo_to_walk_forward(
            train, test, embargo_pct=0.02
        )
        assert len(test_result) == 30

        # 3% of 1000 = 30 embargo → test becomes 20, below limit
        with pytest.raises(ValueError, match="reduces test set to"):
            apply_embargo_to_walk_forward(train, test, embargo_pct=0.03)

    def test_preserves_dataframe_structure(self):
        """Test that DataFrame columns and index are preserved."""
        train = pd.DataFrame({
            'returns': np.random.randn(500),
            'price': np.random.randn(500),
            'direction': np.random.choice([-1, 1], 500)
        })
        test = pd.DataFrame({
            'returns': np.random.randn(500),
            'price': np.random.randn(500),
            'direction': np.random.choice([-1, 1], 500)
        })

        train_result, test_result = apply_embargo_to_walk_forward(
            train, test, embargo_pct=0.02
        )

        # Check columns are preserved
        assert list(train_result.columns) == ['returns', 'price', 'direction']
        assert list(test_result.columns) == ['returns', 'price', 'direction']


class TestIntegrationScenarios:
    """Integration tests for realistic backtest scenarios."""

    def test_typical_stock_backtest(self):
        """Test typical stock backtest configuration."""
        # 5 years daily data ≈ 1260 samples
        data_length = 1260
        train_samples = int(data_length * 0.5)  # 50% split = 630
        embargo_size = calculate_embargo_size(data_length, 0.02, 5)  # 2% = 25

        # Expected: Train 630, Embargo 25, Test 605
        test_samples = data_length - train_samples - embargo_size
        assert embargo_size == 25
        assert test_samples == 605

        # Validation should pass
        assert validate_embargo_split(train_samples, embargo_size, test_samples, 30)

    def test_crypto_backtest_high_embargo(self):
        """Test crypto backtest with higher embargo."""
        # Crypto: 3 years daily ≈ 1095 samples, 5% embargo
        data_length = 1095
        train_samples = int(data_length * 0.5)  # 547 samples
        embargo_size = calculate_embargo_size(data_length, 0.05, 5)  # 5% = 54

        test_samples = data_length - train_samples - embargo_size
        assert embargo_size == 54
        assert test_samples == 494

        assert validate_embargo_split(train_samples, embargo_size, test_samples, 30)

    def test_small_dataset_edge_case(self):
        """Test small dataset where embargo might cause issues."""
        # Only 200 samples total
        data_length = 200
        train_samples = int(data_length * 0.5)  # 100
        embargo_size = calculate_embargo_size(data_length, 0.02, 5)  # 2% = 4, min 5
        test_samples = data_length - train_samples - embargo_size

        assert embargo_size == 5  # Minimum enforced
        assert test_samples == 95

        # Should pass validation
        assert validate_embargo_split(train_samples, embargo_size, test_samples, 30)

    def test_insufficient_data_scenario(self):
        """Test scenario where total data is insufficient after embargo."""
        # Only 100 samples total, 50% split, 2% embargo
        data_length = 100
        train_samples = 50
        embargo_size = calculate_embargo_size(data_length, 0.02, 5)  # Minimum 5
        test_samples = data_length - train_samples - embargo_size

        # Test: 100 - 50 - 5 = 45 samples (should pass)
        assert test_samples == 45
        assert validate_embargo_split(train_samples, embargo_size, test_samples, 30)

        # But with higher split (80%), test becomes too small
        train_samples_high = 80
        test_samples_small = data_length - train_samples_high - embargo_size  # 15 samples

        with pytest.raises(ValueError):
            validate_embargo_split(train_samples_high, embargo_size, test_samples_small, 30)


if __name__ == "__main__":
    # Run tests
    print("Running embargo utilities unit tests...")
    print("=" * 80)

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
