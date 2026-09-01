"""Tests for minimum_track_record_length (Phase 2.2)."""

from __future__ import annotations

import math

import pytest

from algos.wfov.statistical_tests import minimum_track_record_length


class TestMinTRL:
    def test_observed_below_benchmark_is_infinite(self):
        r = minimum_track_record_length(0.5, benchmark_sharpe=1.0)
        assert r["min_trl_observations"] == math.inf
        assert "undefined" in r["interpretation"].lower()

    def test_normal_distribution_gives_finite_result(self):
        # Sharpe 1.5, benchmark 0, normal => ~few hundred days at 95%
        r = minimum_track_record_length(1.5, benchmark_sharpe=0.0,
                                         skewness=0.0, kurtosis=3.0)
        assert math.isfinite(r["min_trl_observations"])
        assert 1 < r["min_trl_months"] < 36

    def test_fat_tails_increase_min_trl(self):
        baseline = minimum_track_record_length(1.5, kurtosis=3.0)
        fat = minimum_track_record_length(1.5, kurtosis=10.0)
        assert fat["min_trl_observations"] > baseline["min_trl_observations"]

    def test_negative_skew_increases_min_trl(self):
        # Negative skew is risky and requires longer track record.
        baseline = minimum_track_record_length(1.5, skewness=0.0)
        skewed = minimum_track_record_length(1.5, skewness=-1.0)
        assert skewed["min_trl_observations"] > baseline["min_trl_observations"]

    def test_higher_observed_sharpe_reduces_min_trl(self):
        # Easier to distinguish a SR=2 from 0 than SR=1 from 0.
        low = minimum_track_record_length(1.0)
        high = minimum_track_record_length(2.0)
        assert high["min_trl_observations"] < low["min_trl_observations"]

    def test_confidence_affects_result(self):
        c90 = minimum_track_record_length(1.5, confidence=0.90)
        c99 = minimum_track_record_length(1.5, confidence=0.99)
        # Higher confidence => need more data.
        assert c99["min_trl_observations"] > c90["min_trl_observations"]

    def test_returns_required_keys(self):
        r = minimum_track_record_length(1.5)
        for k in ["min_trl_observations", "min_trl_years", "min_trl_months",
                  "confidence", "interpretation"]:
            assert k in r
