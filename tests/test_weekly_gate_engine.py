import pytest
import numpy as np
from algos.backtest_code.weekly_gate_engine import apply_gates


class TestApplyGates:
    def test_no_gates_closed(self):
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        signals = {"A": 1, "B": 1, "C": 1}
        result = apply_gates(weights, signals)
        assert result == pytest.approx(weights, abs=1e-9)

    def test_one_gate_closed(self):
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        signals = {"A": 1, "B": 1, "C": -1}
        result = apply_gates(weights, signals)
        assert result["A"] == pytest.approx(0.5, abs=1e-9)
        assert result["B"] == pytest.approx(0.5, abs=1e-9)
        assert result["C"] == pytest.approx(0.0, abs=1e-9)

    def test_weights_sum_to_one(self):
        weights = {"A": 0.15, "B": 0.12, "C": 0.10, "D": 0.18, "E": 0.20, "F": 0.25}
        signals = {"A": 1, "B": -1, "C": 1, "D": -1, "E": 1, "F": -1}
        result = apply_gates(weights, signals)
        active_sum = sum(v for v in result.values() if v > 0)
        assert active_sum == pytest.approx(1.0, abs=1e-9)
        assert result["B"] == 0.0
        assert result["D"] == 0.0
        assert result["F"] == 0.0

    def test_zero_signal_treated_as_gate_closed(self):
        weights = {"A": 0.5, "B": 0.5}
        signals = {"A": 1, "B": 0}
        result = apply_gates(weights, signals)
        assert result["A"] == pytest.approx(1.0, abs=1e-9)
        assert result["B"] == pytest.approx(0.0, abs=1e-9)

    def test_all_gates_closed_returns_all_zero(self):
        weights = {"A": 0.5, "B": 0.5}
        signals = {"A": -1, "B": -1}
        result = apply_gates(weights, signals)
        assert result["A"] == pytest.approx(0.0, abs=1e-9)
        assert result["B"] == pytest.approx(0.0, abs=1e-9)

    def test_missing_signal_defaults_to_gate_open(self):
        weights = {"A": 0.6, "B": 0.4}
        signals = {"A": 1}
        result = apply_gates(weights, signals)
        assert result["A"] == pytest.approx(0.6, abs=1e-9)
        assert result["B"] == pytest.approx(0.4, abs=1e-9)

    def test_concentration_cap_applied(self):
        weights = {"A": 0.10, "B": 0.40, "C": 0.50}
        signals = {"A": -1, "B": 1, "C": 1}
        result = apply_gates(weights, signals, max_weight=0.50)
        assert result["B"] <= 0.50 + 1e-9
        assert result["C"] <= 0.50 + 1e-9

    def test_min_active_tickers_floor(self):
        weights = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        signals = {"A": 1, "B": -1, "C": -1, "D": -1, "E": -1}
        result = apply_gates(weights, signals, min_active_tickers=3)
        assert result["A"] == pytest.approx(0.2, abs=1e-9)
        assert sum(result.values()) == pytest.approx(0.2, abs=1e-9)
