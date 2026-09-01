"""Test that carry-trade signals are logged to signal_history."""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IBKR_DIR = PROJECT_ROOT / "execution"
sys.path.insert(0, str(IBKR_DIR))

from cash_portfolio_manager import CashPortfolioManager


import pytest


@pytest.fixture(autouse=True)
def _ensure_carry_model_file():
    """Create a dummy deployed-model file for the duration of each test.

    _get_carry_signal checks that the configured model path exists before
    generating a signal. In a real deployment a trained pickled GNB model
    lives at strategy_models/carry_USDJPY_model_gnb.pkl; trained weights are
    not shipped with this repo, so these tests create an empty stand-in file.
    Signal generation itself is mocked - only file existence/openability
    matters here.
    """
    model_dir = IBKR_DIR / "strategy_models"
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / "carry_USDJPY_model_gnb.pkl"
    model_file.touch(exist_ok=True)
    yield
    model_file.unlink(missing_ok=True)


class DummyIB:
    def __init__(self, currency_balances):
        self.currency_balances = currency_balances


def test_carry_signal_logs_to_signal_history(monkeypatch):
    """When _get_carry_signal runs via strategy_executor, it must call log_signal."""
    logger = logging.getLogger("test_carry_signal_logging")
    manager = CashPortfolioManager(
        DummyIB({"USD": -5000.0, "JPY": 0.0}),
        logger,
        dry_run=True,
    )

    logged_calls = []

    def fake_log_signal(**kwargs):
        logged_calls.append(kwargs)

    monkeypatch.setattr(
        "cash_portfolio_manager._log_signal_history",
        fake_log_signal,
    )

    fake_executor = MagicMock()
    fake_executor.generate_carry_signal.return_value = 0.7
    manager.strategy_executor = fake_executor

    signal = manager._get_carry_signal("USDJPY", {
        "model_type": "gnb",
        "strategy_model_path": "strategy_models/carry_USDJPY_model_gnb.pkl",
        "scaler_path": "strategy_models/carry_USDJPY_scaler.pkl",
        "lags": 5,
    })

    assert signal == 1
    assert len(logged_calls) == 1, f"Expected 1 log_signal call, got {len(logged_calls)}"
    call = logged_calls[0]
    assert call["ticker"] == "carry:USDJPY"
    assert call["model_type"] == "gnb"
    assert call["strategy_type"] == "carry"
    assert call["signal"] == 1
    assert call["raw_score"] == 0.7


def test_carry_signal_logs_even_on_direct_load_fallback(monkeypatch):
    """When _get_carry_signal falls back to direct model load, it must still log."""
    import numpy as np
    import pickle as _pickle

    logger = logging.getLogger("test_carry_signal_logging_fallback")
    manager = CashPortfolioManager(
        DummyIB({"USD": -5000.0, "JPY": 0.0}),
        logger,
        dry_run=True,
    )

    logged_calls = []

    def fake_log_signal(**kwargs):
        logged_calls.append(kwargs)

    monkeypatch.setattr(
        "cash_portfolio_manager._log_signal_history",
        fake_log_signal,
    )

    # Force fallback: no strategy_executor
    manager.strategy_executor = None

    # Mock the model loading and feature generation
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([-0.5])
    monkeypatch.setattr(_pickle, "load", lambda f: fake_model)
    monkeypatch.setattr(manager, "_get_live_features", lambda pair, cfg: np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))

    # Mock Path.exists to True so the model_path check passes
    original_exists = Path.exists

    def fake_exists(self):
        if "carry_USDJPY_model_gnb.pkl" in str(self):
            return True
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    signal = manager._get_carry_signal("USDJPY", {
        "model_type": "gnb",
        "strategy_model_path": "strategy_models/carry_USDJPY_model_gnb.pkl",
        "scaler_path": "strategy_models/carry_USDJPY_scaler.pkl",
        "lags": 5,
    })

    assert signal == -1
    assert len(logged_calls) == 1, f"Expected 1 log_signal call, got {len(logged_calls)}"
    call = logged_calls[0]
    assert call["ticker"] == "carry:USDJPY"
    assert call["signal"] == -1
