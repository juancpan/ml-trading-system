import pytest
import pandas as pd
import numpy as np


class TestGatedPortfolioBacktest:
    def _make_synthetic_returns(self, n_days=500, n_tickers=5, seed=42):
        np.random.seed(seed)
        dates = pd.bdate_range("2022-01-01", periods=n_days)
        tickers = [f"TICK_{i}" for i in range(n_tickers)]
        returns = pd.DataFrame(
            np.random.normal(0.0003, 0.015, (n_days, n_tickers)),
            index=dates,
            columns=tickers,
        )
        return returns

    def test_backtest_runs_and_returns_results(self):
        from algos.backtest_code.validate_gated_portfolio_oos import (
            run_gated_portfolio_backtest,
        )

        returns = self._make_synthetic_returns()
        hrp_weights = {f"TICK_{i}": 0.2 for i in range(5)}

        def dummy_signal_fn(ticker, features_df, week_date):
            return 1 if hash(ticker + str(week_date)) % 3 != 0 else -1

        result = run_gated_portfolio_backtest(
            returns_df=returns,
            hrp_weights=hrp_weights,
            signal_fn=dummy_signal_fn,
            max_weight=0.40,
            min_active_tickers=2,
        )
        assert "portfolio_returns" in result
        assert "metrics" in result
        assert "weekly_allocations" in result
        assert isinstance(result["portfolio_returns"], pd.Series)
        assert len(result["portfolio_returns"]) > 0
        assert "sharpe" in result["metrics"]
        assert "max_drawdown" in result["metrics"]
        assert "turnover" in result["metrics"]

    def test_all_gates_open_matches_static_hrp(self):
        from algos.backtest_code.validate_gated_portfolio_oos import (
            run_gated_portfolio_backtest,
        )

        returns = self._make_synthetic_returns()
        hrp_weights = {f"TICK_{i}": 0.2 for i in range(5)}

        result = run_gated_portfolio_backtest(
            returns_df=returns,
            hrp_weights=hrp_weights,
            signal_fn=lambda t, f, d: 1,
            max_weight=0.40,
        )
        weight_arr = np.array([0.2] * 5)
        static_returns = (returns * weight_arr).sum(axis=1)
        pd.testing.assert_series_equal(
            result["portfolio_returns"],
            static_returns,
            check_names=False,
            atol=1e-10,
        )
