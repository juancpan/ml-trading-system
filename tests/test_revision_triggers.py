"""Tests for revision_triggers (Phase 2.3)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone, date as Date
from pathlib import Path

import pandas as pd
import pytest

import revision_triggers as rt
import attribution
from equity_history import EquityHistoryWriter


@pytest.fixture
def base_env(tmp_path):
    """Build a minimal environment: baseline JSON, equity history, attribution DB."""
    eq = tmp_path / "equity_history.parquet"
    attr_db = tmp_path / "attribution.db"
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    # Synthetic baseline: backtest annual return p25=0.05 → monthly p25 ~ 0.004,
    # min=-0.30 (annual) → monthly min ~ -0.025.
    # backtest max_dd p95 = -0.20.
    baseline = {
        "portfolio_hash": "abc",
        "generated_at": "2026-05-15T00:00:00+00:00",
        "tickers": {},
        "portfolio_aggregated_metrics": {
            "annual_return.p25": 0.05,
            "annual_return.min": -0.30,
            "max_drawdown.p95": -0.20,
        },
    }
    (baselines / "portfolio_abc_baseline.json").write_text(json.dumps(baseline))
    return {"eq": eq, "attr_db": attr_db, "baselines": baselines}


def _seed_equity(eq_path: Path, *, mtd_dd_pct: float):
    """Seed equity history so MTD drawdown ≈ mtd_dd_pct (as fraction)."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    start = 10_000.0
    end = start * (1 + mtd_dd_pct)
    w = EquityHistoryWriter(path=eq_path)
    w.append(timestamp=base, region="US", nav_usd=start,
             cash_usd=0, gross_exposure=0, leverage=1.0, event="start")
    w.append(timestamp=base + timedelta(days=14), region="US", nav_usd=end,
             cash_usd=0, gross_exposure=0, leverage=1.0, event="eod")


def _seed_hit_rates(db_path: Path, *, ticker_to_rate: dict[str, float]):
    rows = pd.DataFrame([
        {"date": Date(2026, 5, 16), "ticker": t, "model_type": "gnb",
         "window_days": 20, "hits": int(round(20 * r)),
         "misses": int(20 - round(20 * r)),
         "hit_rate": float(r)}
        for t, r in ticker_to_rate.items()
    ])
    attribution.persist_hit_rates(rows, db_path=db_path)


class TestEvaluate:
    def test_ok_when_no_data(self, base_env):
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "ok"

    def test_yellow_on_mild_underperformance(self, base_env):
        # Drawdown of -0.005 is below p25 monthly (~0.004) but above p5.
        _seed_equity(base_env["eq"], mtd_dd_pct=-0.005)
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "yellow"

    def test_orange_on_deep_underperformance(self, base_env):
        # Drawdown of -0.10 is below 5th percentile monthly (~-0.025).
        _seed_equity(base_env["eq"], mtd_dd_pct=-0.10)
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "orange"

    def test_red_on_dd_exceeding_backtest(self, base_env):
        # Drawdown of -0.30 exceeds 1.2x backtest max DD (-0.24).
        _seed_equity(base_env["eq"], mtd_dd_pct=-0.30)
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "red"

    def test_red_when_all_models_below_chance(self, base_env):
        _seed_equity(base_env["eq"], mtd_dd_pct=0.0)
        _seed_hit_rates(base_env["attr_db"],
                        ticker_to_rate={"AAA": 0.40, "TLT": 0.42, "GLD": 0.45})
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "red"

    def test_orange_when_one_model_below_threshold(self, base_env):
        _seed_equity(base_env["eq"], mtd_dd_pct=0.0)
        _seed_hit_rates(base_env["attr_db"],
                        ticker_to_rate={"AAA": 0.55, "TLT": 0.40, "GLD": 0.55})
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "orange"

    def test_ok_when_everything_normal(self, base_env):
        _seed_equity(base_env["eq"], mtd_dd_pct=0.01)
        _seed_hit_rates(base_env["attr_db"],
                        ticker_to_rate={"AAA": 0.55, "TLT": 0.52, "GLD": 0.58})
        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )
        assert ev.tier == "ok"

    def test_latest_hit_rates_not_stale_history_drive_orange(self, base_env):
        _seed_equity(base_env["eq"], mtd_dd_pct=0.01)
        old = Date(2026, 5, 15)
        latest = Date(2026, 5, 16)
        rows = pd.DataFrame([
            {"date": old, "ticker": "AAA", "model_type": "gnb", "window_days": 20,
             "hits": 0, "misses": 20, "total": 20, "unresolved": 0, "hit_rate": 0.0},
            {"date": latest, "ticker": "AAA", "model_type": "gnb", "window_days": 20,
             "hits": 12, "misses": 8, "total": 20, "unresolved": 0, "hit_rate": 0.6},
            {"date": latest, "ticker": "TLT", "model_type": "lstm", "window_days": 20,
             "hits": 0, "misses": 0, "total": 4, "unresolved": 0, "hit_rate": float("nan")},
        ])
        attribution.persist_hit_rates(rows, db_path=base_env["attr_db"])

        ev = rt.evaluate(
            equity_path=base_env["eq"],
            attr_db=base_env["attr_db"],
            baselines_dir=base_env["baselines"],
        )

        assert ev.tier == "ok"
        assert ev.metrics["n_ml_models"] == 1
        assert ev.metrics["n_unresolved_ml_models"] == 1


class TestWriteStatus:
    def test_round_trip(self, tmp_path):
        ev = rt.TriggerEvaluation(
            as_of="2026-05-16T12:00:00+00:00",
            tier="yellow", reasons=["test"], metrics={"k": 1},
        )
        path = tmp_path / "rs.json"
        rt.write_status(ev, path=path)
        payload = json.loads(path.read_text())
        assert payload["tier"] == "yellow"
        assert payload["reasons"] == ["test"]
