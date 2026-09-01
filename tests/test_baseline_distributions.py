"""Tests for baseline_distributions (Phase 2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algos.wfov import baseline_distributions as bd


def _make_summary(ticker: str, model: str, sharpe_p25: float = 0.5,
                  sharpe_p75: float = 1.5, max_dd_p95: float = -0.10) -> dict:
    """Minimal WFOV summary stub."""
    return {
        "metadata": {
            "model_name": model,
            "ticker": ticker,
            "iterations_successful": 400,
        },
        "performance_metrics": {
            "annual_return": {
                "mean": 0.05, "std": 0.18, "min": -0.5, "max": 0.7,
                "median": 0.04, "percentile_25": -0.05,
                "percentile_75": 0.16, "percentile_95": 0.35, "count": 400,
            },
            "annual_volatility": {
                "mean": 0.15, "std": 0.05, "min": 0.10, "max": 0.30,
                "median": 0.14, "percentile_25": 0.12,
                "percentile_75": 0.19, "percentile_95": 0.25, "count": 400,
            },
            "sharpe_ratio": {
                "mean": 1.0, "std": 0.5, "min": -2.0, "max": 3.0,
                "median": 1.0, "percentile_25": sharpe_p25,
                "percentile_75": sharpe_p75, "percentile_95": 2.0, "count": 400,
            },
            "max_drawdown": {
                "mean": -0.12, "std": 0.06, "min": -0.40, "max": -0.02,
                "median": -0.11, "percentile_25": -0.16,
                "percentile_75": -0.08, "percentile_95": max_dd_p95, "count": 400,
            },
            "hit_ratio": {
                "mean": 0.52, "std": 0.04, "min": 0.40, "max": 0.80,
                "median": 0.51, "percentile_25": 0.49,
                "percentile_75": 0.54, "percentile_95": 0.58, "count": 400,
            },
            "skewness": {"mean": 0.0, "std": 0.5, "min": -1, "max": 1,
                          "median": 0.0, "percentile_25": -0.3,
                          "percentile_75": 0.3, "percentile_95": 0.5, "count": 400},
            "kurtosis": {"mean": 3.0, "std": 1.0, "min": 1, "max": 10,
                          "median": 3.0, "percentile_25": 2.5,
                          "percentile_75": 4.0, "percentile_95": 6.0, "count": 400},
            "longest_drawdown_days": {
                "mean": 150, "std": 90, "min": 16, "max": 500,
                "median": 120, "percentile_25": 90,
                "percentile_75": 200, "percentile_95": 320, "count": 400,
            },
        },
    }


class TestCollectPerTickerBaselines:
    def test_matches_exact_ticker_and_model(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        (results_dir / "montec_gnb_BK_400iter_summary.json").write_text(
            json.dumps(_make_summary("AAA", "gnb"))
        )

        portfolio = {"AAA": {"model_type": "gnb"}}
        baseline = bd.collect_per_ticker_baselines(portfolio, results_dir=results_dir)
        assert "AAA" in baseline["tickers"]
        assert baseline["tickers"]["AAA"]["model"] == "gnb"
        assert "sharpe_ratio" in baseline["tickers"]["AAA"]["metrics"]
        assert baseline["tickers"]["AAA"]["metrics"]["sharpe_ratio"]["p25"] == 0.5

    def test_picks_latest_when_multiple_match(self, tmp_path):
        import time
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        f1 = results_dir / "montec_gnb_BK_a_summary.json"
        f1.write_text(json.dumps(_make_summary("AAA", "gnb", sharpe_p25=0.1)))
        time.sleep(0.02)
        f2 = results_dir / "montec_gnb_BK_b_summary.json"
        f2.write_text(json.dumps(_make_summary("AAA", "gnb", sharpe_p25=0.9)))

        baseline = bd.collect_per_ticker_baselines(
            {"AAA": {"model_type": "gnb"}}, results_dir=results_dir,
        )
        assert baseline["tickers"]["AAA"]["metrics"]["sharpe_ratio"]["p25"] == 0.9
        assert "b_summary" in baseline["tickers"]["AAA"]["source_summary"]

    def test_skips_missing_ticker(self, tmp_path, caplog):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        baseline = bd.collect_per_ticker_baselines(
            {"NOSUCH": {"model_type": "gnb"}}, results_dir=results_dir,
        )
        assert baseline["tickers"] == {}

    def test_portfolio_hash_stable_for_same_inputs(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        portfolio = {"AAA": {"model_type": "gnb"}, "TLT": {"model_type": "lstm"}}
        b1 = bd.collect_per_ticker_baselines(portfolio, results_dir=results_dir)
        b2 = bd.collect_per_ticker_baselines(portfolio, results_dir=results_dir)
        assert b1["portfolio_hash"] == b2["portfolio_hash"]


class TestAggregateAndWrite:
    def test_aggregate_produces_metrics(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        for tkr in ["AAA", "TLT", "GLD"]:
            (results_dir / f"montec_gnb_{tkr}_summary.json").write_text(
                json.dumps(_make_summary(tkr, "gnb"))
            )
        portfolio = {tkr: {"model_type": "gnb"} for tkr in ["AAA", "TLT", "GLD"]}
        per_ticker = bd.collect_per_ticker_baselines(portfolio, results_dir=results_dir)
        agg = bd.aggregate_portfolio_baseline(per_ticker)
        assert "sharpe_ratio.p25" in agg
        assert "max_drawdown.p95" in agg

    def test_write_baseline_round_trip(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        (results_dir / "montec_gnb_BK_summary.json").write_text(
            json.dumps(_make_summary("AAA", "gnb"))
        )
        per_ticker = bd.collect_per_ticker_baselines(
            {"AAA": {"model_type": "gnb"}}, results_dir=results_dir,
        )
        out = bd.write_baseline(per_ticker, out_dir=tmp_path / "baselines")
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "portfolio_aggregated_metrics" in payload
        assert "tickers" in payload


class TestKnownGaps:
    """Phase-4 Class-B workflow improvement: baselines self-document
    their coverage gaps. Without this, operators have to manually diff
    the baseline JSON against the live TARGET_ALLOCATION to discover
    which tickers are missing — exactly the failure mode that triggered
    the 2026-05-24 remediation session."""

    def test_known_gaps_enumerates_missing_tickers(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        # Only AAA has a summary; TLT and GLD are missing.
        (results_dir / "montec_gnb_BK_summary.json").write_text(
            json.dumps(_make_summary("AAA", "gnb"))
        )
        portfolio = {
            "AAA": {"model_type": "gnb"},
            "TLT": {"model_type": "lstm"},
            "GLD": {"model_type": "lstm"},
        }
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        assert "known_gaps" in baseline, "baseline must self-document gaps"
        gap_tickers = {g["ticker"] for g in baseline["known_gaps"]}
        assert gap_tickers == {"TLT", "GLD"}, (
            f"expected gaps for TLT and GLD, got {gap_tickers}"
        )

    def test_known_gaps_records_reason(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        # TLT summary exists but with model_name="lstm_optimized", not "lstm"
        sm = _make_summary("TLT", "lstm_optimized")
        (results_dir / "montec_lstm_optimized_TLT_summary.json").write_text(
            json.dumps(sm)
        )
        portfolio = {
            "TLT": {"model_type": "lstm"},   # mismatch case
            "GLD": {"model_type": "lstm"},  # no_summary case
        }
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        gaps_by_ticker = {g["ticker"]: g for g in baseline["known_gaps"]}
        assert gaps_by_ticker["TLT"]["reason"] == "model_name_mismatch"
        assert gaps_by_ticker["GLD"]["reason"] == "no_summary_for_ticker"

    def test_known_gaps_carries_live_weight_when_provided(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        portfolio = {
            "AAA": {"model_type": "gnb", "live_weight": 0.04517},
            "GLD": {"model_type": "lstm", "live_weight": 0.12},
        }
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        gaps = {g["ticker"]: g for g in baseline["known_gaps"]}
        # Both should be in known_gaps (no summaries provided), with live_weight populated
        assert gaps["AAA"]["live_weight"] == 0.04517
        assert gaps["GLD"]["live_weight"] == 0.12

    def test_coverage_by_weight_reported(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        # Cover AAA only; TLT and GLD missing.
        (results_dir / "montec_gnb_BK_summary.json").write_text(
            json.dumps(_make_summary("AAA", "gnb"))
        )
        portfolio = {
            "AAA": {"model_type": "gnb", "live_weight": 0.25},
            "TLT": {"model_type": "lstm", "live_weight": 0.50},
            "GLD": {"model_type": "lstm", "live_weight": 0.25},
        }
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        cov = baseline.get("coverage")
        assert cov is not None
        assert cov["covered_count"] == 1
        assert cov["total_count"] == 3
        assert cov["covered_weight"] == pytest.approx(0.25)
        assert cov["total_weight"] == pytest.approx(1.0)
        assert cov["coverage_by_weight"] == pytest.approx(0.25)
        assert cov["coverage_by_count"] == pytest.approx(1 / 3)

    def test_no_gaps_when_full_coverage(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        for tkr in ["AAA", "TLT"]:
            (results_dir / f"montec_gnb_{tkr}_summary.json").write_text(
                json.dumps(_make_summary(tkr, "gnb"))
            )
        portfolio = {tkr: {"model_type": "gnb"} for tkr in ["AAA", "TLT"]}
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        # known_gaps key exists but is empty
        assert baseline.get("known_gaps") == []
        cov = baseline["coverage"]
        assert cov["coverage_by_count"] == 1.0

    def test_write_baseline_persists_known_gaps(self, tmp_path):
        results_dir = tmp_path / "summaries"
        results_dir.mkdir()
        (results_dir / "montec_gnb_BK_summary.json").write_text(
            json.dumps(_make_summary("AAA", "gnb"))
        )
        portfolio = {
            "AAA": {"model_type": "gnb"},
            "GLD": {"model_type": "lstm"},
        }
        baseline = bd.collect_per_ticker_baselines(
            portfolio, results_dir=results_dir,
        )
        out = bd.write_baseline(baseline, out_dir=tmp_path / "baselines")
        payload = json.loads(out.read_text())
        assert "known_gaps" in payload
        assert "coverage" in payload
        # The aggregated metrics block is still present (backward compat)
        assert "portfolio_aggregated_metrics" in payload
