"""Tests for revision_dashboard (Phase 1.4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

import revision_dashboard  # noqa: E402
from equity_history import EquityHistoryWriter  # noqa: E402
from signal_history import SignalHistoryWriter  # noqa: E402
import attribution  # noqa: E402
from dataclasses import replace


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    """Point all module-level paths at a tmp dir."""
    monkeypatch.setattr(revision_dashboard, "DEFAULT_OUT", tmp_path / "dash.html")
    monkeypatch.setattr(revision_dashboard, "DEFAULT_ATTR_DB", tmp_path / "attribution.db")
    monkeypatch.setattr(revision_dashboard, "DEFAULT_EQUITY", tmp_path / "equity_history.parquet")
    monkeypatch.setattr(revision_dashboard, "DEFAULT_SHADOW", tmp_path / "shadow_history.parquet")
    monkeypatch.setattr(revision_dashboard, "HARD_KILL", tmp_path / "KILL_SWITCH_ACTIVE")
    monkeypatch.setattr(revision_dashboard, "SOFT_HALT", tmp_path / "SOFT_HALT_ACTIVE")
    monkeypatch.setattr(revision_dashboard, "DAILY_MOVE", tmp_path / "DAILY_MOVE_ACTIVE")
    return tmp_path


class TestRender:
    def test_empty_environment_renders_without_error(self, fake_paths):
        out = revision_dashboard.render(fake_paths / "dash.html")
        assert out.exists()
        html = out.read_text()
        assert "Revision Health Dashboard" in html
        assert "Kill-switch sentinels" in html
        # All three sentinels should show as clear.
        assert html.count("clear") >= 3

    def test_hard_kill_sentinel_shows_as_active(self, fake_paths):
        import json
        sentinel = fake_paths / "KILL_SWITCH_ACTIVE"
        sentinel.write_text(json.dumps({
            "written_at": "2026-05-16T12:00:00+00:00",
            "reason": "MTD -10%",
            "details": {"tier": "hard_kill"},
        }))
        out = revision_dashboard.render(fake_paths / "dash.html")
        html = out.read_text()
        assert "ACTIVE" in html
        assert "MTD -10%" in html
        assert "badge-crit" in html

    def test_renders_equity_curve_when_present(self, fake_paths):
        eq_path = fake_paths / "equity_history.parquet"
        w = EquityHistoryWriter(path=eq_path)
        base = datetime.now(timezone.utc) - timedelta(days=10)
        for i in range(10):
            w.append(timestamp=base + timedelta(days=i), region="US",
                     nav_usd=10_000.0 + i * 50, cash_usd=0,
                     gross_exposure=0, leverage=1.0, event="eod")
        out = revision_dashboard.render(fake_paths / "dash.html")
        html = out.read_text()
        assert "<svg" in html

    def test_renders_attribution_table_when_present(self, fake_paths):
        from datetime import date as Date
        db = fake_paths / "attribution.db"
        attr = attribution.DailyAttribution(
            date=Date.today(), region="US",
            total_pnl=120.0, execution_drag=-5.0,
            signal_contribution=float("nan"),
            weighting_contribution=float("nan"),
            sizing_contribution=float("nan"),
            nav_open=10_000, nav_close=10_120,
        )
        attribution.persist_daily_attribution(attr, db_path=db)
        out = revision_dashboard.render(fake_paths / "dash.html")
        html = out.read_text()
        assert "+120.00" in html
        assert "US" in html
        assert "region_nav_delta" in html
        assert "Phase 1.4 pending" in html
        assert "total_pnl" not in html
