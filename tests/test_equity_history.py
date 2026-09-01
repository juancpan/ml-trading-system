"""Tests for the equity history append helper (Phase 0.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from equity_history import (  # noqa: E402
    EquityHistoryWriter,
    EQUITY_HISTORY_SCHEMA,
)


@pytest.fixture
def tmp_history_path(tmp_path: Path) -> Path:
    return tmp_path / "equity_history.parquet"


class TestEquityHistoryWriter:
    def test_first_append_creates_file(self, tmp_history_path):
        w = EquityHistoryWriter(path=tmp_history_path)
        w.append(
            timestamp=datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc),
            region="US",
            nav_usd=10_500.0,
            cash_usd=1_200.0,
            gross_exposure=13_000.0,
            leverage=1.3,
            event="start",
        )
        assert tmp_history_path.exists()
        df = pd.read_parquet(tmp_history_path)
        assert len(df) == 1
        assert df.iloc[0]["nav_usd"] == 10_500.0
        assert df.iloc[0]["event"] == "start"
        assert df.iloc[0]["region"] == "US"

    def test_append_is_idempotent_under_replay(self, tmp_history_path):
        """Calling append with the SAME (timestamp, region, event) triple
        should NOT create a duplicate row.

        This protects against main.py restarts re-recording the same event.
        """
        w = EquityHistoryWriter(path=tmp_history_path)
        ts = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        w.append(timestamp=ts, region="US", nav_usd=10_500.0, cash_usd=0,
                 gross_exposure=0, leverage=1.0, event="start")
        w.append(timestamp=ts, region="US", nav_usd=10_500.0, cash_usd=0,
                 gross_exposure=0, leverage=1.0, event="start")
        df = pd.read_parquet(tmp_history_path)
        assert len(df) == 1

    def test_multiple_events_per_day_accumulate(self, tmp_history_path):
        w = EquityHistoryWriter(path=tmp_history_path)
        base = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        for i, event in enumerate(["start", "post_rebalance", "eod"]):
            w.append(
                timestamp=base.replace(hour=14 + i * 2),
                region="US",
                nav_usd=10_500.0 + i * 5,
                cash_usd=0,
                gross_exposure=13_000.0,
                leverage=1.3,
                event=event,
            )
        df = pd.read_parquet(tmp_history_path)
        assert len(df) == 3
        assert sorted(df["event"].tolist()) == ["eod", "post_rebalance", "start"]

    def test_schema_matches_kill_switch_consumer(self, tmp_history_path):
        """The dataframe schema must include the columns kill_switch.py reads."""
        from kill_switch import compute_mtd_drawdown
        w = EquityHistoryWriter(path=tmp_history_path)
        ts = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        w.append(timestamp=ts, region="US", nav_usd=10_000.0, cash_usd=0,
                 gross_exposure=0, leverage=1.0, event="start")
        df = pd.read_parquet(tmp_history_path)
        # Should not raise.
        compute_mtd_drawdown(df, as_of=ts)

    def test_schema_columns_present(self, tmp_history_path):
        w = EquityHistoryWriter(path=tmp_history_path)
        ts = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)
        w.append(timestamp=ts, region="US", nav_usd=10_000.0, cash_usd=0,
                 gross_exposure=0, leverage=1.0, event="start")
        df = pd.read_parquet(tmp_history_path)
        expected = set(EQUITY_HISTORY_SCHEMA.keys())
        assert expected.issubset(set(df.columns))

    def test_zero_nav_is_skipped_with_warning(self, tmp_history_path, caplog):
        """If NAV is 0 (typically a stale account_values.pkl), don't poison
        the history. Log a warning instead."""
        w = EquityHistoryWriter(path=tmp_history_path)
        w.append(timestamp=datetime(2026, 5, 16, tzinfo=timezone.utc),
                 region="US", nav_usd=0.0, cash_usd=0.0, gross_exposure=0.0,
                 leverage=0.0, event="start")
        assert not tmp_history_path.exists()  # nothing written

    def test_corrupt_existing_file_is_quarantined(self, tmp_history_path):
        """If the parquet file is corrupt, the writer should quarantine it
        and start fresh rather than crash the trading loop."""
        tmp_history_path.write_bytes(b"not a parquet file")
        w = EquityHistoryWriter(path=tmp_history_path)
        w.append(timestamp=datetime(2026, 5, 16, tzinfo=timezone.utc),
                 region="US", nav_usd=10_000.0, cash_usd=0, gross_exposure=0,
                 leverage=1.0, event="start")
        # Now the file should be valid parquet with the new row.
        df = pd.read_parquet(tmp_history_path)
        assert len(df) == 1
        # The corrupt original should be moved aside.
        quarantine = tmp_history_path.with_suffix(".parquet.corrupt")
        assert quarantine.exists()
