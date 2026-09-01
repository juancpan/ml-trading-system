"""Phase 0 integration smoke test.

End-to-end: write an equity history that triggers a hard kill, run the
kill_switch CLI, verify the sentinel is written and exit code is 2.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
IBKR_DIR = REPO / "execution"


def _alert_safe_env() -> dict:
    """Env for kill_switch subprocesses that GUARANTEES no real Telegram.

    These tests feed synthetic -10%/-6% drawdowns to the real kill_switch
    CLI, which would otherwise dispatch live alerts. We strip the Telegram
    credentials and set SKIP_ALERTS so neither the Python nor the shell alert
    path can fire, regardless of what env_loader picks up from .env.
    """
    env = dict(os.environ)
    env.pop("TELEGRAM_BOT_TOKEN", None)
    env.pop("TELEGRAM_CHAT_ID", None)
    env["SKIP_ALERTS"] = "1"
    # Tell env_loader not to reload .env in this subprocess (see env_loader.py).
    env["PYTEST_CURRENT_TEST"] = env.get("PYTEST_CURRENT_TEST", "integration")
    return env


@pytest.fixture
def isolated_ibkr_dir(tmp_path, monkeypatch):
    """Create a tmp execution-like dir with stubbed paths the modules use.

    We can't easily monkey-patch the module-level path constants without
    re-importing. Instead, we just verify the writer/CLI agree on a tmp
    path via subprocess + harness.
    """
    return tmp_path


def test_hard_kill_end_to_end(isolated_ibkr_dir, tmp_path):
    """Drop a synthetic equity_history.parquet with a 10% MTD drawdown,
    invoke kill_switch CLI, verify exit code 2 and sentinel content."""
    eq_path = tmp_path / "equity_history.parquet"
    sentinel_path = tmp_path / "KILL_SWITCH_ACTIVE"

    # Build equity history: month opens at 10_000, today at 9_000 → -10%.
    base = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base, "region": "US", "event": "start",
         "nav_usd": 10_000.0, "cash_usd": 0.0, "gross_exposure": 0.0,
         "leverage": 1.0, "kill_switch_active": False},
        {"timestamp": base + timedelta(days=14), "region": "US", "event": "eod",
         "nav_usd": 9_000.0, "cash_usd": 0.0, "gross_exposure": 0.0,
         "leverage": 1.0, "kill_switch_active": False},
    ]
    pd.DataFrame(rows).to_parquet(eq_path, index=False)

    # Harness: import kill_switch with patched paths.
    harness = (
        f"import sys; sys.path.insert(0, '{IBKR_DIR}'); "
        "import kill_switch; "
        "from pathlib import Path; "
        f"kill_switch.EQUITY_HISTORY_PATH = Path('{eq_path}'); "
        f"kill_switch.HARD_KILL_SENTINEL = Path('{sentinel_path}'); "
        f"kill_switch.SOFT_HALT_SENTINEL = Path('{tmp_path}/SOFT_HALT_ACTIVE'); "
        f"kill_switch.DAILY_MOVE_SENTINEL = Path('{tmp_path}/DAILY_MOVE_ACTIVE'); "
        "sys.exit(kill_switch.main())"
    )
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True, text=True, timeout=15, env=_alert_safe_env(),
    )

    assert result.returncode == 2, (
        f"Expected exit code 2 (hard kill), got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert sentinel_path.exists(), "KILL_SWITCH_ACTIVE sentinel not written"
    import json
    payload = json.loads(sentinel_path.read_text())
    assert payload["details"]["tier"] == "hard_kill"
    assert payload["details"]["mtd_drawdown"] == pytest.approx(-0.10, abs=1e-9)


def test_soft_halt_end_to_end(tmp_path):
    """6% MTD drawdown → soft halt, exit 1, SOFT_HALT_ACTIVE sentinel."""
    eq_path = tmp_path / "equity_history.parquet"
    soft_sentinel = tmp_path / "SOFT_HALT_ACTIVE"
    hard_sentinel = tmp_path / "KILL_SWITCH_ACTIVE"

    base = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base, "region": "US", "event": "start",
         "nav_usd": 10_000.0, "cash_usd": 0.0, "gross_exposure": 0.0,
         "leverage": 1.0, "kill_switch_active": False},
        {"timestamp": base + timedelta(days=14), "region": "US", "event": "eod",
         "nav_usd": 9_400.0, "cash_usd": 0.0, "gross_exposure": 0.0,
         "leverage": 1.0, "kill_switch_active": False},
    ]
    pd.DataFrame(rows).to_parquet(eq_path, index=False)

    harness = (
        f"import sys; sys.path.insert(0, '{IBKR_DIR}'); "
        "import kill_switch; "
        "from pathlib import Path; "
        f"kill_switch.EQUITY_HISTORY_PATH = Path('{eq_path}'); "
        f"kill_switch.HARD_KILL_SENTINEL = Path('{hard_sentinel}'); "
        f"kill_switch.SOFT_HALT_SENTINEL = Path('{soft_sentinel}'); "
        f"kill_switch.DAILY_MOVE_SENTINEL = Path('{tmp_path}/DAILY_MOVE_ACTIVE'); "
        "sys.exit(kill_switch.main())"
    )
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True, text=True, timeout=15, env=_alert_safe_env(),
    )

    assert result.returncode == 1, (
        f"Expected exit 1 (soft halt), got {result.returncode}.\n{result.stderr}"
    )
    assert soft_sentinel.exists()
    assert not hard_sentinel.exists()


def test_ok_end_to_end(tmp_path):
    """Flat performance → OK, exit 0, no sentinels."""
    eq_path = tmp_path / "equity_history.parquet"
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base + timedelta(days=i), "region": "US",
         "event": "eod", "nav_usd": 10_000.0 + i, "cash_usd": 0,
         "gross_exposure": 0, "leverage": 1.0, "kill_switch_active": False}
        for i in range(7)
    ]
    pd.DataFrame(rows).to_parquet(eq_path, index=False)

    harness = (
        f"import sys; sys.path.insert(0, '{IBKR_DIR}'); "
        "import kill_switch; "
        "from pathlib import Path; "
        f"kill_switch.EQUITY_HISTORY_PATH = Path('{eq_path}'); "
        f"kill_switch.HARD_KILL_SENTINEL = Path('{tmp_path}/HARD'); "
        f"kill_switch.SOFT_HALT_SENTINEL = Path('{tmp_path}/SOFT'); "
        f"kill_switch.DAILY_MOVE_SENTINEL = Path('{tmp_path}/DAILY'); "
        "sys.exit(kill_switch.main())"
    )
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True, text=True, timeout=15, env=_alert_safe_env(),
    )

    assert result.returncode == 0
    assert not (tmp_path / "HARD").exists()
    assert not (tmp_path / "SOFT").exists()
    assert not (tmp_path / "DAILY").exists()
