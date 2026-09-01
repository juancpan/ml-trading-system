import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `from algos.backtest_code...` imports resolve
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Add execution/ as well, because modules there use `from config import ...`
# (i.e. they assume `execution/` is on sys.path).
sys.path.insert(0, str(ROOT / "execution"))

# ---------------------------------------------------------------------------
# CRITICAL test isolation: never send real Telegram alerts from the test suite.
#
# Several tests invoke the real kill_switch.main()/apply_decision() path with
# SYNTHETIC equity (e.g. test_phase0_integration's -10% / -6% drawdowns). If
# TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are present in the environment (now loaded
# from .env via env_loader), those tests would dispatch REAL Telegram alerts
# with fake numbers — exactly the false hard_kill/soft_halt warnings observed
# 2026-06-09. alerting.send_alert is a no-op when the creds are absent, so we
# scrub them.
#
# We scrub at MODULE IMPORT time (before any test or subprocess runs) so that
# subprocess-spawning tests inherit a clean os.environ, AND via an autouse
# fixture as belt-and-suspenders. See MEMORY.md "Test isolation: Telegram".
# ---------------------------------------------------------------------------
for _var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch):
    """Guarantee no test can dispatch a real Telegram alert."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Also signal the shell-side alert path to stay silent if any subprocess
    # consults it.
    monkeypatch.setenv("SKIP_ALERTS", "1")
