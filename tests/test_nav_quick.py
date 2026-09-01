"""Tests for nav_quick (Phase 0.3)."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest


NAV_QUICK = Path(__file__).resolve().parent.parent / "execution" / "nav_quick.py"


def _run_nav_quick(env_state_dir: Path) -> tuple[int, str, str]:
    """Run nav_quick.py with a controlled STATE_FILE.

    We re-run the script via subprocess with a monkey-patched STATE_FILE
    by injecting a tiny harness that imports the module then prints the
    result. This avoids module-state pollution between tests.
    """
    harness = (
        f"import sys; sys.path.insert(0, '{NAV_QUICK.parent}'); "
        "import nav_quick; "
        f"nav_quick.STATE_FILE = __import__('pathlib').Path('{env_state_dir}') / 'account_values.pkl'; "
        "sys.exit(nav_quick.main())"
    )
    result = subprocess.run(
        [sys.executable, "-c", harness],
        capture_output=True, text=True, timeout=15,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


class TestNavQuick:
    def test_returns_nonzero_when_state_missing(self, tmp_path):
        rc, out, _err = _run_nav_quick(tmp_path)
        assert rc == 1
        assert out == "0"

    def test_returns_nav_when_state_present(self, tmp_path):
        state = {"NetLiquidation": {"value": 12345.67, "currency": "USD", "accountName": "DU..."}}
        with open(tmp_path / "account_values.pkl", "wb") as f:
            pickle.dump(state, f)
        rc, out, _err = _run_nav_quick(tmp_path)
        assert rc == 0
        assert float(out) == pytest.approx(12345.67)

    def test_rejects_zero_nav(self, tmp_path):
        state = {"NetLiquidation": {"value": 0.0, "currency": "USD", "accountName": "x"}}
        with open(tmp_path / "account_values.pkl", "wb") as f:
            pickle.dump(state, f)
        rc, out, _err = _run_nav_quick(tmp_path)
        assert rc == 1
        assert out == "0"

    def test_handles_raw_float_value(self, tmp_path):
        # Older format may store float directly, not dict.
        state = {"NetLiquidation": 9876.5}
        with open(tmp_path / "account_values.pkl", "wb") as f:
            pickle.dump(state, f)
        rc, out, _err = _run_nav_quick(tmp_path)
        assert rc == 0
        assert float(out) == pytest.approx(9876.5)

    def test_corrupt_pickle_returns_error(self, tmp_path):
        (tmp_path / "account_values.pkl").write_bytes(b"not a pickle")
        rc, out, _err = _run_nav_quick(tmp_path)
        assert rc == 1
        assert out == "0"
