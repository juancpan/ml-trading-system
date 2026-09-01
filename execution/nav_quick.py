"""Print latest NetLiquidation in USD from ``account_values.pkl``.

Used by ``run_region.sh`` to pass ``--nav`` to ``preflight_check.py``.

Exit codes:
* 0 — printed a positive NAV to stdout.
* 1 — could not read state file / NAV missing / NAV non-positive. The
  caller should treat this as a preflight failure and refuse to trade.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Load .env into os.environ (only keys not already set). See env_loader.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import env_loader  # noqa: F401  (side-effect: populates os.environ)
except Exception:
    pass

STATE_FILE = Path(__file__).resolve().parent / "account_values.pkl"


def get_nav() -> float | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "rb") as f:
            account_values = pickle.load(f)
    except Exception:
        return None

    val = account_values.get("NetLiquidation")
    if val is None:
        return None
    try:
        # `update_account_value` stores {"value": float, ...}
        if isinstance(val, dict):
            nav = float(val.get("value", 0.0))
        else:
            nav = float(val)
    except (TypeError, ValueError):
        return None
    return nav if nav > 0 else None


def main() -> int:
    nav = get_nav()
    if nav is None:
        print("0", file=sys.stdout)
        print("nav_quick: could not determine NAV from account_values.pkl",
              file=sys.stderr)
        return 1
    # Format with no decimals — preflight_check.py accepts float strings.
    print(f"{nav:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
