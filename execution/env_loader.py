"""Minimal, dependency-free ``.env`` loader for the trading system.

Why this exists
---------------
Credentials (``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``) and connection
settings (``IBKR_PORT``) are read from the process environment by
``alerting.py`` and friends. Cron jobs do NOT inherit an interactive shell's
exported variables, so for weeks the cron failure alerts silently no-op'd
because the credentials were never in cron's environment.

This loader reads a project-root ``.env`` file and populates ``os.environ``
for any key that is **not already set**, so:

* an interactive ``export`` still takes precedence (developer override), and
* cron / direct ``python foo.py`` invocations get the values they need.

Design constraints (matching ``alerting.py``):

* **No third-party dependency** (no ``python-dotenv``). The live trading
  process already pulls heavy ML stacks; the config path must not add a
  fragile import.
* **Never raises** on a missing/garbled ``.env``. Loading config must not be
  able to crash a trading run.

Usage
-----
Import for its side effect at the top of an entrypoint::

    import env_loader  # noqa: F401  (loads .env into os.environ)

or call explicitly::

    from env_loader import load_dotenv
    load_dotenv()
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root is the parent of execution/.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"


def _parse_line(line: str) -> tuple[str, str] | None:
    """Parse a single ``KEY=VALUE`` line. Returns None for blanks/comments.

    Supports:
      * leading/trailing whitespace
      * ``#`` comment lines and blank lines
      * optional ``export KEY=VALUE`` prefix (shell-compatible)
      * surrounding single or double quotes around the value
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``. Returns the dict of values applied.

    Args:
        path: Path to the ``.env`` file. Defaults to ``<project_root>/.env``.
        override: If True, overwrite variables already present in the
            environment. Default False (existing env wins — interactive
            ``export`` beats the file).

    Returns:
        The mapping of keys that were actually set into ``os.environ`` by
        this call (excludes keys skipped because they were already set).
    """
    # Test isolation: never auto-load .env under pytest. Otherwise tests that
    # invoke the real kill_switch/alerting path with synthetic data could pick
    # up live Telegram credentials and dispatch FALSE alerts (observed
    # 2026-06-09). Tests that genuinely need creds can set them explicitly.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return {}

    env_path = Path(path) if path is not None else _DEFAULT_ENV_PATH
    applied: dict[str, str] = {}
    try:
        if not env_path.is_file():
            return applied
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            key, value = parsed
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            applied[key] = value
    except Exception:
        # Config loading must never crash a trading run.
        return applied
    return applied


# Load on import so ``import env_loader`` is enough at an entrypoint top.
load_dotenv()
