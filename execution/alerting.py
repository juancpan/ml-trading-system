"""Lightweight Telegram alerting (Phase 0.4 of the Revision Protocol).

Stateless. Credentials come from environment variables (matching the
existing ``portfolio_oversight/config/oversight_config.yaml`` convention):

    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

If either is missing, ``send_alert`` is a graceful no-op so unit tests
and unconfigured developer machines do not crash.

Design choices:

* No external dependencies beyond ``urllib`` from the stdlib. The live
  trading process already pulls in heavy ML stacks; the alerting path
  must not add fragile imports.
* Failures in the network call do NOT propagate. Alerting is
  defense-in-depth, not the primary signal. Sentinel files are the
  authoritative kill-switch state.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

LOGGER = logging.getLogger(__name__)


_TELEGRAM_API_FMT = "https://api.telegram.org/bot{token}/sendMessage"
_VALID_SEVERITY = ("info", "warning", "critical")


def _post_to_telegram(url: str, payload: dict[str, Any]) -> bool:
    """POST to the Telegram Bot API. Returns True on HTTP 2xx, else False.

    Isolated so tests can mock the network boundary.
    """
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        LOGGER.warning("Telegram POST failed: %s", exc)
        return False


def send_alert(title: str, message: str, severity: str = "info") -> bool:
    """Send a Telegram alert. Returns True if delivered, False otherwise.

    Severity is one of ``info``, ``warning``, ``critical``. Unknown values
    are normalized to ``info``.

    Side-effects:

    * Logs the alert at the matching log level.
    * If both env vars are present, posts to Telegram.
    """
    sev = (severity or "info").lower()
    if sev not in _VALID_SEVERITY:
        sev = "info"

    log_method = {
        "info": LOGGER.info,
        "warning": LOGGER.warning,
        "critical": LOGGER.error,
    }[sev]
    log_method("ALERT [%s] %s: %s", sev.upper(), title, message)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        LOGGER.debug(
            "Telegram credentials not set (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID); "
            "alert not dispatched."
        )
        return False

    text = f"[{sev.upper()}] {title}\n\n{message}"
    url = _TELEGRAM_API_FMT.format(token=token)
    payload = {"chat_id": chat, "text": text}

    try:
        return _post_to_telegram(url, payload)
    except Exception as exc:
        LOGGER.error("send_alert unexpected failure: %s", exc)
        return False


def send_kill_switch_alert(decision: Any) -> bool:
    """Convenience for kill_switch.apply_decision.

    Accepts a :class:`kill_switch.KillSwitchDecision` and dispatches a
    severity-mapped alert.
    """
    tier = getattr(decision, "tier", None)
    severity_map = {
        "hard_kill": "critical",
        "soft_halt": "warning",
        "daily_move_alarm": "warning",
        "ok": "info",
    }
    severity = severity_map.get(getattr(tier, "value", "ok"), "info")
    title = f"Kill-switch: {getattr(tier, 'value', 'unknown')}"
    message = getattr(decision, "reason", "(no reason)")
    return send_alert(title, message, severity)
