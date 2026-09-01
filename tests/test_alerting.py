"""Tests for the alerting module (Phase 0.4).

The Telegram send path is exercised via a fake transport so no real HTTP
call is made. We verify:

* The bot token / chat ID are pulled from env vars.
* Missing credentials → graceful no-op (does NOT raise).
* The message format includes severity, title, and body.
* A transport failure does NOT propagate (defense-in-depth).
"""

from __future__ import annotations

from unittest import mock

import pytest

import alerting  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


class TestSendAlert:
    def test_missing_credentials_no_op(self):
        # No env vars set: should not raise.
        result = alerting.send_alert("title", "body", "warning")
        assert result is False  # delivered=False

    def test_dispatches_when_credentials_present(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

        with mock.patch.object(alerting, "_post_to_telegram") as post:
            post.return_value = True
            ok = alerting.send_alert("ks-fire", "MTD -9%", "critical")

        assert ok is True
        assert post.call_count == 1
        url_arg, payload = post.call_args.args
        assert "fake-token" in url_arg
        assert payload["chat_id"] == "12345"
        # Body should contain severity, title, and message.
        body = payload["text"]
        assert "critical" in body.lower()
        assert "ks-fire" in body
        assert "MTD -9%" in body

    def test_transport_failure_swallowed(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

        with mock.patch.object(alerting, "_post_to_telegram", side_effect=RuntimeError("network")):
            # Must NOT raise.
            result = alerting.send_alert("t", "b", "warning")

        assert result is False

    def test_severity_prefix_emoji_or_text(self, monkeypatch):
        """The formatted text should distinguish severity levels for the
        operator's at-a-glance scanning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")

        with mock.patch.object(alerting, "_post_to_telegram") as post:
            post.return_value = True
            alerting.send_alert("t1", "b1", "critical")
            alerting.send_alert("t2", "b2", "warning")
            alerting.send_alert("t3", "b3", "info")

        bodies = [c.args[1]["text"] for c in post.call_args_list]
        assert any("CRITICAL" in b.upper() for b in bodies[:1])
        assert any("WARNING" in b.upper() for b in bodies[1:2])
        assert any("INFO" in b.upper() for b in bodies[2:3])
