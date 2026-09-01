"""
User Confirmation Manager for Cross-Market Transition Safety

Handles user interaction when risky cross-market transitions are detected.
Supports:
- Interactive stdin input
- File-based flag for automation
- Timeout with safe default
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging
import sys
import select
import time
from pathlib import Path


class TransitionAction(Enum):
    """User decision on how to handle a risky transition."""
    PROCEED = "proceed"           # Execute all trades as planned
    CANCEL = "cancel"             # Cancel all trades today
    SELLS_ONLY = "sells_only"     # Execute only SELL orders
    SCALE_DOWN = "scale_down"     # Scale down buys to fit constraints


@dataclass
class TransitionDecision:
    """Result of user confirmation request."""
    action: TransitionAction
    reason: str                   # Why this decision was made
    user_input: Optional[str]     # Raw user input (if interactive)
    timed_out: bool = False       # Whether decision was due to timeout


class UserConfirmationManager:
    """
    Manages user confirmation for risky cross-market transitions.

    Supports multiple input modes:
    1. Interactive stdin (default)
    2. File-based flag (for automation)
    3. Auto-timeout with safe default
    """

    # File flag for automated confirmation
    # Create this file with content "proceed", "cancel", "sells_only", or "scale_down"
    FLAG_FILE = Path(__file__).parent / ".transition_decision_flag"

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize UserConfirmationManager.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Load config
        self._load_config()

    def _load_config(self):
        """Load confirmation configuration."""
        from config import (
            TRANSITION_CONFIRMATION_TIMEOUT,
            TRANSITION_DEFAULT_ON_TIMEOUT,
        )

        self.timeout = TRANSITION_CONFIRMATION_TIMEOUT
        self.default_action = TRANSITION_DEFAULT_ON_TIMEOUT

    def request_confirmation(
        self,
        metrics,
        report: str
    ) -> TransitionDecision:
        """
        Request user confirmation for a risky transition.

        Args:
            metrics: TransitionMetrics from safety analysis
            report: Formatted report string to display

        Returns:
            TransitionDecision with user's choice
        """
        # First, check for file-based flag (automation)
        decision = self._check_flag_file()
        if decision:
            self.logger.info(f"Using file flag decision: {decision.action.value}")
            return decision

        # Display the report
        self._display_report(report)

        # Display options
        self._display_options(metrics)

        # Get user input with timeout
        return self._get_user_input()

    def _check_flag_file(self) -> Optional[TransitionDecision]:
        """
        Check for file-based decision flag.

        This allows automation systems to pre-set the decision.

        Returns:
            TransitionDecision if flag file exists, None otherwise
        """
        if not self.FLAG_FILE.exists():
            return None

        try:
            content = self.FLAG_FILE.read_text().strip().lower()
            self.FLAG_FILE.unlink()  # Remove flag after reading (one-time use)

            action_map = {
                "proceed": TransitionAction.PROCEED,
                "p": TransitionAction.PROCEED,
                "cancel": TransitionAction.CANCEL,
                "c": TransitionAction.CANCEL,
                "sells_only": TransitionAction.SELLS_ONLY,
                "sells": TransitionAction.SELLS_ONLY,
                "s": TransitionAction.SELLS_ONLY,
                "scale_down": TransitionAction.SCALE_DOWN,
                "scale": TransitionAction.SCALE_DOWN,
                "d": TransitionAction.SCALE_DOWN,
            }

            action = action_map.get(content)
            if action:
                return TransitionDecision(
                    action=action,
                    reason=f"File flag: {self.FLAG_FILE}",
                    user_input=content
                )
            else:
                self.logger.warning(f"Invalid flag file content: '{content}'")
                return None

        except Exception as e:
            self.logger.warning(f"Error reading flag file: {e}")
            return None

    def _display_report(self, report: str):
        """Display the transition analysis report."""
        print("\n" + report)
        sys.stdout.flush()

    def _display_options(self, metrics):
        """Display available options to the user."""
        print("\n" + "=" * 70)
        print("OPTIONS:")
        print("  [P] Proceed      - Execute all trades as planned")
        print("  [C] Cancel       - Skip all trades today (safe)")
        print("  [S] Sells-only   - Execute only SELL orders")
        print("  [D] Scale-down   - Reduce BUY orders to fit within limits")
        print("")
        print(f"  Timeout: {self.timeout}s | Default on timeout: {self.default_action.upper()}")
        print("=" * 70)
        print("")
        sys.stdout.flush()

    def _get_user_input(self) -> TransitionDecision:
        """
        Get user input with timeout.

        Uses select() for non-blocking input on Unix systems.
        Falls back to simple input() with timeout on Windows.

        Returns:
            TransitionDecision based on user input or timeout
        """
        prompt = "Enter choice [P/C/S/D]: "
        print(prompt, end="", flush=True)

        start_time = time.time()
        remaining = self.timeout

        try:
            # Try to use select for non-blocking input (Unix)
            if hasattr(select, 'select') and sys.stdin.isatty():
                while remaining > 0:
                    # Check if input is available
                    ready, _, _ = select.select([sys.stdin], [], [], min(1.0, remaining))

                    if ready:
                        user_input = sys.stdin.readline().strip().lower()
                        return self._parse_input(user_input)

                    elapsed = time.time() - start_time
                    remaining = self.timeout - elapsed

                    # Show countdown every 30 seconds
                    if int(remaining) % 30 == 0 and int(remaining) != self.timeout:
                        print(f"\r[{int(remaining)}s remaining] {prompt}", end="", flush=True)

                # Timeout reached
                print("\n[TIMEOUT]")
                return self._get_default_decision()

            else:
                # Fallback for non-TTY or Windows: simple input with timeout thread
                # For simplicity, just use blocking input on Windows
                # In production, consider using threading or asyncio
                self.logger.warning(
                    "Interactive input not available. "
                    f"Using default action: {self.default_action}"
                )
                return self._get_default_decision()

        except KeyboardInterrupt:
            print("\n[Interrupted]")
            return TransitionDecision(
                action=TransitionAction.CANCEL,
                reason="User interrupted with Ctrl+C",
                user_input=None
            )

        except Exception as e:
            self.logger.error(f"Error getting user input: {e}")
            return self._get_default_decision()

    def _parse_input(self, user_input: str) -> TransitionDecision:
        """
        Parse user input and return corresponding decision.

        Args:
            user_input: Raw user input string

        Returns:
            TransitionDecision based on input
        """
        input_lower = user_input.strip().lower()

        action_map = {
            "p": TransitionAction.PROCEED,
            "proceed": TransitionAction.PROCEED,
            "c": TransitionAction.CANCEL,
            "cancel": TransitionAction.CANCEL,
            "s": TransitionAction.SELLS_ONLY,
            "sells": TransitionAction.SELLS_ONLY,
            "sells_only": TransitionAction.SELLS_ONLY,
            "d": TransitionAction.SCALE_DOWN,
            "scale": TransitionAction.SCALE_DOWN,
            "scale_down": TransitionAction.SCALE_DOWN,
        }

        action = action_map.get(input_lower)

        if action:
            return TransitionDecision(
                action=action,
                reason=f"User selected: {action.value}",
                user_input=user_input
            )
        else:
            # Invalid input - treat as cancel (safe)
            self.logger.warning(f"Invalid input '{user_input}', treating as cancel")
            return TransitionDecision(
                action=TransitionAction.CANCEL,
                reason=f"Invalid input '{user_input}', defaulting to cancel",
                user_input=user_input
            )

    def _get_default_decision(self) -> TransitionDecision:
        """
        Get the default decision (used on timeout or error).

        Returns:
            TransitionDecision with default action
        """
        action_map = {
            "cancel": TransitionAction.CANCEL,
            "proceed": TransitionAction.PROCEED,
            "sells_only": TransitionAction.SELLS_ONLY,
            "scale_down": TransitionAction.SCALE_DOWN,
        }

        action = action_map.get(self.default_action.lower(), TransitionAction.CANCEL)

        return TransitionDecision(
            action=action,
            reason=f"Timeout ({self.timeout}s), using default: {self.default_action}",
            user_input=None,
            timed_out=True
        )

    @classmethod
    def set_flag_file(cls, action: str) -> bool:
        """
        Create a flag file for automated decision.

        This can be called by external scripts to pre-set the decision
        before the trading system runs.

        Args:
            action: One of "proceed", "cancel", "sells_only", "scale_down"

        Returns:
            True if flag file was created successfully
        """
        valid_actions = {"proceed", "cancel", "sells_only", "scale_down"}
        if action.lower() not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")

        try:
            cls.FLAG_FILE.write_text(action.lower())
            return True
        except Exception:
            return False

    @classmethod
    def clear_flag_file(cls) -> bool:
        """
        Remove the flag file if it exists.

        Returns:
            True if file was removed or didn't exist
        """
        try:
            if cls.FLAG_FILE.exists():
                cls.FLAG_FILE.unlink()
            return True
        except Exception:
            return False
