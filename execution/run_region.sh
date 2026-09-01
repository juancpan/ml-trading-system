#!/bin/bash
# Region-based IBKR trading script with kill-switch and preflight gating.
# Usage: ./run_region.sh REGION
# Example: ./run_region.sh EUROPE
#
# Gate order (each gate must pass before the next):
#   1. Kill-switch sentinels — if any of {KILL_SWITCH_ACTIVE, SOFT_HALT_ACTIVE,
#      DAILY_MOVE_ACTIVE} exist, behaviour depends on which:
#        - KILL_SWITCH_ACTIVE → refuse to trade until the file is removed.
#        - SOFT_HALT_ACTIVE   → still run main.py (it enforces entry blocks itself).
#        - DAILY_MOVE_ACTIVE  → skip trading for the day, exit 0.
#   2. nav_quick.py — must return a positive NAV.
#   3. preflight_check.py --nav <nav> --with-ibkr — must pass.
#   4. main.py --region <REGION> — actual trading.
#
# Any failure of gates 1–3 → no trading and (if SKIP_ALERTS!=1) Telegram alert.

set -u

REGION="${1:-ALL}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
# Interpreter: override via PYTHON env var, else resolve from PATH.
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/cron_${REGION}_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Load .env (gitignored). Cron does NOT inherit the interactive shell's
# exports, so without this the Telegram credentials are absent and alerts
# silently no-op — which is exactly what masked the May 2026 outage.
# `set -a` exports every variable assigned while sourcing.
# ---------------------------------------------------------------------------
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$REPO_ROOT/.env"
    set +a
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Interpreter-independent Telegram alert. Used as a fallback when the Python
# alert path cannot run (e.g. the interpreter/volume is unavailable — the
# Era-1 failure mode). Token is passed via --data-urlencode (NOT in argv /
# the URL path) so it never appears in `ps` output or shell history.
curl_alert() {
    # curl_alert "TITLE" "MESSAGE" "SEVERITY"
    [ "${SKIP_ALERTS:-0}" = "1" ] && return 0
    [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && return 0
    [ -z "${TELEGRAM_CHAT_ID:-}" ] && return 0
    local text="[$3] $1

$2"
    curl -s -o /dev/null \
        --max-time 10 \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${text}" >> "$LOG_FILE" 2>&1 || true
}

alert() {
    # alert "TITLE" "MESSAGE" "SEVERITY"
    if [ "${SKIP_ALERTS:-0}" = "1" ]; then
        return 0
    fi
    # Primary path: Python alerting (richer logging). If the interpreter is
    # healthy this delivers; if it fails for any reason, fall back to curl.
    if ! "$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from alerting import send_alert
send_alert('$1', '$2', '$3')
" >> "$LOG_FILE" 2>&1; then
        curl_alert "$1" "$2" "$3"
    fi
}

log "=== IBKR Trading Session ==="
log "Region: $REGION"
log "Working dir: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Per-region single-instance lock
# ---------------------------------------------------------------------------
# Prevents two concurrent runs of the SAME region (e.g. a hung previous
# session + a fresh cron fire) from launching two main.py processes that would
# fight over state files / the IBKR client id / the ZMQ port. `mkdir` is
# atomic on all POSIX filesystems (macOS has no `flock`), so it is a reliable
# mutex. A stale lock (owning PID no longer alive) is reclaimed automatically.
#
# BUGFIX (2026-06-16): the original release used `rmdir`, which only removes
# EMPTY directories. Because we write a `pid` file INTO the lock dir, `rmdir`
# always failed ("Directory not empty") and the failure was hidden by
# `2>/dev/null || true`. Result: the lock was NEVER released on any exit path,
# every run leaked its .lock_<REGION>/, and the next day's stale-reclaim (also
# using rmdir) could not clear it -> `mkdir` failed -> the session aborted with
# exit 6 and did NOT trade (see logs/cron_{US,CANADA}_20260616_*.log). The fix:
#   * release with `rm -rf` (dir + pid file), guarded so it can only ever touch
#     a path of the form "$SCRIPT_DIR/.lock_<REGION>" (never empty/"/"/$SCRIPT_DIR);
#   * surface (log) any release/reclaim failure instead of swallowing it;
#   * keep `mkdir` as the atomic acquire and the kill -0 stale check.
LOCK_DIR="$SCRIPT_DIR/.lock_${REGION}"

release_lock() {
    # Defensive: never rm -rf anything that isn't exactly our computed lock dir.
    case "$LOCK_DIR" in
        "$SCRIPT_DIR"/.lock_*) ;;  # expected shape — safe to remove
        *)
            log "REFUSING to release suspicious LOCK_DIR='$LOCK_DIR' (not \$SCRIPT_DIR/.lock_*)."
            return 0
            ;;
    esac
    [ -z "$LOCK_DIR" ] && return 0
    if [ -d "$LOCK_DIR" ]; then
        if ! rm -rf "$LOCK_DIR"; then
            log "WARNING: failed to release lock dir '$LOCK_DIR'. A future run may have to reclaim it."
        fi
    fi
}

acquire_lock() {
    echo "$$" > "$LOCK_DIR/pid"
    trap release_lock EXIT
    log "Lock acquired for region $REGION (PID $$)."
}

if mkdir "$LOCK_DIR" 2>/dev/null; then
    acquire_lock
else
    OWNER_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$OWNER_PID" ] && kill -0 "$OWNER_PID" 2>/dev/null; then
        log "LOCK HELD: region $REGION already running as PID $OWNER_PID. Refusing to start a second instance."
        alert "Region already running" "Region=$REGION skipped: a prior session (PID $OWNER_PID) is still active and holds the lock. No second instance launched." "warning"
        exit 6
    else
        # Lock-age alarm: a dead-owner lock older than any plausible session
        # (sessions run at most ~7h) usually means a previous run leaked it
        # (crash/SIGKILL, or — historically — the rmdir release bug). Reclaim
        # either way, but alert distinctly so a recurring leak is visible.
        LOCK_AGE_SECS=$(( $(date +%s) - $(stat -f %m "$LOCK_DIR" 2>/dev/null || echo "$(date +%s)") ))
        if [ "$LOCK_AGE_SECS" -gt 28800 ]; then  # > 8h
            log "Stale lock for region $REGION is ${LOCK_AGE_SECS}s old (>8h) with dead owner '${OWNER_PID:-none}'. Likely a leaked lock."
            alert "Stale lock reclaimed" "Region=$REGION reclaimed a leaked lock (${LOCK_AGE_SECS}s old, dead PID '${OWNER_PID:-none}'). Trading proceeds, but investigate why the prior session did not release its lock." "warning"
        fi
        log "Stale lock for region $REGION (owner PID '${OWNER_PID:-none}' not alive). Reclaiming."
        release_lock
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            acquire_lock
        else
            log "Could not acquire lock for region $REGION after reclaiming stale lock. Aborting."
            alert "Lock acquisition failed" "Region=$REGION could not acquire run lock even after reclaiming a stale lock at '$LOCK_DIR'. No trading. Investigate filesystem permissions/leftover lock." "critical"
            exit 6
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Gate 0 — Environment availability (interpreter bootstrap)
# ---------------------------------------------------------------------------
# If the interpreter cannot even bootstrap ("Fatal Python error: ... No
# module named 'encodings'" — e.g. a broken env or OS access change at cron
# time), that is NOT a strategy/preflight failure — it is an environment
# failure, and it must be reported distinctly (exit 5) so the daily/monthly
# audits don't misclassify it. The alert here uses curl_alert directly
# because the Python alert path is exactly what may be broken.

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    log "ENVIRONMENT UNAVAILABLE: interpreter not found or not executable (${PYTHON:-unset})."
    log "This is NOT a strategy/preflight failure. Region=$REGION not run."
    curl_alert "Environment unavailable" "Region=$REGION: interpreter missing/not executable. No trading. Check the PYTHON env var / PATH for cron." "critical"
    exit 5
fi

if ! "$PYTHON" -c "import encodings, sys" >/dev/null 2>>"$LOG_FILE"; then
    log "ENVIRONMENT UNAVAILABLE: interpreter failed to bootstrap (stdlib import failed)."
    log "Symptom class: 'No module named encodings' — environment issue, NOT a code/preflight failure."
    log "Region=$REGION not run."
    curl_alert "Environment unavailable" "Region=$REGION: Python interpreter failed to bootstrap (stdlib unavailable). No trading. Check the interpreter environment." "critical"
    exit 5
fi
log "Gate 0 PASSED (interpreter bootstrap ok)."

# ---------------------------------------------------------------------------
# Gate 1 — Kill-switch sentinels
# ---------------------------------------------------------------------------

if [ -f "$SCRIPT_DIR/KILL_SWITCH_ACTIVE" ]; then
    log "HARD KILL sentinel present. Refusing to trade."
    log "Contents:"
    cat "$SCRIPT_DIR/KILL_SWITCH_ACTIVE" | tee -a "$LOG_FILE"
    alert "Kill-switch active" "Region=$REGION refused to trade because KILL_SWITCH_ACTIVE sentinel present." "critical"
    exit 2
fi

if [ -f "$SCRIPT_DIR/DAILY_MOVE_ACTIVE" ]; then
    # Daily-move alarm: skip trading for the day.
    SENT_DATE=$(stat -f "%Sm" -t "%Y-%m-%d" "$SCRIPT_DIR/DAILY_MOVE_ACTIVE" 2>/dev/null || echo "unknown")
    TODAY=$(date +%Y-%m-%d)
    if [ "$SENT_DATE" = "$TODAY" ]; then
        log "DAILY_MOVE_ACTIVE sentinel present (today). Skipping trading."
        alert "Daily-move alarm" "Region=$REGION skipped because DAILY_MOVE_ACTIVE sentinel from today." "warning"
        exit 0
    else
        log "DAILY_MOVE_ACTIVE sentinel is from $SENT_DATE (stale). Removing and proceeding."
        rm -f "$SCRIPT_DIR/DAILY_MOVE_ACTIVE"
    fi
fi

if [ -f "$SCRIPT_DIR/SOFT_HALT_ACTIVE" ]; then
    log "SOFT_HALT_ACTIVE sentinel present. main.py will block new ml_signal entries."
    alert "Soft halt active" "Region=$REGION running with ml_signal entries blocked." "warning"
    # Fall through — main.py reads the sentinel itself.
fi

# ---------------------------------------------------------------------------
# Gate 2 — NAV
# ---------------------------------------------------------------------------

NAV=$("$PYTHON" "$SCRIPT_DIR/nav_quick.py" 2>>"$LOG_FILE")
NAV_RC=$?
if [ "$NAV_RC" -ne 0 ] || [ -z "$NAV" ] || [ "$NAV" = "0" ]; then
    # Gate 0 already proved the interpreter/volume are healthy, so a failure
    # here genuinely means account_values.pkl is unreadable/stale/missing —
    # NOT an environment problem. Keep this distinction for §3/§5 audits.
    log "NAV READ FAILED (rc=$NAV_RC, nav=$NAV): account_values.pkl unreadable/stale/NAV non-positive. Aborting before preflight. (Interpreter/volume already verified OK by Gate 0.)"
    alert "NAV unavailable" "Region=$REGION cannot determine NAV from account_values.pkl (file stale/missing/NAV<=0). Trading aborted. Note: environment is healthy (Gate 0 passed); this is a state-file issue." "critical"
    exit 3
fi
log "NAV: $NAV"

# ---------------------------------------------------------------------------
# Gate 3 — Preflight
# ---------------------------------------------------------------------------

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
    log "SKIP_PREFLIGHT=1; bypassing preflight (NOT RECOMMENDED)."
else
    log "Running preflight_check.py --nav $NAV --with-ibkr --region $REGION"
    "$PYTHON" "$SCRIPT_DIR/preflight_check.py" --nav "$NAV" --with-ibkr --region "$REGION" >> "$LOG_FILE" 2>&1
    PRE_RC=$?
    if [ "$PRE_RC" -ne 0 ]; then
        log "Preflight FAILED (rc=$PRE_RC). Refusing to trade."
        alert "Preflight failed" "Region=$REGION preflight returned $PRE_RC. See $LOG_FILE." "critical"
        exit 4
    fi
    log "Preflight PASSED."
fi

# ---------------------------------------------------------------------------
# Gate 4 — main.py
# ---------------------------------------------------------------------------

log "Launching main.py --region $REGION"
"$PYTHON" main.py --region "$REGION" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

log "main.py finished. Exit code: $EXIT_CODE"

# ---------------------------------------------------------------------------
# Failure detection (closes the "cron looked green" P0 gap)
# ---------------------------------------------------------------------------
# Two distinct silent-failure classes were masking real outages:
#   1. main.py exits non-zero (e.g. ZMQ bind collision -> exit 1) but the
#      session still appeared green because the cron line ended with the
#      kill_switch exit 0.
#   2. main.py exits 0 yet printed a fatal/Traceback to the run log during
#      teardown (e.g. the ml_dtypes AttributeError on 2026-06-08).
# We now alert on BOTH: a non-zero exit code, AND any fatal marker in the log
# even when the exit code is 0.
if [ "$EXIT_CODE" -ne 0 ]; then
    log "main.py FAILED (exit $EXIT_CODE). Trading did NOT complete normally."
    alert "Trading run failed" "Region=$REGION: main.py exited $EXIT_CODE. Trading likely did NOT occur. See $LOG_FILE." "critical"
fi

# Scan the run log for fatal markers regardless of exit code. Limit to this
# run's log file. Exclude benign IBKR farm/info lines that contain 'error'.
if grep -Eq 'Traceback \(most recent call last\)|Fatal Python error|zmq\.error\.ZMQError|AttributeError|ModuleNotFoundError' "$LOG_FILE"; then
    MARKER=$(grep -Eo 'Traceback \(most recent call last\)|Fatal Python error|zmq\.error\.ZMQError|AttributeError|ModuleNotFoundError' "$LOG_FILE" | sort -u | tr '\n' ',' )
    log "FATAL MARKER(S) detected in run log (exit code was $EXIT_CODE): ${MARKER%,}"
    alert "Fatal marker in run log" "Region=$REGION: detected [${MARKER%,}] in $LOG_FILE (main.py exit=$EXIT_CODE). Investigate — a green exit code may be hiding a real failure." "critical"
fi

# Post-run kill-switch evaluation. Reads latest equity_history.parquet,
# writes sentinels if a tier fires. Non-zero exit codes here are
# informational; we do not act on them directly.
log "Post-run kill_switch evaluation..."
"$PYTHON" "$SCRIPT_DIR/kill_switch.py" >> "$LOG_FILE" 2>&1
KS_RC=$?
log "kill_switch.py exit code: $KS_RC"
if [ "$KS_RC" = "2" ]; then
    alert "Kill-switch fired (post-run)" "Region=$REGION post-run kill_switch tier=hard_kill. Sentinel written." "critical"
elif [ "$KS_RC" = "1" ]; then
    alert "Soft halt fired (post-run)" "Region=$REGION post-run kill_switch tier=soft_halt. Sentinel written." "warning"
fi

# -----------------------------------------------------------------------------
# Post-run observability: daily PnL attribution + per-model hit-rate, then
# regenerate the Revision Health dashboard.
#
# attribution.py is the SOLE producer of attribution.db (tables
# attribution_daily + model_hit_rate); revision_dashboard.py is a read-only
# renderer. Previously nothing ran the producer, so the dashboard rendered
# fresh HTML over a frozen DB. We run the producer here, per-region, then
# re-render so the HTML reflects rows just written.
#
# Exit codes from attribution.py (see its main()):
#   0 ok | 1 hard fail | 2 soft skip (no equity for region/day) | 3 hit-rates empty
log "Post-run attribution (region=$REGION)..."
"$PYTHON" "$SCRIPT_DIR/attribution.py" --region "$REGION" --allow-missing >> "$LOG_FILE" 2>&1
ATTR_RC=$?
log "attribution.py exit code: $ATTR_RC"
case "$ATTR_RC" in
    0) ;;  # success, nothing to say
    2) log "attribution: no equity history for $REGION today — soft skip." ;;
    3) alert "Hit-rates empty" "Region=$REGION attribution persisted but per-model hit-rates were EMPTY (yfinance unavailable/empty?). model_hit_rate not advanced." "warning" ;;
    *) alert "Attribution failed" "Region=$REGION attribution.py exited $ATTR_RC. attribution.db may be stale; dashboard will show stale data." "warning" ;;
esac

log "Regenerating revision dashboard..."
"$PYTHON" "$SCRIPT_DIR/revision_dashboard.py" >> "$LOG_FILE" 2>&1
DASH_RC=$?
log "revision_dashboard.py exit code: $DASH_RC"
if [ "$DASH_RC" != "0" ]; then
    alert "Dashboard render failed" "Region=$REGION revision_dashboard.py exited $DASH_RC. See $LOG_FILE." "warning"
fi

# Keep only last 30 days of cron logs
find "$LOG_DIR" -name "cron_*.log" -mtime +30 -delete 2>/dev/null

exit $EXIT_CODE
