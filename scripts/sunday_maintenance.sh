#!/bin/bash
# sunday_maintenance.sh — weekly IBKR data refresh + ML model retrain.
#
# Replaces the manual "run update_market_data on a hunch" workflow with a
# reliable weekly job. Runs in two phases:
#   1. Refresh the parquet store from IBKR (gold standard data source).
#   2. Retrain any stale ML models from the freshly-updated parquet.
#
# Schedule (crontab_regions.txt):
#   0 2 * * 0 <repo>/scripts/sunday_maintenance.sh >> <repo>/logs/sunday_maintenance.log 2>&1
#
# IBKR is the single source of truth for all market data. yfinance is DEPRECATED
# and is not used for training.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
LOG_DIR="$REPO_ROOT/execution/logs"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Load .env for Telegram alerting (same as run_region.sh).
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    . "$REPO_ROOT/.env"
    set +a
fi

curl_alert() {
    [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && return 0
    [ -z "${TELEGRAM_CHAT_ID:-}" ] && return 0
    local text="[$2] $1

$3"
    curl -s -o /dev/null --max-time 10 \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${text}" || true
}

echo "[$(ts)] ============================================================"
echo "[$(ts)] SUNDAY MAINTENANCE: IBKR data refresh + ML retrain"
echo "[$(ts)] ============================================================"
echo "[$(ts)] IBKR is the gold standard for market data. yfinance is deprecated."

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Phase 1: Refresh parquet store from IBKR
# ---------------------------------------------------------------------------
echo "[$(ts)] Phase 1: Refreshing parquet store from IBKR..."
"$PYTHON" -m algos.common.update_market_data --source ibkr --ibkr-port 4002
UPDATE_RC=$?
echo "[$(ts)] update_market_data exit code: $UPDATE_RC"

if [ "$UPDATE_RC" -ne 0 ]; then
    echo "[$(ts)] WARNING: parquet refresh exited $UPDATE_RC. Retrain may use stale data."
    curl_alert "Sunday data refresh failed" "update_market_data exited $UPDATE_RC. ML retrain skipped to avoid training on stale parquet. Check the log." "warning"
    echo "[$(ts)] ABORTING: refusing to retrain on stale parquet after failed IBKR refresh."
    exit "$UPDATE_RC"
fi

# ---------------------------------------------------------------------------
# Phase 2: Retrain stale ML models from the freshly-updated parquet
# ---------------------------------------------------------------------------
echo "[$(ts)] Phase 2: Retraining stale ML models from IBKR-sourced parquet..."
"$PYTHON" "$SCRIPT_DIR/retrain_models.py" --max-age-days 30
RETRAIN_RC=$?
echo "[$(ts)] retrain_models.py exit code: $RETRAIN_RC"

if [ "$RETRAIN_RC" -ne 0 ]; then
    echo "[$(ts)] WARNING: retrain exited $RETRAIN_RC. Some models may not have updated."
    curl_alert "Sunday ML retrain had failures" "retrain_models.py exited $RETRAIN_RC. Some models may not have been refreshed. Check the log." "warning"
else
    echo "[$(ts)] Retrain complete (all targets fresh or successfully retrained)."
fi

echo "[$(ts)] Sunday maintenance complete."
