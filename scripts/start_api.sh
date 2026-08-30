#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

API_PORT="${API_PORT:-8080}"

if is_running api; then
    echo "API already running (pid $(cat "$RUN_DIR/api.pid"))"
    exit 0
fi

if already_up "http://localhost:$API_PORT/health"; then
    echo "API is already up at localhost:$API_PORT (not started by this script — leaving it alone)."
    echo "stop_all.sh won't manage it either, since there's no PID on record for it."
    exit 0
fi

cd "$REPO_ROOT"
start_bg api "$REPO_ROOT/logs/api.log" \
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT"

wait_for_http "http://localhost:$API_PORT/health" "API" 30
