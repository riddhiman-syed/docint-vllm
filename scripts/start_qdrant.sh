#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Adjust QDRANT_BIN if your binary lives elsewhere.
QDRANT_BIN="${QDRANT_BIN:-$REPO_ROOT/qdrant/qdrant}"
QDRANT_STORAGE="${QDRANT_STORAGE:-$REPO_ROOT/qdrant_storage}"

if is_running qdrant; then
    echo "Qdrant already running (pid $(cat "$RUN_DIR/qdrant.pid"))"
    exit 0
fi

if already_up "http://localhost:6333/collections"; then
    echo "Qdrant is already up at localhost:6333 (not started by this script — leaving it alone)."
    echo "stop_all.sh won't manage it either, since there's no PID on record for it."
    exit 0
fi

if [ ! -x "$QDRANT_BIN" ]; then
    echo "Qdrant binary not found or not executable at $QDRANT_BIN"
    echo "Set QDRANT_BIN to the correct path, e.g.: QDRANT_BIN=/path/to/qdrant ./scripts/start_qdrant.sh"
    exit 1
fi

mkdir -p "$QDRANT_STORAGE"
QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
    start_bg qdrant "$REPO_ROOT/logs/qdrant.log" "$QDRANT_BIN"

wait_for_http "http://localhost:6333/collections" "Qdrant" 30
