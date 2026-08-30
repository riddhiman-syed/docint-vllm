#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for name in api vllm qdrant; do
    pidfile="$RUN_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (pid $pid)..."
            kill "$pid"
        else
            echo "$name not running (stale pid file)"
        fi
        rm -f "$pidfile"
    else
        echo "$name: no pid file, nothing to stop"
    fi
done
