#!/usr/bin/env bash
# Shared helpers, sourced by the other scripts/*.sh files.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_ROOT/.run"
mkdir -p "$RUN_DIR" "$REPO_ROOT/logs"

# wait_for_http <url> <label> <timeout_seconds>
wait_for_http() {
    local url="$1" label="$2" timeout="${3:-60}"
    local waited=0
    echo "Waiting for $label at $url ..."
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$timeout" ]; then
            echo "Timed out after ${timeout}s waiting for $label at $url"
            return 1
        fi
        if [ $((waited % 20)) -eq 0 ]; then
            echo "  ... still waiting for $label (${waited}s elapsed)"
        fi
    done
    echo "$label is up (took ${waited}s)"
}

# start_bg <name> <logfile> <command...>
# Starts a background process, records its PID to $RUN_DIR/<name>.pid
start_bg() {
    local name="$1" logfile="$2"
    shift 2
    nohup "$@" > "$logfile" 2>&1 &
    echo $! > "$RUN_DIR/$name.pid"
    echo "Started $name (pid $(cat "$RUN_DIR/$name.pid")), logging to $logfile"
}

is_running() {
    local name="$1"
    local pidfile="$RUN_DIR/$name.pid"
    [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

# already_up <url> — true if something (managed by us or not) is already
# answering at this URL. Used to avoid double-starting a service that's
# already running from a previous session or was started manually.
already_up() {
    curl -sf "$1" > /dev/null 2>&1
}
