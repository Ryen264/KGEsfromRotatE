#!/bin/sh
# Check status of the background job started by nohup/run.sh.
#
# Usage:
#   ./nohup/check.sh
#   ./nohup/check.sh -f    # follow latest log

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/nohup/job.pid"
META_FILE="$ROOT/nohup/job.meta"
LOG_DIR="$ROOT/nohup/logs"
LATEST_LINK="$LOG_DIR/latest.log"

if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
    if [ -L "$LATEST_LINK" ] || [ -f "$LATEST_LINK" ]; then
        tail -f "$LATEST_LINK"
    else
        echo "No log file found under $LOG_DIR"
        exit 1
    fi
    exit 0
fi

echo "=== KGE nohup job status ==="

if [ ! -f "$PID_FILE" ]; then
    echo "Status: not running (no pid file)"
    if [ -L "$LATEST_LINK" ] || [ -f "$LATEST_LINK" ]; then
        echo "Latest log: $LATEST_LINK"
    fi
    exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
    echo "Status: running"
    echo "PID:    $PID"
else
    echo "Status: stale pid file (process $PID not running)"
    exit 1
fi

if [ -f "$META_FILE" ]; then
    echo "--- job meta ---"
    cat "$META_FILE"
fi

if [ -L "$LATEST_LINK" ] || [ -f "$LATEST_LINK" ]; then
    echo "--- last 20 log lines ($LATEST_LINK) ---"
    tail -n 20 "$LATEST_LINK"
fi
