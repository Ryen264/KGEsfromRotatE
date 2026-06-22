#!/bin/sh
# Start visualization training in the background with nohup.
#
# Usage:
#   ./nohup/run.sh configs/ComplEx_WN18RR_ce.json
#   ./nohup/run.sh configs/ComplEx_WN18RR_ce.json --gpu 0 --display-epochs 100 --no-show
#
# Logs:  nohup/logs/<config-stem>_<timestamp>.log
# PID:   nohup/job.pid

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOHUP_DIR="$ROOT/nohup"
LOG_DIR="$NOHUP_DIR/logs"
PID_FILE="$NOHUP_DIR/job.pid"
META_FILE="$NOHUP_DIR/job.meta"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Job already running (pid $OLD_PID). Stop it first: ./nohup/stop.sh"
        exit 1
    fi
    rm -f "$PID_FILE" "$META_FILE"
fi

CONFIG="${1:-configs/ComplEx_WN18RR_ce.json}"
shift 2>/dev/null || true

if [ ! -f "$ROOT/$CONFIG" ] && [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

if [ -f "$ROOT/.venv/bin/activate" ]; then
    . "$ROOT/.venv/bin/activate"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_STEM="$(basename "$CONFIG" .json)"
LOG_FILE="$LOG_DIR/${CONFIG_STEM}_${TIMESTAMP}.log"
LATEST_LINK="$LOG_DIR/latest.log"

cd "$ROOT"

CMD="python -u visualization/main.py $CONFIG --no-show $*"
echo "Starting: $CMD"
echo "Log: $LOG_FILE"

nohup sh -c "$CMD" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$PID_FILE"
{
    echo "pid=$PID"
    echo "started=$(date -Iseconds 2>/dev/null || date)"
    echo "config=$CONFIG"
    echo "log=$LOG_FILE"
    echo "cmd=$CMD"
} > "$META_FILE"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

echo "Started pid $PID"
echo "Check status: ./nohup/check.sh"
echo "Tail log:     tail -f $LOG_FILE"
