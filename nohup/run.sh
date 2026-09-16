#!/bin/bash
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

# Kích hoạt môi trường ảo nếu có
if [ -f "$ROOT/.venv/bin/activate" ]; then
    . "$ROOT/.venv/bin/activate"
fi

# Tự động điều hướng: 
# Nếu tham số đầu tiên là file .json -> chạy visualization/main.py
# Ngược lại -> chạy thẳng main.py ở root
if [[ "$1" == *.json ]]; then
    CONFIG="$1"
    shift
    if [ ! -f "$ROOT/$CONFIG" ] && [ ! -f "$CONFIG" ]; then
        echo "Config not found: $CONFIG"
        exit 1
    fi
    CMD="python -u visualization/main.py $CONFIG --no-show $*"
    CONFIG_STEM="$(basename "$CONFIG" .json)"
else
    if [ -z "$1" ]; then
        echo "Usage:"
        echo "  ./nohup/run.sh <config.json> [args]"
        echo "  ./nohup/run.sh --do_train [args]"
        exit 1
    fi
    CMD="python -u main.py $*"
    CONFIG_STEM="main_cli"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${CONFIG_STEM}_${TIMESTAMP}.log"
LATEST_LINK="$LOG_DIR/latest.log"

cd "$ROOT"

echo "Starting: $CMD"
echo "Log: $LOG_FILE"

nohup sh -c "$CMD" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$PID_FILE"
{
    echo "pid=$PID"
    echo "started=$(date -Iseconds 2>/dev/null || date)"
    if [[ -n "$CONFIG" ]]; then
        echo "config=$CONFIG"
    fi
    echo "log=$LOG_FILE"
    echo "cmd=$CMD"
} > "$META_FILE"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

echo "Started pid $PID"
echo "Check status: ./nohup/check.sh"
echo "Tail log:     tail -f $LOG_FILE"