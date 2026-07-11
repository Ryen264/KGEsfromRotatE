#!/bin/sh
# Stop the background emb_tsne job started by nohup/emb_tsne/run.sh.
#
# Usage:
#   ./nohup/emb_tsne/stop.sh

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PID_FILE="$ROOT/nohup/emb_tsne/job.pid"
META_FILE="$ROOT/nohup/emb_tsne/job.meta"

if [ ! -f "$PID_FILE" ]; then
    echo "No job pid file found ($PID_FILE)."
    exit 1
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running. Cleaning up pid file."
    rm -f "$PID_FILE" "$META_FILE"
    exit 0
fi

echo "Stopping pid $PID ..."
kill "$PID" 2>/dev/null || true

i=0
while kill -0 "$PID" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 10 ]; then
        echo "Force stopping pid $PID and child processes ..."
        pkill -P "$PID" 2>/dev/null || true
        kill -9 "$PID" 2>/dev/null || true
        break
    fi
    sleep 1
done

rm -f "$PID_FILE" "$META_FILE"
echo "Stopped."
