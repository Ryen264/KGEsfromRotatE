#!/bin/sh
# Train and evaluate RotatE on FB15k-237 in the background with nohup.
#
# Hyperparameters match best_config.sh (ICLR 2019 paper settings).
#
# Usage:
#   ./nohup/run_rotate_fb15k237.sh           # GPU 0, save id 0 (CPU if CUDA unavailable)
#   ./nohup/run_rotate_fb15k237.sh 1         # GPU 1
#   ./nohup/run_rotate_fb15k237.sh 0 run1    # GPU 0, save id run1
#   ./nohup/run_rotate_fb15k237.sh cpu       # force CPU
#
# Logs:  nohup/logs/RotatE_FB15k-237_<save_id>_<timestamp>.log
# PID:   nohup/job.pid
# Model: models/RotatE_FB15k-237_<save_id>/

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOHUP_DIR="$ROOT/nohup"
LOG_DIR="$NOHUP_DIR/logs"
PID_FILE="$NOHUP_DIR/job.pid"
META_FILE="$NOHUP_DIR/job.meta"

FORCE_CPU=0
if [ "${1:-}" = "cpu" ]; then
    FORCE_CPU=1
    GPU=""
    SAVE_ID="${2:-0}"
else
    GPU="${1:-0}"
    SAVE_ID="${2:-0}"
fi

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Job already running (pid $OLD_PID). Stop it first: ./nohup/stop.sh"
        exit 1
    fi
    rm -f "$PID_FILE" "$META_FILE"
fi

if [ -f "$ROOT/.venv/bin/activate" ]; then
    . "$ROOT/.venv/bin/activate"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/RotatE_FB15k-237_${SAVE_ID}_${TIMESTAMP}.log"
LATEST_LINK="$LOG_DIR/latest.log"
SAVE_PATH="models/RotatE_FB15k-237_${SAVE_ID}"

cd "$ROOT"

CUDA_FLAG=""
DEVICE="cpu"
if [ "$FORCE_CPU" -eq 0 ]; then
  if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_FLAG="--cuda"
    DEVICE="gpu:${GPU}"
  else
    echo "WARNING: CUDA not available (driver too old or no GPU). Falling back to CPU."
    echo "         Training will be much slower. Use './nohup/run_rotate_fb15k237.sh cpu' to skip this check."
    echo "         Or install a PyTorch build matching your NVIDIA driver: https://pytorch.org"
    echo ""
  fi
fi

CUDA_ENV=""
if [ -n "$GPU" ] && [ -n "$CUDA_FLAG" ]; then
  CUDA_ENV="CUDA_VISIBLE_DEVICES=${GPU} "
fi

CMD="${CUDA_ENV}python -u codes/run.py --do_train ${CUDA_FLAG} --do_valid --do_test \
  --data_path data/FB15k-237 --model RotatE \
  -n 256 -b 1024 -d 1000 -g 9.0 -a 1.0 -adv \
  -lr 0.00005 --max_steps 100000 \
  --train_chunk_size 256 \
  -save ${SAVE_PATH} --test_batch_size 16 -de"

echo "Starting RotatE on FB15k-237"
echo "Device:   $DEVICE"
echo "Save id:  $SAVE_ID"
echo "Save to:  $SAVE_PATH"
echo "Log:      $LOG_FILE"
echo "Command:  $CMD"

nohup sh -c "$CMD" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$PID_FILE"
{
    echo "pid=$PID"
    echo "started=$(date -Iseconds 2>/dev/null || date)"
    echo "dataset=FB15k-237"
    echo "model=RotatE"
    echo "save_id=$SAVE_ID"
    echo "save_path=$SAVE_PATH"
    echo "device=$DEVICE"
    echo "log=$LOG_FILE"
    echo "cmd=$CMD"
} > "$META_FILE"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

echo "Started pid $PID"
echo "Check status: ./nohup/check.sh"
echo "Tail log:     tail -f $LOG_FILE"
