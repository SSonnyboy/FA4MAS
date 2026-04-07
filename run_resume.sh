#!/usr/bin/env sh
set -eu

if [ -f ./.env ]; then
  set -a && . ./.env && set +a
fi

mkdir -p logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
PIDS_FILE="logs/pids_resume_${RUN_TS}.txt"

# 可选参数：指定续跑的 run_id（例如 20260406_235712）。
RUN_ID="${1:-}"

METHODS="all_at_once baseline chief echo"
JOB_COUNT=0

for METHOD in $METHODS; do
  CONFIG="configs/${METHOD}.json"
  if [ ! -f "$CONFIG" ]; then
    echo "[$METHOD] Skip: config not found ($CONFIG)"
    continue
  fi

  LOG_FILE="logs/${METHOD}_resume_${RUN_TS}.log"
  CMD="python run_experiment.py --config \"$CONFIG\" --resume"
  if [ -n "$RUN_ID" ]; then
    CMD="$CMD --run-id \"$RUN_ID\""
  fi

  # shellcheck disable=SC2086
  nohup sh -c "$CMD" >"$LOG_FILE" 2>&1 &
  PID=$!
  JOB_COUNT=$((JOB_COUNT + 1))

  echo "$PID $METHOD $CONFIG $LOG_FILE resume run_id=${RUN_ID:-auto_latest}" >>"$PIDS_FILE"
  echo "[$METHOD] Resume started in background. PID=$PID Log=$LOG_FILE"
done

if [ "$JOB_COUNT" -eq 0 ]; then
  echo "No resume jobs started."
  exit 1
fi

echo "Started $JOB_COUNT resume jobs in background."
echo "PIDs file: $PIDS_FILE"
if [ -n "$RUN_ID" ]; then
  echo "Run ID: $RUN_ID"
else
  echo "Run ID: auto latest per method/dataset"
fi
