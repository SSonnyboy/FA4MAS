#!/usr/bin/env sh
set -eu

set -a && . ./.env && set +a

mkdir -p logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
PIDS_FILE="logs/pids_${RUN_TS}.txt"

JOB_COUNT=0
FOUND_CONFIG=0

for CONFIG in configs/*.json; do
  # If the glob didn't match anything, keep compatibility with POSIX sh.
  if [ ! -f "$CONFIG" ]; then
    continue
  fi

  FOUND_CONFIG=1
  CONFIG_BASENAME="$(basename "$CONFIG")"
  if [ "$CONFIG_BASENAME" = "blade.json" ]; then
    continue
  fi

  METHOD="${CONFIG_BASENAME%.json}"
  LOG_FILE="logs/${METHOD}_${RUN_TS}.log"

  nohup python run_experiment.py --config "$CONFIG" >"$LOG_FILE" 2>&1 &
  PID=$!
  JOB_COUNT=$((JOB_COUNT + 1))

  echo "$PID $METHOD $CONFIG $LOG_FILE" >>"$PIDS_FILE"
  echo "[$METHOD] Started in background. PID=$PID Log=$LOG_FILE"
done

if [ "$FOUND_CONFIG" -eq 0 ]; then
  echo "No config files found in configs/."
  exit 1
fi

if [ "$JOB_COUNT" -eq 0 ]; then
  echo "No jobs started (only blade config found)."
  exit 0
fi

echo "Started $JOB_COUNT jobs in background."
echo "PIDs file: $PIDS_FILE"
