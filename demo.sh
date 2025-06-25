#!/usr/bin/env bash
# ------------------------------------------------------------
# run_experiments.sh – batch‑runner for ContinuousSpace agents
# ------------------------------------------------------------
# This script launches a handful of preset experiments so you
# can reproduce paper‑ready curves with one command.
#
# 1. Feel free to edit the `experiments` array below to add or
#    remove configurations.
# 2. Make it executable once:  chmod +x run_experiments.sh
# 3. Run it:                   ./run_experiments.sh
# ------------------------------------------------------------

set -euo pipefail

# Change this if your Python binary lives elsewhere
PYTHON_BIN="python"

# Each element is a complete CLI argument string for train.py
experiments=(
  "--agent_type DQN --episodes 400 --steps 100 --gamma 0.95 --lr 1e-3 --sa --br 4 --tol 300 --env 1"
  "--agent_type DuelingDQN --episodes 500 --steps 100 --gamma 0.99 --lr 1e-2 --sa --br 4 --tol 300 --env 0"
  "--agent_type DQN --episodes 500 --steps 100 --gamma 0.95 --lr 1e-3 --sa --br 4 --tol 300 --env 1"
  "--agent_type DuelingDQN --episodes 500 --steps 250 --gamma 0.99 --lr 5e-3 --sa --br 4 --tol 300 --env 1"
)

# Timing – start global stopwatch

start_time=$(date +%s)

for params in "${experiments[@]}"; do
  echo "============================================================"
  echo "Running: $PYTHON_BIN train.py $params"
  echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"
  $PYTHON_BIN train.py $params
  echo "Finished  $(date '+%Y-%m-%d %H:%M:%S')"
  echo
done

echo "All experiments completed successfully."


# Show total elapsed wall‑clock time

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Format as HH:MM:SS for readability
printf -v elapsed_hms '%02d:%02d:%02d' \
  $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

echo "Total runtime: $elapsed_hms (hh:mm:ss)"