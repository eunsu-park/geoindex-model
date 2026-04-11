#!/bin/bash
# Run validation, MCD, and attention analysis for a given experiment.
#
# Usage:
#   ./run.sh <config_name> [epoch] [device]
#
# Examples:
#   ./run.sh in2d_out24h_A1 best        # GPU (default: cuda)
#   ./run.sh in1d_out12h 10 mps         # Mac MPS
#   ./run.sh in3d_out24h best cpu       # CPU

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <config_name> [epoch] [device]"
    echo "  config_name: Hydra config name (e.g., in2d_out24h_A1)"
    echo "  epoch:       Checkpoint epoch (default: best)"
    echo "  device:      cuda, mps, or cpu (default: cuda)"
    exit 1
fi

config_name=$1
epoch=${2:-best}
device=${3:-cuda}

echo "Config: $config_name"
echo "Epoch:  $epoch"
echo "Device: $device"
echo "Running validation, MCD, attention in parallel..."
echo ""

OPTS="--config-name=$config_name environment.device=$device"
LOG_DIR="$HOME/tmp/${config_name}/logs"
mkdir -p "$LOG_DIR"

python scripts/validate.py $OPTS validation.epoch=$epoch \
    > "$LOG_DIR/validation.log" 2>&1 &
PID_VAL=$!

python analysis/monte_carlo_dropout.py $OPTS mcd.epoch=$epoch \
    > "$LOG_DIR/mcd.log" 2>&1 &
PID_MCD=$!

python analysis/run_attention.py $OPTS attention.epoch=$epoch \
    > "$LOG_DIR/attention.log" 2>&1 &
PID_ATT=$!

echo "PIDs: validation=$PID_VAL, mcd=$PID_MCD, attention=$PID_ATT"
echo "Logs: $LOG_DIR/"
echo ""

# Wait and report results
FAIL=0
for name_pid in "validation:$PID_VAL" "mcd:$PID_MCD" "attention:$PID_ATT"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    wait "$pid" && code=0 || code=$?
    if [ $code -eq 0 ]; then
        echo "[OK]   $name (exit $code)"
    else
        echo "[FAIL] $name (exit $code) — see $LOG_DIR/${name}.log"
        FAIL=1
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "All tasks completed successfully."
else
    echo "Some tasks failed. Check logs in $LOG_DIR/"
fi
