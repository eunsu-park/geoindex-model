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
echo ""

OPTS="--config-name=$config_name environment.device=$device"

python scripts/validate.py $OPTS validation.epoch=$epoch
python analysis/monte_carlo_dropout.py $OPTS mcd.epoch=$epoch
python analysis/run_attention.py $OPTS attention.epoch=$epoch
