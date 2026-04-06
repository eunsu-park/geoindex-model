#!/bin/bash
# Run validation, MCD, and attention analysis for a given experiment.
#
# Usage:
#   ./run.sh <config_name> [epoch]
#
# Examples:
#   ./run.sh in2d_out24h_A1 best
#   ./run.sh in1d_out12h 10
#   ./run.sh in3d_out24h          # defaults to epoch=best

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <config_name> [epoch]"
    echo "  config_name: Hydra config name (e.g., in2d_out24h_A1)"
    echo "  epoch:       Checkpoint epoch (default: best)"
    exit 1
fi

config_name=$1
epoch=${2:-best}

echo "Config: $config_name"
echo "Epoch:  $epoch"
echo ""

python scripts/validate.py --config-name=$config_name validation.epoch=$epoch
python analysis/monte_carlo_dropout.py --config-name=$config_name mcd.epoch=$epoch
python analysis/run_attention.py --config-name=$config_name attention.epoch=$epoch
