#!/bin/bash
# Run validation and analysis for local experiment
# Usage: ./run_local.sh [epoch]
# Example: ./run_local.sh 1

set -e  # Exit on error

# Default epoch
EPOCH=${1:-1}

echo "========================================"
echo "Running local experiment analysis"
echo "Epoch: $EPOCH"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Validation
echo ""
echo "[1/4] Running Validation..."
echo "----------------------------------------"
python scripts/validate.py --config-name=local validation.epoch=$EPOCH

# 2. Attention Analysis
echo ""
echo "[2/4] Running Attention Analysis..."
echo "----------------------------------------"
python analysis/run_attention.py --config-name=local attention.epoch=$EPOCH

# 3. Saliency Analysis
echo ""
echo "[3/4] Running Saliency Analysis..."
echo "----------------------------------------"
python analysis/run_saliency.py --config-name=local saliency.epoch=$EPOCH

# 4. Monte Carlo Dropout
echo ""
echo "[4/4] Running Monte Carlo Dropout..."
echo "----------------------------------------"
python analysis/monte_carlo_dropout.py --config-name=local mcd.epoch=$EPOCH


echo ""
echo "========================================"
echo "All analyses completed for epoch $EPOCH"
echo "========================================"
echo ""
echo "All analyses completed. Results saved under experiment directory."
