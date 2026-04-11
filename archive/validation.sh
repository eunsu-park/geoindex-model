#!/bin/bash
# Batch validation script for experiment analysis
# Usage: ./validation.sh <version> [model_types...]
#
# Arguments:
#   version     - Experiment version number (e.g., 7 for baseline_v7, fusion_v7)
#   model_types - (Optional) Space-separated list of model types to validate
#                 Valid types: baseline, fusion, transformer
#                 Default: baseline fusion
#
# This script runs validation for:
#   - Epochs: 5, 10, 15, 20, 25, best
#   - Models: {model_type}_vN for each specified model type
#
# Examples:
#   ./validation.sh 7                    # Validates baseline_v7 and fusion_v7
#   ./validation.sh 9 transformer        # Validates transformer_v9 only
#   ./validation.sh 8 baseline fusion    # Validates baseline_v8 and fusion_v8
#   ./validation.sh 10 transformer fusion # Validates transformer_v10 and fusion_v10

set -e  # Exit on error

# =============================================================================
# Arguments
# =============================================================================
VERSION=${1:-}

if [ -z "$VERSION" ]; then
    echo "Error: Version number required"
    echo "Usage: ./validation.sh <version> [model_types...]"
    echo ""
    echo "Model types: baseline, fusion, transformer"
    echo ""
    echo "Examples:"
    echo "  ./validation.sh 7                    # baseline_v7, fusion_v7"
    echo "  ./validation.sh 9 transformer        # transformer_v9 only"
    echo "  ./validation.sh 8 baseline fusion    # baseline_v8, fusion_v8"
    exit 1
fi

# Shift to get model types (remaining arguments)
shift

# =============================================================================
# Configuration
# =============================================================================
EPOCHS=(5 10 15 20 25)

# If model types provided, use them; otherwise default to baseline and fusion
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("baseline" "fusion")
fi

# Validate model types
for MODEL in "${MODELS[@]}"; do
    case $MODEL in
        baseline|fusion|transformer|linear|tcn)
            ;;
        *)
            echo "Error: Unknown model type '$MODEL'"
            echo "Valid options: baseline, fusion, transformer, linear, tcn"
            exit 1
            ;;
    esac
done

# =============================================================================
# Run Validation
# =============================================================================
echo "========================================"
echo "Batch Validation for v${VERSION}"
echo "========================================"
echo "Models: ${MODELS[*]/%/_v${VERSION}}"
echo "Epochs: ${EPOCHS[*]}, best"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Experiment-specific model type overrides
# Some experiments were trained with a different model_type than their name suggests
# =============================================================================
get_actual_model_type() {
    local EXP_NAME=$1
    local DEFAULT_MODEL=$2

    # Check for known mismatches (experiment trained with different model type)
    case $EXP_NAME in
        baseline_v11)
            # v11 was accidentally trained as fusion
            echo "fusion"
            ;;
        *)
            echo "$DEFAULT_MODEL"
            ;;
    esac
}

# Run for each epoch
for EPOCH in "${EPOCHS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        EXP_NAME="${MODEL}_v${VERSION}"
        ACTUAL_MODEL=$(get_actual_model_type "$EXP_NAME" "$MODEL")
        echo "[${MODEL}] Epoch ${EPOCH}..."
        ./run_analysis.sh "$ACTUAL_MODEL" "$EPOCH" "$EXP_NAME" 2>/dev/null || echo "  Skipped (checkpoint not found)"
    done
done

# Run for best epoch
for MODEL in "${MODELS[@]}"; do
    EXP_NAME="${MODEL}_v${VERSION}"
    ACTUAL_MODEL=$(get_actual_model_type "$EXP_NAME" "$MODEL")
    echo "[${MODEL}] Epoch best..."
    ./run_analysis.sh "$ACTUAL_MODEL" "best" "$EXP_NAME"
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Validation completed for v${VERSION}"
echo "========================================"
echo ""
echo "Results saved to:"
for MODEL in "${MODELS[@]}"; do
    echo "  ${MODEL}_v${VERSION}/validation/"
done
echo ""
echo "Best results:"
for MODEL in "${MODELS[@]}"; do
    echo "  ${MODEL}_v${VERSION}/validation/best/validation_results.txt"
done
