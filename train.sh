#!/bin/bash
# Training script for solar wind prediction experiments
# Usage: ./train.sh <experiment> [model_type]
#
# Arguments:
#   experiment  - Experiment name/version (e.g., v11, baseline_v11)
#   model_type  - (Optional) Model type override: baseline, fusion, transformer, linear
#
# Examples:
#   ./train.sh v11                  # Train fusion_v11 (default model type)
#   ./train.sh v11 baseline         # Train baseline_v11
#   ./train.sh v11 fusion           # Train fusion_v11
#   ./train.sh v11b baseline        # Train baseline_v11b (variant)

set -e  # Exit on error

# =============================================================================
# Arguments
# =============================================================================
EXPERIMENT=${1:-}
MODEL_TYPE=${2:-transformer}  # Default to transformer (CSV timeseries mode)

if [ -z "$EXPERIMENT" ]; then
    echo "Error: Experiment name required"
    echo "Usage: ./train.sh <experiment> [model_type]"
    echo ""
    echo "Model types: baseline, fusion, transformer, linear"
    echo ""
    echo "Examples:"
    echo "  ./train.sh v11                  # fusion_v11"
    echo "  ./train.sh v11 baseline         # baseline_v11"
    echo "  ./train.sh v11b baseline        # baseline_v11b (variant)"
    exit 1
fi

# Validate model type
case $MODEL_TYPE in
    baseline|fusion|transformer|linear|tcn)
        ;;
    *)
        echo "Error: Unknown model_type '$MODEL_TYPE'"
        echo "Valid options: baseline, fusion, transformer, linear, tcn"
        exit 1
        ;;
esac

# Construct experiment name
EXP_NAME="${MODEL_TYPE}_${EXPERIMENT}"

# =============================================================================
# Experiment Configurations
# =============================================================================
# Base options for all experiments
BASE_OPTS="experiment.name=$EXP_NAME model.model_type=$MODEL_TYPE"

# Experiment-specific configurations
case $EXPERIMENT in
    v11|v11a)
        # Phase 1: Overfitting Fix (all improvements)
        EXTRA_OPTS="
            training.lr_warmup.enable=true
            training.lr_warmup.warmup_epochs=5
            training.lr_warmup.warmup_start_factor=0.1
            training.scheduler_type=cosine_annealing
            training.gradient_accumulation_steps=4
            training.early_stopping_patience=15
        "
        ;;
    v11b)
        # Phase 1 variant: LR Warmup + Cosine Annealing only (no gradient accumulation)
        EXTRA_OPTS="
            training.lr_warmup.enable=true
            training.lr_warmup.warmup_epochs=3
            training.lr_warmup.warmup_start_factor=0.1
            training.scheduler_type=cosine_annealing
            training.gradient_accumulation_steps=1
            training.early_stopping_patience=15
        "
        ;;
    v11c)
        # Phase 1 variant: Cosine Annealing only
        EXTRA_OPTS="
            training.lr_warmup.enable=false
            training.scheduler_type=cosine_annealing
            training.gradient_accumulation_steps=1
            training.early_stopping_patience=15
        "
        ;;
    v12)
        # Phase 2: Data Augmentation (placeholder)
        EXTRA_OPTS="
            training.lr_warmup.enable=true
            training.scheduler_type=cosine_annealing
            training.early_stopping_patience=15
        "
        ;;
    v13|v13a)
        # Phase 3: TCN Encoder (default config)
        # Note: model_type override required: ./train.sh v13 tcn
        EXTRA_OPTS="
            model.tcn_channels=[64,128,256]
            model.tcn_kernel_size=3
            model.tcn_dropout=0.1
        "
        ;;
    v13b)
        # Phase 3 variant: Deeper TCN (4 layers)
        EXTRA_OPTS="
            model.tcn_channels=[64,128,256,512]
            model.tcn_kernel_size=3
            model.tcn_dropout=0.1
        "
        ;;
    v13c)
        # Phase 3 variant: Larger kernel (kernel=5)
        EXTRA_OPTS="
            model.tcn_channels=[64,128,256]
            model.tcn_kernel_size=5
            model.tcn_dropout=0.1
        "
        ;;
    *)
        # Default: No extra options
        EXTRA_OPTS=""
        ;;
esac

# =============================================================================
# Run Training
# =============================================================================
echo "========================================"
echo "Training: $EXP_NAME"
echo "========================================"
echo "Model type: $MODEL_TYPE"
echo "Experiment: $EXPERIMENT"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build and run command
CMD="python scripts/train.py --config-name=local $BASE_OPTS $EXTRA_OPTS"
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "Training completed: $EXP_NAME"
echo "========================================"
