#!/bin/bash
# Universal analysis script for all model types
# Usage: ./run_analysis.sh <model_type> <epoch> [experiment_name]
#
# Arguments:
#   model_type      - fusion, transformer, convlstm, baseline
#   epoch           - Epoch number to analyze
#   experiment_name - (Optional) Experiment name for result folder
#                     Default: same as model_type
#
# Model types:
#   fusion      - Multi-modal fusion (SDO + OMNI)
#   transformer - OMNI-only Transformer
#   convlstm    - SDO-only ConvLSTM
#   baseline    - Conv3D + Linear baseline (Son et al. 2023)
#
# Examples:
#   ./run_analysis.sh fusion 10              # Results in fusion/
#   ./run_analysis.sh fusion 10 fusion_v2    # Results in fusion_v2/
#   ./run_analysis.sh baseline 5 baseline_lr01

set -e  # Exit on error

# =============================================================================
# Arguments
# =============================================================================
MODEL_TYPE=${1:-fusion}
EPOCH=${2:-1}
EXP_NAME=${3:-$MODEL_TYPE}  # Default: same as model_type

# Validate model type
case $MODEL_TYPE in
    fusion|transformer|convlstm|baseline|linear|tcn)
        ;;
    *)
        echo "Error: Unknown model_type '$MODEL_TYPE'"
        echo "Valid options: fusion, transformer, convlstm, baseline, linear, tcn"
        exit 1
        ;;
esac

# =============================================================================
# Determine analysis compatibility
# =============================================================================
# Attention: requires Transformer (fusion, transformer)
# Saliency: requires ConvLSTM/Conv3D (fusion, convlstm, baseline)

RUN_ATTENTION=false
RUN_SALIENCY=false

case $MODEL_TYPE in
    fusion)
        RUN_ATTENTION=true
        RUN_SALIENCY=true
        ;;
    transformer)
        RUN_ATTENTION=true
        RUN_SALIENCY=false
        ;;
    convlstm)
        RUN_ATTENTION=false
        RUN_SALIENCY=true
        ;;
    baseline)
        RUN_ATTENTION=false
        RUN_SALIENCY=true
        ;;
    linear)
        RUN_ATTENTION=false
        RUN_SALIENCY=false
        ;;
    tcn)
        RUN_ATTENTION=false
        RUN_SALIENCY=false
        ;;
esac

# Count total steps
TOTAL_STEPS=2  # Validation + MCD always run
if [ "$RUN_ATTENTION" = true ]; then ((TOTAL_STEPS++)); fi
if [ "$RUN_SALIENCY" = true ]; then ((TOTAL_STEPS++)); fi

# =============================================================================
# Setup
# =============================================================================
echo "========================================"
echo "Running analysis: $MODEL_TYPE model"
echo "Experiment: $EXP_NAME"
echo "Epoch: $EPOCH"
echo "========================================"
echo ""
echo "Analyses to run:"
echo "  - Validation: YES"
echo "  - Attention:  $([ "$RUN_ATTENTION" = true ] && echo "YES" || echo "NO (no Transformer)")"
echo "  - Saliency:   $([ "$RUN_SALIENCY" = true ] && echo "YES" || echo "NO (no ConvLSTM/Conv3D)")"
echo "  - MCD:        YES"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model options
MODEL_OPTS="model.model_type=$MODEL_TYPE experiment.name=$EXP_NAME"

# =============================================================================
# Experiment-specific model hyperparameters
# =============================================================================
# Add experiment-specific overrides for models trained with non-default configs

case $EXP_NAME in
    transformer_v9)
        # v9: Trained with larger transformer
        MODEL_OPTS="$MODEL_OPTS model.d_model=256 model.transformer_nhead=8"
        MODEL_OPTS="$MODEL_OPTS model.transformer_num_layers=3 model.transformer_dim_feedforward=512"
        ;;
    fusion_v2)
        # v2: Trained with d_model=256
        MODEL_OPTS="$MODEL_OPTS model.d_model=256"
        ;;
    baseline_v2)
        # baseline uses default config
        ;;
    linear_v10)
        # v10: Trained with d_model=256
        MODEL_OPTS="$MODEL_OPTS model.d_model=256"
        ;;
    baseline_v11)
        # v11: Accidentally trained as fusion model, uses default config (d_model=128)
        # Note: MODEL_TYPE should be "fusion" when validating this experiment
        ;;
    *_v11|*_v11b|*_v11c)
        # v11 variants: Use default config (d_model=128)
        ;;
    tcn_v13|tcn_v13a)
        # v13: Default TCN (3 layers, kernel=3)
        MODEL_OPTS="$MODEL_OPTS model.tcn_channels=[64,128,256] model.tcn_kernel_size=3"
        ;;
    tcn_v13b)
        # v13b: Deeper TCN (4 layers, kernel=3)
        MODEL_OPTS="$MODEL_OPTS model.tcn_channels=[64,128,256,512] model.tcn_kernel_size=3"
        ;;
    tcn_v13c)
        # v13c: Larger kernel (3 layers, kernel=5)
        MODEL_OPTS="$MODEL_OPTS model.tcn_channels=[64,128,256] model.tcn_kernel_size=5"
        ;;
esac

# =============================================================================
# Run analyses
# =============================================================================
STEP=1

# 1. Validation (always)
echo ""
echo "[$STEP/$TOTAL_STEPS] Running Validation..."
echo "----------------------------------------"
python scripts/validate.py --config-name=local $MODEL_OPTS validation.epoch=$EPOCH
((STEP++))

# 2. Attention (if supported)
if [ "$RUN_ATTENTION" = true ]; then
    echo ""
    echo "[$STEP/$TOTAL_STEPS] Running Attention Analysis..."
    echo "----------------------------------------"
    # python analysis/run_attention.py --config-name=local $MODEL_OPTS attention.epoch=$EPOCH
    ((STEP++))
fi

# 3. Saliency (if supported)
if [ "$RUN_SALIENCY" = true ]; then
    echo ""
    echo "[$STEP/$TOTAL_STEPS] Running Saliency Analysis..."
    echo "----------------------------------------"
    # python analysis/run_saliency.py --config-name=local $MODEL_OPTS saliency.epoch=$EPOCH
    ((STEP++))
fi

# 4. Monte Carlo Dropout (always)
echo ""
echo "[$STEP/$TOTAL_STEPS] Running Monte Carlo Dropout..."
echo "----------------------------------------"
# python analysis/monte_carlo_dropout.py --config-name=local $MODEL_OPTS mcd.epoch=$EPOCH

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "All analyses completed!"
echo "========================================"
echo "Model: $MODEL_TYPE"
echo "Experiment: $EXP_NAME"
echo "Epoch: $EPOCH"
echo ""
echo "Results saved to:"
echo "  Validation: $RESULT_BASE/validation/epoch_$(printf '%04d' $EPOCH)"
if [ "$RUN_ATTENTION" = true ]; then
    echo "  Attention:  $RESULT_BASE/attention/epoch_$(printf '%04d' $EPOCH)"
fi
if [ "$RUN_SALIENCY" = true ]; then
    echo "  Saliency:   $RESULT_BASE/saliency/epoch_$(printf '%04d' $EPOCH)"
fi
echo "  MCD:        $RESULT_BASE/mcd/epoch_$(printf '%04d' $EPOCH)"
