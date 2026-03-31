#!/bin/bash
#SBATCH --job-name=ap_train
#SBATCH --output=experiments/logs/train_%A_%a.out
#SBATCH --error=experiments/logs/train_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER@njit.edu

# =============================================================================
# SLURM Job Array for Large-Scale Experiment Training
# =============================================================================
#
# Usage:
#   # Submit all 1176 experiments (max 50 concurrent)
#   sbatch --array=1-1176%50 experiments/slurm/train_array.slurm
#
#   # Submit subset for testing
#   sbatch --array=1-10 experiments/slurm/train_array.slurm
#
#   # Submit specific experiments
#   sbatch --array=1,5,10,100 experiments/slurm/train_array.slurm
#
#   # Resume failed experiments (comma-separated list)
#   sbatch --array=45,67,234 experiments/slurm/train_array.slurm
#
# =============================================================================

# Exit on error
set -e

# =============================================================================
# Environment Setup
# =============================================================================

# Load modules (adjust for your cluster)
module purge
module load python/3.10
module load cuda/11.8

# Activate conda environment (if using)
# source activate ap_prediction

# Project paths
PROJECT_ROOT="/mmfs1/home/hl545/ap/renew/02_Regression"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p experiments/logs

# =============================================================================
# Experiment Configuration
# =============================================================================

# Get experiment ID from SLURM array task ID
EXP_ID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Config file path
CONFIG_NAME="exp_${EXP_ID}"
CONFIG_PATH="experiments/configs"

echo "=============================================="
echo "SLURM Job Array - Experiment Training"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Experiment ID: $EXP_ID"
echo "Config:        ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start Time:    $(date)"
echo "=============================================="

# =============================================================================
# Update Tracking Status: Running
# =============================================================================

python experiments/experiment_tracker.py update "$EXP_ID" submitted \
    --slurm-job-id "$SLURM_JOB_ID" \
    --start-time "$(date -Iseconds)" \
    2>/dev/null || echo "Warning: Could not update tracker (submitted)"

python experiments/experiment_tracker.py update "$EXP_ID" running \
    2>/dev/null || echo "Warning: Could not update tracker (running)"

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "Starting training..."
echo ""

# Run training with Hydra
python scripts/train.py \
    --config-path="../${CONFIG_PATH}" \
    --config-name="$CONFIG_NAME"

TRAIN_EXIT_CODE=$?

# =============================================================================
# Update Tracking Status: Completed or Failed
# =============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"

    python experiments/experiment_tracker.py update "$EXP_ID" completed \
        --end-time "$(date -Iseconds)" \
        2>/dev/null || echo "Warning: Could not update tracker (completed)"
else
    echo ""
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"

    python experiments/experiment_tracker.py update "$EXP_ID" failed \
        --end-time "$(date -Iseconds)" \
        --notes "Exit code: $TRAIN_EXIT_CODE" \
        2>/dev/null || echo "Warning: Could not update tracker (failed)"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Job Summary"
echo "=============================================="
echo "Experiment ID: $EXP_ID"
echo "Exit Code:     $TRAIN_EXIT_CODE"
echo "End Time:      $(date)"
echo "=============================================="

exit $TRAIN_EXIT_CODE
