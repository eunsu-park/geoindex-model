#!/bin/bash
# Cross-validation training runner.
#
# Trains every (model x fold) combination for a fixed I/O config.
# Runs up to MAX_JOBS processes concurrently.
#
# Usage:
#   ./train_cv.sh                                  # All 14 models x 5 folds
#   ./train_cv.sh --model transformer              # Only transformer
#   ./train_cv.sh --fold 1                         # Only fold 1
#   ./train_cv.sh --model gnn_transformer --fold 3 # Specific model and fold
#   ./train_cv.sh --io in12h_out24h                # Override io (default: in12h_out12h)
#   ./train_cv.sh --max-jobs 4                     # Limit parallel jobs
#   ./train_cv.sh --dry-run                        # Print configs without running
#   ./train_cv.sh --config-name dev                # Use configs/dev.yaml (default: local)
#
# Naming convention:
#   experiment.name = "${io}_${model}_fold${N}"
#   e.g. in12h_out12h_gnn_transformer_fold3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

shopt -s nullglob

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
MODEL_FILTER=""
FOLD_FILTER=""
IO="in12h_out12h"
DRY_RUN=false
CONFIG_NAME="local"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)   MAX_JOBS="$2"; shift 2 ;;
        --model)      MODEL_FILTER="$2"; shift 2 ;;
        --fold)       FOLD_FILTER="$2"; shift 2 ;;
        --io)         IO="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train_cv.sh [--model MODEL] [--fold N] [--io IO] [--max-jobs N] [--dry-run] [--config-name NAME]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Collect configs (model x fold cross product)
# =============================================================================
MODEL_CONFIGS=()
for f in configs/model/*.yaml; do
    name=$(basename "$f" .yaml)
    if [[ -n "$MODEL_FILTER" && "$name" != "$MODEL_FILTER" ]]; then
        continue
    fi
    MODEL_CONFIGS+=("$name")
done
MODEL_CONFIGS=($(printf '%s\n' "${MODEL_CONFIGS[@]}" | sort))

FOLDS=()
for f in configs/cv/fold*.yaml; do
    name=$(basename "$f" .yaml)   # fold1, fold2, ...
    if [[ -n "$FOLD_FILTER" && "$name" != "fold${FOLD_FILTER}" && "$name" != "$FOLD_FILTER" ]]; then
        continue
    fi
    FOLDS+=("$name")
done
FOLDS=($(printf '%s\n' "${FOLDS[@]}" | sort))

CONFIGS=()
DISPLAY_NAMES=()
for mdl in "${MODEL_CONFIGS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        exp_name="${IO}_${mdl}_${fold}"
        CONFIGS+=("+io=${IO} +model=${mdl} +cv=${fold} experiment.name=${exp_name}")
        DISPLAY_NAMES+=("${exp_name}")
    done
done

TOTAL=${#CONFIGS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "No configs matched (model='$MODEL_FILTER', fold='$FOLD_FILTER')"
    exit 1
fi

echo "========================================"
echo "Cross-Validation Training Runner"
echo "========================================"
echo "Total runs:    $TOTAL  (${#MODEL_CONFIGS[@]} models x ${#FOLDS[@]} folds)"
echo "Max parallel:  $MAX_JOBS"
echo "I/O config:    $IO"
echo "Model filter:  ${MODEL_FILTER:-all}"
echo "Fold filter:   ${FOLD_FILTER:-all}"
echo "Config name:   $CONFIG_NAME"
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Runs to execute:"
    for name in "${DISPLAY_NAMES[@]}"; do
        echo "  $name"
    done
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/train_cv_logs"
mkdir -p "$LOG_DIR"

RUNNING_PIDS=()
RUNNING_NAMES=()
COMPLETED=0
FAILED=0
STARTED=0

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; do
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!RUNNING_PIDS[@]}"; do
            pid=${RUNNING_PIDS[$i]}
            name=${RUNNING_NAMES[$i]}
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
                NEW_NAMES+=("$name")
            else
                wait "$pid" && code=0 || code=$?
                COMPLETED=$((COMPLETED + 1))
                if [[ $code -eq 0 ]]; then
                    echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
                else
                    echo "[FAIL]  $name  (exit $code) - see $LOG_DIR/${name}.log"
                    FAILED=$((FAILED + 1))
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_NAMES=("${NEW_NAMES[@]}")
        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; then
            sleep 5
        fi
    done
}

for idx in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$idx]}"
    display_name="${DISPLAY_NAMES[$idx]}"

    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $display_name  ($STARTED/$TOTAL, running: ${#RUNNING_PIDS[@]}+1)"

    # shellcheck disable=SC2086
    python scripts/train.py --config-name="$CONFIG_NAME" $cfg \
        > "$LOG_DIR/${display_name}.log" 2>&1 &

    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$display_name")
done

for i in "${!RUNNING_PIDS[@]}"; do
    pid=${RUNNING_PIDS[$i]}
    name=${RUNNING_NAMES[$i]}
    wait "$pid" && code=0 || code=$?
    COMPLETED=$((COMPLETED + 1))
    if [[ $code -eq 0 ]]; then
        echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
    else
        echo "[FAIL]  $name  (exit $code) - see $LOG_DIR/${name}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "CV Training Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR/"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed runs:"
    grep -l "Error\|Exception\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | while read f; do
        echo "  $(basename "$f" .log)"
    done
    exit 1
fi
