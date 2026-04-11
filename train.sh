#!/bin/bash
# Parallel training for all experiment configs.
#
# Runs up to MAX_JOBS processes concurrently.
# When one finishes, the next config in the queue starts automatically.
#
# Usage (new — config groups):
#   ./train.sh                                  # Run all io × model combos
#   ./train.sh --filter out12h                  # Only io configs matching "out12h"
#   ./train.sh --model transformer              # Only transformer model
#   ./train.sh --filter in2d --model gnn_tcn    # Specific io + model
#   ./train.sh --max-jobs 4                     # Limit to 4 parallel jobs
#   ./train.sh --dry-run                        # Print configs without running
#
# Usage (legacy — flat config files in configs/archive/):
#   ./train.sh --legacy                         # Run all archived flat configs
#   ./train.sh --legacy --filter out12h         # Filter archived configs
#
# Usage (file-based):
#   ./train.sh --config-file list.txt           # Run configs from file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
CONFIG_FILE=""
FILTER=""
MODEL_FILTER=""
DRY_RUN=false
LEGACY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --legacy)
            LEGACY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train.sh [--config-file FILE] [--max-jobs N] [--filter PATTERN] [--model MODEL] [--legacy] [--dry-run]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Collect configs
# =============================================================================
# Each entry is either:
#   - legacy mode: a config name (e.g. "in2d_out12h_transformer")
#   - group mode:  "io=in2d_out12h model=transformer" (space-separated overrides)
CONFIGS=()
DISPLAY_NAMES=()

if [[ -n "$CONFIG_FILE" ]]; then
    # Read from file (skip empty lines and comments)
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [[ -z "$line" ]] && continue
        CONFIGS+=("$line")
        DISPLAY_NAMES+=("$line")
    done < "$CONFIG_FILE"
elif $LEGACY; then
    # Legacy mode: glob from configs/archive/
    for f in configs/archive/in[123]d_out*.yaml; do
        name=$(basename "$f" .yaml)
        if [[ -n "$FILTER" && ! "$name" =~ $FILTER ]]; then
            continue
        fi
        CONFIGS+=("$name")
        DISPLAY_NAMES+=("$name")
    done
    IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS
    IFS=$'\n' DISPLAY_NAMES=($(sort <<<"${DISPLAY_NAMES[*]}")); unset IFS
else
    # New config group mode: io × model cross product
    IO_CONFIGS=()
    for f in configs/io/*.yaml; do
        io_name=$(basename "$f" .yaml)
        if [[ -n "$FILTER" && ! "$io_name" =~ $FILTER ]]; then
            continue
        fi
        IO_CONFIGS+=("$io_name")
    done
    IFS=$'\n' IO_CONFIGS=($(sort <<<"${IO_CONFIGS[*]}")); unset IFS

    MODEL_CONFIGS=()
    for f in configs/model/*.yaml; do
        model_name=$(basename "$f" .yaml)
        if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
            continue
        fi
        MODEL_CONFIGS+=("$model_name")
    done
    IFS=$'\n' MODEL_CONFIGS=($(sort <<<"${MODEL_CONFIGS[*]}")); unset IFS

    for io in "${IO_CONFIGS[@]}"; do
        for mdl in "${MODEL_CONFIGS[@]}"; do
            exp_name="${io}_${mdl}"
            CONFIGS+=("+io=${io} +model=${mdl} experiment.name=${exp_name}")
            DISPLAY_NAMES+=("${exp_name}")
        done
    done
fi

TOTAL=${#CONFIGS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "No configs found (config-file: '$CONFIG_FILE', filter: '$FILTER', model: '$MODEL_FILTER')"
    exit 1
fi

echo "========================================"
echo "Parallel Training Runner"
echo "========================================"
echo "Total configs: $TOTAL"
echo "Max parallel:  $MAX_JOBS"
if $LEGACY; then
    echo "Mode:          legacy (archive)"
elif [[ -n "$CONFIG_FILE" ]]; then
    echo "Source:        $CONFIG_FILE"
else
    echo "Mode:          config groups (io × model)"
fi
echo "Filter:        ${FILTER:-none}"
echo "Model:         ${MODEL_FILTER:-all}"
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Configs to run:"
    for name in "${DISPLAY_NAMES[@]}"; do
        echo "  $name"
    done
    echo ""
    echo "Total: $TOTAL configs"
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/train_logs"
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
                wait "$pid"
                code=$?
                COMPLETED=$((COMPLETED + 1))
                if [[ $code -eq 0 ]]; then
                    echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
                else
                    echo "[FAIL]  $name  (exit $code) — see $LOG_DIR/${name}.log"
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

    if $LEGACY || [[ -n "$CONFIG_FILE" ]]; then
        # Legacy / file mode: config name passed as --config-name
        python scripts/train.py --config-name="$cfg" \
            > "$LOG_DIR/${display_name}.log" 2>&1 &
    else
        # Config group mode: overrides passed as positional args
        # shellcheck disable=SC2086
        python scripts/train.py --config-name=local $cfg \
            > "$LOG_DIR/${display_name}.log" 2>&1 &
    fi

    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$display_name")
done

for i in "${!RUNNING_PIDS[@]}"; do
    pid=${RUNNING_PIDS[$i]}
    name=${RUNNING_NAMES[$i]}
    wait "$pid"
    code=$?
    COMPLETED=$((COMPLETED + 1))
    if [[ $code -eq 0 ]]; then
        echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
    else
        echo "[FAIL]  $name  (exit $code) — see $LOG_DIR/${name}.log"
        FAILED=$((FAILED + 1))
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR/"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed configs:"
    grep -l "Error\|Exception\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | while read f; do
        echo "  $(basename "$f" .log)"
    done
    exit 1
fi
