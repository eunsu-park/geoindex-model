#!/bin/bash
# Parallel saliency map extraction for all experiment configs.
#
# Runs up to MAX_JOBS processes concurrently.
# When one finishes, the next config in the queue starts automatically.
#
# Usage:
#   ./saliency.sh                           # Run all configs, epoch=best
#   ./saliency.sh --config-file list.txt    # Run configs from file
#   ./saliency.sh --max-jobs 4              # Limit to 4 parallel jobs
#   ./saliency.sh --filter out12h           # Only configs matching "out12h"
#   ./saliency.sh --dry-run                 # Print configs without running
#   ./saliency.sh --epoch 10                # Use epoch 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
CONFIG_FILE=""
FILTER=""
DRY_RUN=false
EPOCH="best"

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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./saliency.sh [--config-file FILE] [--max-jobs N] [--filter PATTERN] [--dry-run] [--epoch EPOCH]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Collect configs
# =============================================================================
CONFIGS=()

if [[ -n "$CONFIG_FILE" ]]; then
    # Read from file (skip empty lines and comments)
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [[ -z "$line" ]] && continue
        CONFIGS+=("$line")
    done < "$CONFIG_FILE"
else
    # Glob from configs/ directory
    for f in configs/in[123]d_out*.yaml; do
        name=$(basename "$f" .yaml)
        if [[ -n "$FILTER" && ! "$name" =~ $FILTER ]]; then
            continue
        fi
        CONFIGS+=("$name")
    done
    IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS
fi

TOTAL=${#CONFIGS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "No configs found (config-file: '$CONFIG_FILE', filter: '$FILTER')"
    exit 1
fi

echo "========================================"
echo "Parallel Saliency Analysis Runner"
echo "========================================"
echo "Total configs: $TOTAL"
echo "Max parallel:  $MAX_JOBS"
echo "Source:        ${CONFIG_FILE:-glob (filter: ${FILTER:-none})}"
echo "Epoch:         $EPOCH"
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Configs to run:"
    for cfg in "${CONFIGS[@]}"; do
        echo "  $cfg"
    done
    echo ""
    echo "Total: $TOTAL configs"
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/saliency_logs"
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

for cfg in "${CONFIGS[@]}"; do
    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $cfg  ($STARTED/$TOTAL, running: ${#RUNNING_PIDS[@]}+1)"

    python analysis/run_saliency.py --config-name="$cfg" saliency.epoch="$EPOCH" \
        > "$LOG_DIR/${cfg}.log" 2>&1 &

    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$cfg")
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
echo "Saliency Analysis Complete"
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
