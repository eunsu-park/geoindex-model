#!/bin/bash
# Parallel training for all experiment configs.
#
# Runs up to MAX_JOBS training processes concurrently.
# When one finishes, the next config in the queue starts automatically.
#
# Usage:
#   ./train.sh                    # Run all 81 configs (default)
#   ./train.sh --max-jobs 4       # Limit to 4 parallel jobs
#   ./train.sh --filter out12h    # Only configs matching "out12h"
#   ./train.sh --dry-run          # Print configs without running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_JOBS="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train.sh [--max-jobs N] [--filter PATTERN] [--dry-run]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Collect configs
# =============================================================================
CONFIGS=()
for f in configs/in[123]d_out*.yaml; do
    name=$(basename "$f" .yaml)
    if [[ -n "$FILTER" && ! "$name" =~ $FILTER ]]; then
        continue
    fi
    CONFIGS+=("$name")
done

# Sort for consistent ordering
IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS

TOTAL=${#CONFIGS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "No configs found (filter: '$FILTER')"
    exit 1
fi

echo "========================================"
echo "Parallel Training Runner"
echo "========================================"
echo "Total configs: $TOTAL"
echo "Max parallel:  $MAX_JOBS"
echo "Filter:        ${FILTER:-none}"
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
# Training loop
# =============================================================================
LOG_DIR="$HOME/tmp/train_logs"
mkdir -p "$LOG_DIR"

RUNNING_PIDS=()
RUNNING_NAMES=()
COMPLETED=0
FAILED=0
STARTED=0

# Wait for a slot to open (any one job finishes)
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

# Launch all configs
for cfg in "${CONFIGS[@]}"; do
    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $cfg  ($STARTED/$TOTAL, running: ${#RUNNING_PIDS[@]}+1)"

    python scripts/train.py --config-name="$cfg" \
        > "$LOG_DIR/${cfg}.log" 2>&1 &

    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$cfg")
done

# Wait for remaining jobs
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
