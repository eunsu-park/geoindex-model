#!/bin/bash
# Temporary runner for remaining MCD + attention analyses.
#
# Scans save_root for missing outputs and runs only the missing io x model
# combinations in parallel. Safe to re-run — completed experiments are skipped.
#
# Usage:
#   ./run_remaining_analysis.sh                    # Run all missing (mcd + attention)
#   ./run_remaining_analysis.sh --only mcd         # MCD only
#   ./run_remaining_analysis.sh --only attention   # Attention only
#   ./run_remaining_analysis.sh --dry-run          # Print task list, do not run
#   SAVE_ROOT=/custom/path ./run_remaining_analysis.sh
#   ./run_remaining_analysis.sh --epoch best       # Checkpoint epoch (default: best)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
EPOCH="best"
DRY_RUN=false
ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        --epoch)    EPOCH="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Resolve save_root
# =============================================================================
if [[ -z "${SAVE_ROOT:-}" ]]; then
    SAVE_ROOT=$(grep -E "^\s*save_root:" configs/local.yaml \
        | head -1 \
        | sed -E 's/.*save_root:[[:space:]]*"?([^"]*)"?.*/\1/')
fi
if [[ -z "$SAVE_ROOT" || ! -d "$SAVE_ROOT" ]]; then
    echo "ERROR: SAVE_ROOT not found or invalid: '$SAVE_ROOT'"
    echo "Set SAVE_ROOT env var or fix configs/local.yaml."
    exit 1
fi

# =============================================================================
# Matrix definition
# =============================================================================
IO_CONFIGS=(
    in6h_out6h  in6h_out12h  in6h_out18h  in6h_out24h
    in12h_out6h in12h_out12h in12h_out18h in12h_out24h
    in18h_out6h in18h_out12h in18h_out18h in18h_out24h
    in1d_out6h  in1d_out12h  in1d_out18h  in1d_out24h
    in2d_out6h  in2d_out12h  in2d_out18h  in2d_out24h
    in3d_out6h  in3d_out12h  in3d_out18h  in3d_out24h
)

ALL_MODELS=(linear transformer tcn patchtst timesnet
            gnn_transformer gnn_tcn gnn_bilstm gnn_patchtst)
ATTN_MODELS=(transformer patchtst gnn_transformer gnn_patchtst)

# =============================================================================
# Detect missing tasks
# =============================================================================
TASKS=()

run_mcd=true
run_attn=true
case "$ONLY" in
    mcd)       run_attn=false ;;
    attention) run_mcd=false ;;
    "")        ;;
    *) echo "Invalid --only value: $ONLY (use mcd or attention)"; exit 1 ;;
esac

for io in "${IO_CONFIGS[@]}"; do
    if $run_mcd; then
        for m in "${ALL_MODELS[@]}"; do
            out="$SAVE_ROOT/${io}_${m}/mcd/${EPOCH}/npz.zip"
            [[ -f "$out" ]] || TASKS+=("mcd:${io}:${m}")
        done
    fi
    if $run_attn; then
        for m in "${ATTN_MODELS[@]}"; do
            out="$SAVE_ROOT/${io}_${m}/attention/${EPOCH}/npz.zip"
            [[ -f "$out" ]] || TASKS+=("attention:${io}:${m}")
        done
    fi
done

TOTAL=${#TASKS[@]}

echo "========================================"
echo "Remaining Analysis Runner"
echo "========================================"
echo "Save root:     $SAVE_ROOT"
echo "Only:          ${ONLY:-mcd+attention}"
echo "Epoch:         $EPOCH"
echo "Max parallel:  $MAX_JOBS"
echo "Missing tasks: $TOTAL"
echo "========================================"

if [[ $TOTAL -eq 0 ]]; then
    echo "All targets already have outputs. Nothing to do."
    exit 0
fi

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Missing tasks:"
    for t in "${TASKS[@]}"; do echo "  $t"; done
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/remaining_analysis_logs"
mkdir -p "$LOG_DIR"

RUNNING_PIDS=()
RUNNING_NAMES=()
COMPLETED=0
FAILED=0
STARTED=0

reap_finished() {
    local new_pids=()
    local new_names=()
    for i in "${!RUNNING_PIDS[@]}"; do
        local pid=${RUNNING_PIDS[$i]}
        local name=${RUNNING_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_names+=("$name")
        else
            local code=0
            wait "$pid" || code=$?
            COMPLETED=$((COMPLETED + 1))
            if [[ $code -eq 0 ]]; then
                echo "[DONE]  $name  ($COMPLETED/$TOTAL)"
            else
                echo "[FAIL]  $name  (exit $code) — $LOG_DIR/${name}.log"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
    RUNNING_NAMES=("${new_names[@]}")
}

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; do
        reap_finished
        [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]] && sleep 5
    done
}

for t in "${TASKS[@]}"; do
    phase=${t%%:*}
    rest=${t#*:}
    io=${rest%%:*}
    mdl=${rest#*:}
    exp="${io}_${mdl}"
    name="${phase}__${exp}"

    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $name  ($STARTED/$TOTAL, running: $((${#RUNNING_PIDS[@]} + 1)))"

    python "analysis/run_${phase}.py" --config-name=local \
        "+io=${io}" "+model=${mdl}" "experiment.name=${exp}" \
        "${phase}.epoch=${EPOCH}" \
        > "$LOG_DIR/${name}.log" 2>&1 &
    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$name")
done

# Drain remaining jobs
while [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; do
    reap_finished
    [[ ${#RUNNING_PIDS[@]} -gt 0 ]] && sleep 5
done

echo ""
echo "========================================"
echo "Remaining Analysis Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR"
echo "========================================"

[[ $FAILED -eq 0 ]] || exit 1
