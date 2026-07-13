#!/bin/bash
# Run pending validation / MCD / attention jobs in parallel.
#
# Scans save_root for missing outputs across the io x model matrix and
# executes only the combinations whose expected artifact is absent.
# Safe to re-run — completed experiments are skipped automatically.
#
# Completion markers (per phase, under {save_root}/{experiment}/):
#   validation/{epoch}/validation_results.csv
#   mcd/{epoch}/npz.zip
#   attention/{epoch}/npz.zip   (only for transformer, patchtst,
#                                gnn_transformer, gnn_patchtst)
#
# Usage:
#   ./run_pending.sh                              # validation + mcd + attention
#   ./run_pending.sh --phases mcd,attention       # subset (comma-separated)
#   ./run_pending.sh --phases validation          # single phase
#   ./run_pending.sh --dry-run                    # print tasks, do not execute
#   ./run_pending.sh --max-jobs 4 --epoch best
#   ./run_pending.sh --config-name dev            # use configs/dev.yaml (default: local)
#   SAVE_ROOT=/custom/path ./run_pending.sh       # override results root

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
EPOCH="best"
DRY_RUN=false
PHASES="validation,mcd,attention"
CONFIG_NAME="local"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)    MAX_JOBS="$2"; shift 2 ;;
        --epoch)       EPOCH="$2"; shift 2 ;;
        --phases)      PHASES="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate requested phases
IFS=',' read -r -a PHASE_LIST <<< "$PHASES"
for p in "${PHASE_LIST[@]}"; do
    case $p in
        validation|mcd|attention) ;;
        *) echo "Invalid phase: $p (allowed: validation, mcd, attention)"; exit 1 ;;
    esac
done

# =============================================================================
# Resolve save_root
# =============================================================================
CONFIG_FILE_PATH="configs/${CONFIG_NAME}.yaml"
if [[ -z "${SAVE_ROOT:-}" ]]; then
    if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
        echo "ERROR: Config file not found: $CONFIG_FILE_PATH"
        exit 1
    fi
    SAVE_ROOT=$(grep -E "^\s*save_root:" "$CONFIG_FILE_PATH" \
        | head -1 \
        | sed -E 's/.*save_root:[[:space:]]*"?([^"]*)"?.*/\1/')
fi
if [[ -z "$SAVE_ROOT" || ! -d "$SAVE_ROOT" ]]; then
    echo "ERROR: SAVE_ROOT not found or invalid: '$SAVE_ROOT'"
    echo "Set SAVE_ROOT env var or fix $CONFIG_FILE_PATH."
    exit 1
fi

# =============================================================================
# Experiment-name prefix + hp GNN-node fix (server profiles)
#   server_ap -> "ap_" prefix ; server_hp -> "hp_" prefix + drop the inherited
#   ap30 GNN node. The prefix also scopes the save_root scan below so ap and hp
#   results (ap_*/hp_*) are detected independently. Other profiles: no prefix.
#   NOTE: server_hp inherits save_root from server_ap, so it is not literally in
#   configs/server_hp.yaml — run this as `SAVE_ROOT=... ./run_pending.sh
#   --config-name server_hp`.
# =============================================================================
case "$CONFIG_NAME" in
    server_ap) EXP_PREFIX="ap_" ;;
    server_hp) EXP_PREFIX="hp_" ;;
    *)         EXP_PREFIX="" ;;
esac
EXTRA_ARGS=()
if [[ "$CONFIG_NAME" == "server_hp" ]]; then
    EXTRA_ARGS+=("~data.timeseries.gnn_variable_groups.ap30")
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

ALL_MODELS=(linear transformer tcn patchtst timesnet lstm bilstm
            gnn_transformer gnn_tcn gnn_bilstm gnn_patchtst
            gnn_lstm gnn_timesnet gnn_linear)
ATTN_MODELS=(transformer patchtst gnn_transformer gnn_patchtst)

# Phase -> completion marker (relative to {save_root}/{experiment}/)
marker_for_phase() {
    case $1 in
        validation) echo "validation/${EPOCH}/validation_results.csv" ;;
        mcd)        echo "mcd/${EPOCH}/npz.zip" ;;
        attention)  echo "attention/${EPOCH}/npz.zip" ;;
    esac
}

# Phase -> python entry script
runner_for_phase() {
    case $1 in
        validation) echo "scripts/validate.py" ;;
        mcd)        echo "analysis/run_mcd.py" ;;
        attention)  echo "analysis/run_attention.py" ;;
    esac
}

# Phase -> applicable models
models_for_phase() {
    case $1 in
        attention) echo "${ATTN_MODELS[@]}" ;;
        *)         echo "${ALL_MODELS[@]}" ;;
    esac
}

# =============================================================================
# Detect pending tasks
# =============================================================================
TASKS=()

for phase in "${PHASE_LIST[@]}"; do
    marker=$(marker_for_phase "$phase")
    read -r -a models <<< "$(models_for_phase "$phase")"
    for io in "${IO_CONFIGS[@]}"; do
        for m in "${models[@]}"; do
            if [[ ! -f "$SAVE_ROOT/${EXP_PREFIX}${io}_${m}/${marker}" ]]; then
                TASKS+=("${phase}:${io}:${m}")
            fi
        done
    done
done

TOTAL=${#TASKS[@]}

echo "========================================"
echo "Pending Analysis Runner"
echo "========================================"
echo "Save root:     $SAVE_ROOT"
echo "Config name:   $CONFIG_NAME"
echo "Phases:        $PHASES"
echo "Epoch:         $EPOCH"
echo "Max parallel:  $MAX_JOBS"
echo "Pending tasks: $TOTAL"
echo "========================================"

if [[ $TOTAL -eq 0 ]]; then
    echo "All targets already complete. Nothing to do."
    exit 0
fi

# Per-phase summary
for phase in "${PHASE_LIST[@]}"; do
    cnt=0
    for t in "${TASKS[@]}"; do
        [[ "${t%%:*}" == "$phase" ]] && cnt=$((cnt + 1))
    done
    printf "  %-11s %d\n" "$phase:" "$cnt"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Pending tasks:"
    for t in "${TASKS[@]}"; do echo "  $t"; done
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/pending_analysis_logs"
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
        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; then
            sleep 5
        fi
    done
}

for t in "${TASKS[@]}"; do
    phase=${t%%:*}
    rest=${t#*:}
    io=${rest%%:*}
    mdl=${rest#*:}
    exp="${EXP_PREFIX}${io}_${mdl}"
    name="${phase}__${exp}"

    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $name  ($STARTED/$TOTAL, running: $((${#RUNNING_PIDS[@]} + 1)))"

    runner=$(runner_for_phase "$phase")
    python "$runner" --config-name="$CONFIG_NAME" \
        "+io=${io}" "+model=${mdl}" "experiment.name=${exp}" \
        "${phase}.epoch=${EPOCH}" "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/${name}.log" 2>&1 &
    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$name")
done

# Drain remaining jobs
while [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; do
    reap_finished
    if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "Pending Analysis Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR"
echo "========================================"

[[ $FAILED -eq 0 ]] || exit 1
