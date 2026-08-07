#!/bin/bash
# Auxiliary channel-forecasting probe: does making the model forecast every INPUT
# channel over the target window, not just the index, improve the index forecast?
#
# Motivation, measured before writing this (measurements/wind_oracle_curve.py):
# handing the model the true future wind over the 12 h window is worth per-lead
# rho 0.594 -> 0.813, and Bz alone accounts for +0.166 of that +0.220. Nothing
# else tried in this investigation moved rho by more than +0.017. So the useful
# thing a trunk can learn is the future wind, and the auxiliary task says so out
# loud instead of hoping the index loss implies it.
#
# A 4-seed MLP probe on identical anchors (measurements/multitask_direct.py) put
# the gain at per-lead rho 0.5413 +-0.0023 -> 0.5710 +-0.0029, rho at 12 h
# 0.334 -> 0.376, peak rho 0.665 -> 0.679, with non-overlapping seed ranges.
# This script asks whether it survives on the real architecture.
#
# The aux01 variant is a diagnostic rather than a candidate: if turning the
# auxiliary term down to a tenth costs the gain while aux05/aux1/aux2 land
# together, the term is doing work rather than acting as generic regularisation.
#
# MC-dropout is off during validation (validation.mcd_samples=0): the probe only
# needs the deterministic forecast, and that makes each validation ~100x cheaper.
#
# Usage:
#   ./exploration/2026-08/analysis/aux_forecast_probe.sh --config-name server_ap          # all variants
#   ./exploration/2026-08/analysis/aux_forecast_probe.sh --variant aux1 --seeds 3
#   ./exploration/2026-08/analysis/aux_forecast_probe.sh --io in12h_out12h --model gnn_transformer
#   ./exploration/2026-08/analysis/aux_forecast_probe.sh --dry-run
#   ./exploration/2026-08/analysis/aux_forecast_probe.sh --config-name mac_ap \
#       --variant aux1 --extra "training.epochs=1" --tag smoke      # smoke test
#
# Results land in $SAVE_ROOT/auxprobe[_<tag>]_<target>_<io>_<model>_<variant>[_s<seed>]/ .
# Score with (the pass mark is on per-lead and peak rho, which only the first
# of these reports; the second adds MAE, tail reproduction and dispersion):
#   python exploration/2026-08/analysis/score_per_lead.py \
#       --results-dir $SAVE_ROOT --group-seeds --prefix auxprobe[_<tag>]_<target>_<io>_<model>
#   python exploration/2026-08/analysis/compare_loss_variants.py \
#       --results-dir $SAVE_ROOT --prefix auxprobe[_<tag>]_<target>_<io>_<model> --ridge

set -e
set -f  # list/dict Hydra overrides contain [ ] -- keep the shell from globbing them

# This script lives at exploration/<month>/analysis/, so the repo root -- where
# scripts/train.py is -- is three levels up, not one.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_NAME="server_ap"
IO="in12h_out12h"
MODEL="gnn_transformer"
VARIANT=""
SEEDS=1
EXTRA=""
TAG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        --io)          IO="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --variant)     VARIANT="$2"; shift 2 ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        # extra Hydra overrides appended to every run, e.g. for a smoke test:
        #   --extra "training.epochs=1 experiment.batch_size=64"
        --extra)       EXTRA="$2"; shift 2 ;;
        # suffix on every experiment name, so a re-run under different settings
        # lands beside the first one instead of overwriting it. Required
        # whenever --extra changes training: without it the earlier results are
        # destroyed and the comparison they were run for is gone.
        --tag)         TAG="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Changing the training settings without renaming the runs would overwrite the
# results those settings are meant to be compared against. Refuse rather than
# silently destroy them.
if [[ -n "$EXTRA" && -z "$TAG" && "$DRY_RUN" == false ]]; then
    echo "--extra changes the training settings, so the runs need their own names."
    echo "Re-run with --tag <name> (e.g. --tag p30) to keep the existing results."
    exit 1
fi

# ap or hp, taken from the config name (server_ap -> ap).
TARGET="${CONFIG_NAME##*_}"

# The tag goes before the target so a tagged sweep and the untagged one are
# disjoint under --prefix matching, in both directions.
NAME_BASE="auxprobe${TAG:+_$TAG}_${TARGET}_${IO}_${MODEL}"

# The auxiliary head needs targets the table dataset alone can slice.
DATASET_MODE="data.timeseries.dataset_mode=table"

ORDER="baseline aux1 aux05 aux2 aux_mae aux01"

variant_overrides() {
    case "$1" in
        # control: the auxiliary task off, everything else identical
        baseline)  echo "" ;;
        # the probe's setting: equal weighting of every channel, MSE, lambda 1
        aux1)      echo "training.aux_forecast.enabled=true training.aux_forecast.weight=1.0" ;;
        # is the gain sensitive to how loud the auxiliary term is?
        aux05)     echo "training.aux_forecast.enabled=true training.aux_forecast.weight=0.5" ;;
        aux2)      echo "training.aux_forecast.enabled=true training.aux_forecast.weight=2.0" ;;
        # L1 on the auxiliary term: the wind channels are heavy-tailed, and the
        # index head already uses a tier-weighted L2
        aux_mae)   echo "training.aux_forecast.enabled=true training.aux_forecast.weight=1.0 training.aux_forecast.base_loss=mae" ;;
        # diagnostic, not a candidate: the auxiliary term turned down to a tenth.
        # If aux05/aux1/aux2 land together but this one loses the gain, the term
        # is doing work rather than acting as generic regularisation. (The
        # sharper diagnostic -- up-weighting the index channel INSIDE the
        # auxiliary loss, which halved the gain in the MLP probe -- needs a
        # per-channel weight the config does not expose, so it stays in
        # measurements/multitask_direct.py.)
        aux01)     echo "training.aux_forecast.enabled=true training.aux_forecast.weight=0.1" ;;
        *) return 1 ;;
    esac
}

if [[ -n "$VARIANT" ]]; then
    if ! variant_overrides "$VARIANT" >/dev/null 2>&1; then
        echo "Unknown variant '$VARIANT'. Available: $ORDER"; exit 1
    fi
    ORDER="$VARIANT"
fi

echo "=========================================================================="
echo "Aux-forecast probe:  config=$CONFIG_NAME  io=$IO  model=$MODEL"
echo "Variants:            $ORDER"
echo "Seeds per variant:   $SEEDS"
echo "=========================================================================="

run_one() {
    local variant="$1" seed_idx="$2"
    local extra exp seed_override=""
    extra="$(variant_overrides "$variant")"
    exp="${NAME_BASE}_${variant}"
    if [[ "$SEEDS" -gt 1 ]]; then
        exp="${exp}_s${seed_idx}"
        seed_override="experiment.seed=$((250104 + seed_idx))"
    fi

    echo "--- [$variant seed $seed_idx] $exp"
    if $DRY_RUN; then
        echo "    train:    python scripts/train.py --config-name=$CONFIG_NAME +io=$IO +model=$MODEL experiment.name=$exp $DATASET_MODE $extra $seed_override $EXTRA"
        echo "    validate: python scripts/validate.py --config-name=$CONFIG_NAME +io=$IO +model=$MODEL experiment.name=$exp validation.epoch=best validation.mcd_samples=0 $DATASET_MODE $extra $seed_override $EXTRA"
        return
    fi

    # errexit is suspended inside an `if` condition, so fail explicitly: a failed
    # train would otherwise fall through to a validate with no checkpoint.
    # shellcheck disable=SC2086
    python scripts/train.py --config-name="$CONFIG_NAME" +io="$IO" +model="$MODEL" \
        experiment.name="$exp" $DATASET_MODE $extra $seed_override $EXTRA || return 1
    # shellcheck disable=SC2086
    python scripts/validate.py --config-name="$CONFIG_NAME" +io="$IO" +model="$MODEL" \
        experiment.name="$exp" validation.epoch=best validation.mcd_samples=0 \
        $DATASET_MODE $extra $seed_override $EXTRA || return 1
}

# One run failing must not abandon the rest of the sweep.
FAILED=""
for v in $ORDER; do
    for s in $(seq 0 $((SEEDS - 1))); do
        if ! run_one "$v" "$s"; then
            echo "!!! variant '$v' seed $s FAILED -- continuing"
            FAILED="$FAILED ${v}/s${s}"
        fi
    done
done
[[ -n "$FAILED" ]] && echo "Failed runs:$FAILED"

echo
echo "Done. Compare with (substitute your results root; no angle brackets -- the shell would"
echo "read them as a redirect and argparse then reports a missing --results-dir argument):"
echo "  python ${PROBE_DIR#$REPO_ROOT/}/score_per_lead.py \\"
echo "      --results-dir /path/to/results --group-seeds \\"
echo "      --prefix ${NAME_BASE}"
echo "  python ${PROBE_DIR#$REPO_ROOT/}/compare_loss_variants.py \\"
echo "      --results-dir /path/to/results \\"
echo "      --prefix ${NAME_BASE} --ridge"
echo
echo "This probe has run and is CLOSED. 2026-08-07, 6 variants x 2 seeds, then a"
echo "patience-30 re-run of baseline and aux1: per-lead rho +0.0051 against a predicted"
echo "+0.030, peak rho down 0.0039, both pass marks failed. The longer patience changed"
echo "nothing -- 33-35 epochs instead of 13-16, same best epoch, same loss to five"
echo "decimals -- so the early-stopping confound is refuted rather than open."
echo
echo "What replicates is narrow: rho at 12 h moves +0.0185, 11x the baseline seed sd,"
echo "with a clean dose response through aux01. Long-lead discrimination only; the curve,"
echo "the peak and the error do not improve. training.aux_forecast stays off by default."
echo "Record: measurements/README-coauthor-proposals.md."
