#!/bin/bash
# Loss / target-transform probe: train one io x model config under several loss and target
# variants, then validate each, to test whether the storm under-prediction (predictions
# reproduce ~39% of observed ap at >=100 ap) is caused by the symmetric squared-error
# objective in log1p space rather than by a lack of information in the inputs.
#
# MC-dropout is switched off during validation (validation.mcd_samples=0) -- the probe only
# needs the deterministic forecast, and that makes each validation ~100x cheaper.
#
# Usage:
#   ./analysis/loss_probe.sh --config-name server_ap                 # all variants
#   ./analysis/loss_probe.sh --config-name server_ap --variant pinball_q75
#   ./analysis/loss_probe.sh --io in6h_out6h --model gnn_transformer
#   ./analysis/loss_probe.sh --dry-run
#
# Results land in $SAVE_ROOT/probe_<target>_<io>_<model>_<variant>/ .
# Compare with: python analysis/compare_loss_variants.py --results-dir $SAVE_ROOT

set -e
set -f  # the list/dict Hydra overrides contain [ ] -- keep the shell from globbing them

# This script was written at analysis/ in the repo root and later moved under
# exploration/<month>/analysis/, so the root is three levels up, not one.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_NAME="server_ap"
IO="in12h_out12h"
MODEL="gnn_transformer"
VARIANT=""
MAX_JOBS=4
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        --io)          IO="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --variant)     VARIANT="$2"; shift 2 ;;
        --max-jobs)    MAX_JOBS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ap or hp, taken from the config name (server_ap -> ap).
TARGET="${CONFIG_NAME##*_}"
TVAR="${TARGET}30"

# Input list without the autoregressive target channel (variant: no_target_channel).
# The list override replaces outright, but a dict override MERGES into the existing node --
# passing the seven solar-wind groups would leave the ap30 group in place and trip
# build_gnn_node_groups ("GNN group 'ap30' contains variable 'ap30' not found in
# input_variables"). Delete that one key instead. Tilde expansion does not apply to the
# result of a variable expansion, so the ~ survives unquoted.
SW_VARS="[v_avg,v_min,v_max,np_avg,np_min,np_max,t_avg,t_min,t_max,bx_avg,bx_min,bx_max,by_avg,by_min,by_max,bz_avg,bz_min,bz_max,bt_avg,bt_min,bt_max]"
SW_DROP_GROUP="~data.timeseries.gnn_variable_groups.${TVAR}"

ORDER="baseline peakhead1 peakhead05 peakhead_mae peak1 peak05 peak_soft peak_mae notemporal temporal_rev tier_off patience25 leadnorm leadnorm_mae exceed50 exceed30 mse mae sww_l1 sww_strong pinball_q75 pinball_q90 target_zscore no_target_channel"

# variant name -> extra Hydra overrides (case, not an associative array: bash 3.2 compat)
variant_overrides() {
    case "$1" in
        # control: what every current run used
        baseline)     echo "" ;;
        # --- loss/optimization surgery -------------------------------------------------
        # The temporal weight is linspace(0.5, 1.0) over the horizon, so the LEAST
        # predictable leads carry twice the weight of the most predictable ones; combined
        # with the tier weights the span is 16:1 in favour of (long lead x storm), the
        # cells where a constant beats the model. These four separate the culprits.
        # tier ON, temporal OFF
        notemporal)   echo "training.solar_wind_weighted.combine_temporal=false" ;;
        # tier ON, temporal reversed (emphasise the predictable short leads)
        temporal_rev) echo "training.solar_wind_weighted.temporal_weight_range=[1.0,0.5]" ;;
        # tier OFF, temporal ON -- threshold mode with high_weight=1.0 makes every ap
        # weight 1.0, neutralising the tiers while the temporal factor still applies
        tier_off)     echo "training.solar_wind_weighted.weighting_mode=threshold training.solar_wind_weighted.high_weight=1.0" ;;
        # every run so far early-stopped at epoch 14-15 (patience 10 -> best epoch ~5 of 100)
        patience25)   echo "training.early_stopping_patience=25" ;;
        # Removing the tier weights alone still left the models on a worse MAE/dispersion
        # frontier than a per-lead ridge. These equalise each lead's contribution, which is
        # the property the ridge gets for free by fitting every lead separately.
        leadnorm)     echo "training.regression_loss_type=lead_normalized" ;;
        leadnorm_mae) echo "training.regression_loss_type=lead_normalized training.lead_normalized.base_loss=mae" ;;
        # Taking the max of 24 separately shrunk conditional means discounts the extremeness
        # twice. Fitting the block maximum as its own quantity raises the correlation with the
        # observed 12-h peak from 0.572 to 0.686 and tail reproduction from 0.411 to 0.569 in a
        # ridge on the same inputs, at an unchanged dispersion-to-rho ratio -- discrimination,
        # not inflation. These add that supervision to the existing outputs -- and that turned
        # out to be the flaw: with no timing information in the peak term, the model satisfied
        # it by making output step 21 a maximum register (99.7 % of anchors peak at 11.0 h)
        # and flattening the rest. Superseded by peakhead*; kept for reproducibility.
        # The corrected form: a separate scalar head, so the peak term never reaches the 24
        # curve outputs and cannot be satisfied by spiking one of them. The peak* variants
        # below are kept because they are what motivated this, not because they are usable.
        peakhead1)    echo "training.peak_head.enabled=true training.peak_head.weight=1.0" ;;
        peakhead05)   echo "training.peak_head.enabled=true training.peak_head.weight=0.5" ;;
        peakhead_mae) echo "training.peak_head.enabled=true training.peak_head.weight=1.0 training.peak_head.base_loss=mae" ;;
        peak1)        echo "training.regression_loss_type=peak_augmented training.peak.weight=1.0" ;;
        peak05)       echo "training.regression_loss_type=peak_augmented training.peak.weight=0.5" ;;
        peak_soft)    echo "training.regression_loss_type=peak_augmented training.peak.weight=1.0 training.peak.soft_tau=0.5" ;;
        peak_mae)     echo "training.regression_loss_type=peak_augmented training.peak.weight=1.0 training.peak.base_loss=mae" ;;
        # Residual spread rises 5x across the prediction deciles (rank correlation +0.96), so
        # P(Y>=t|X) is not monotone in the predicted mean and no refitted threshold on it is
        # Bayes-optimal. These read the output as a per-lead exceedance logit instead.
        # They emit probabilities: score them with the warning tools, not the regression
        # metrics, which do not apply.
        exceed50)     echo "training.regression_loss_type=exceedance_bce training.exceedance.threshold=50.0" ;;
        exceed30)     echo "training.regression_loss_type=exceedance_bce training.exceedance.threshold=30.0" ;;
        # symmetric losses without the NOAA tier weighting -- isolates the weighting's contribution
        mse)          echo "training.regression_loss_type=mse" ;;
        mae)          echo "training.regression_loss_type=mae" ;;
        # keep the tier weighting, swap the inner loss (L1 is less mean-seeking than L2)
        sww_l1)       echo "training.solar_wind_weighted.base_loss=mae" ;;
        # push the tier weighting much harder
        sww_strong)   echo "training.solar_wind_weighted.high_weight=40.0 training.solar_wind_weighted.alpha=10.0" ;;
        # asymmetric point loss: penalize under-prediction more than over-prediction
        pinball_q75)  echo "training.regression_loss_type=pinball training.pinball.quantile=0.75" ;;
        pinball_q90)  echo "training.regression_loss_type=pinball training.pinball.quantile=0.90" ;;
        # drop the log1p compression from the target transform (stats pkl already carries mean/std)
        target_zscore) echo "data.timeseries.normalization.methods.${TVAR}=zscore" ;;
        # not a loss variant: the masking ablation showed the target channel hurts 12h-input models
        no_target_channel) echo "data.timeseries.input_variables=${SW_VARS} ${SW_DROP_GROUP}" ;;
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
echo "Loss probe:  config=$CONFIG_NAME  io=$IO  model=$MODEL  target=$TVAR"
echo "Variants:    $ORDER"
echo "=========================================================================="

run_one() {
    local variant="$1"
    local extra
    extra="$(variant_overrides "$variant")"
    local exp="probe_${TARGET}_${IO}_${MODEL}_${variant}"

    echo "--- [$variant] $exp"
    if $DRY_RUN; then
        echo "    train:    python scripts/train.py --config-name=$CONFIG_NAME +io=$IO +model=$MODEL experiment.name=$exp $extra"
        echo "    validate: python scripts/validate.py --config-name=$CONFIG_NAME +io=$IO +model=$MODEL experiment.name=$exp validation.epoch=best validation.mcd_samples=0 $extra"
        return
    fi

    # errexit is suspended inside an `if` condition, so fail explicitly: a failed train
    # would otherwise fall through to a validate with no checkpoint.
    # shellcheck disable=SC2086
    python scripts/train.py --config-name="$CONFIG_NAME" +io="$IO" +model="$MODEL" \
        experiment.name="$exp" $extra || return 1
    # shellcheck disable=SC2086
    python scripts/validate.py --config-name="$CONFIG_NAME" +io="$IO" +model="$MODEL" \
        experiment.name="$exp" validation.epoch=best validation.mcd_samples=0 $extra || return 1
}

# One variant failing must not abandon the rest of the sweep.
FAILED=""
for v in $ORDER; do
    if ! run_one "$v"; then
        echo "!!! variant '$v' FAILED -- continuing"
        FAILED="$FAILED $v"
    fi
done
[[ -n "$FAILED" ]] && echo "Failed variants:$FAILED"

echo
echo "Done. Compare with (substitute your results root; no angle brackets -- the shell would"
echo "read them as a redirect and argparse then reports a missing --results-dir argument):"
echo "  python exploration/2026-08/analysis/compare_loss_variants.py --results-dir /path/to/results \\"
echo "      --prefix probe_${TARGET}_${IO}_${MODEL} --ridge"
