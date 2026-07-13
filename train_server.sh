#!/bin/bash
# Launch table-mode training on the GPU server.
#
# Data layout expected under environment.data_root
#   (/home/eunsupark/Storage/geoindex/datasets):
#     data.parquet
#     total_ap/{train_index.csv, validation_index.csv}   # ap30 labels
#     total_hp/{train_index.csv, validation_index.csv}   # hp30 labels
# Results are written under /home/eunsupark/Storage/geoindex/results.
#
# Prereq:  conda activate geoindex
#
# Usage:
#   ./train_server.sh <ap|hp> <profile|io> [model] [hydra overrides...]
#
#   ./train_server.sh ap standard                 # ap30, GNN+Transformer, 2d->12h
#   ./train_server.sh hp standard                 # hp30, same architecture
#   ./train_server.sh ap quick                    # Linear baseline, 1d->6h
#   ./train_server.sh ap in2d_out24h gnn_patchtst # explicit io + model
#   ./train_server.sh hp standard training.epochs=50 experiment.batch_size=256
#
# Profiles (mirror the ./ap CLI):
#   quick    -> io=in1d_out6h   model=linear
#   standard -> io=in2d_out12h  model=gnn_transformer
#   extended -> io=in2d_out24h  model=gnn_transformer

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:?Usage: ./train_server.sh <ap|hp> <profile|io> [model] [overrides...]}"
shift

case "$TARGET" in
    ap) CONFIG_NAME="server_ap" ;;
    hp) CONFIG_NAME="server_hp" ;;
    *)  echo "Error: first argument must be 'ap' or 'hp' (got '$TARGET')"; exit 1 ;;
esac

# Resolve a named profile, or take explicit io + model.
case "${1:?Error: profile or io required}" in
    quick)    IO="in1d_out6h";  MODEL="linear";          shift ;;
    standard) IO="in2d_out12h"; MODEL="gnn_transformer"; shift ;;
    extended) IO="in2d_out24h"; MODEL="gnn_transformer"; shift ;;
    *)
        IO="$1"
        MODEL="${2:?Error: model required when io is given explicitly}"
        shift 2
        ;;
esac

# Prefix experiment name with the target so ap/hp results never collide.
EXP="${TARGET}_${IO}_${MODEL}"

# hp uses SW + hp30 inputs, so the inherited ap30 GNN node must be dropped
# (Hydra deep-merges dicts, so it cannot be removed from within server_hp.yaml).
EXTRA_ARGS=()
if [[ "$TARGET" == "hp" ]]; then
    EXTRA_ARGS+=("~data.timeseries.gnn_variable_groups.ap30")
fi

echo "========================================"
echo "target:      $TARGET"
echo "config-name: $CONFIG_NAME"
echo "io / model:  $IO / $MODEL"
echo "experiment:  $EXP"
echo "overrides:   $*"
echo "========================================"

exec python scripts/train.py \
    --config-name="$CONFIG_NAME" \
    +io="$IO" \
    +model="$MODEL" \
    experiment.name="$EXP" \
    "${EXTRA_ARGS[@]}" \
    "$@"
