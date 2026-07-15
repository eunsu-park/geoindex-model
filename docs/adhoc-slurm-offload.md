# Ad-hoc: Offloading batch analysis to a Slurm cluster

> **Status: AD-HOC / TEMPORARY — not part of the regular pipeline.**
> This is a one-off runbook for moving the *already-trained* analysis workload
> (validation / MCD / attention) off the single RTX 3090 and onto a large
> Slurm HPC cluster for a faster batch pass. Training is assumed complete; this
> covers inference-only analysis. Nothing here is wired into `run_pending.sh`
> or the standard configs. Delete or ignore once the one-off run is done.

## Why this is cheap to move

The analysis path (validation, MCD, attention) runs in **table mode** and
reads only `data.parquet` + index CSVs — **no database is involved**. There is
no `egghouse` / `psycopg` import anywhere in `scripts/validate.py`,
`analysis/run_mcd.py`, or `analysis/run_attention.py`. So the cluster needs
**no DB access, no `egghouse`, no raw downloads**. Only code + a small data
bundle + checkpoints.

> Note: `saliency` is **excluded** — `analysis/run_saliency.py` hard-requires
> SDO image input (`batch["sdo"]`, `model(sw, image)`), which the SW-only table
> models do not provide. It will `KeyError` on every experiment. The runnable
> set is exactly **validation + MCD + attention** (what `run_pending.sh` drives).

## 1. Transfer payload

Small, except for checkpoints.

| Item | Source | Size | Needed |
|------|--------|------|--------|
| Code | `geoindex-model` (git clone or rsync) | small | yes |
| `data.parquet` | `datasets/` | 37 MB | yes |
| `total_ap/`, `total_hp/` (train/val index) | `datasets/` | ~2 MB each | yes |
| `cv5/`, `cv5_hp/` (fold indices + per-fold stats) | `datasets/` | ~7 MB each | only for CV |
| Checkpoints `results/{ap_,hp_}*/checkpoint/model_best.pth` | `results/` | ~few GB | yes |
| DB / `egghouse` / `sw_events` / raw downloads | — | — | **no** |

Only `model_best.pth` per experiment is needed (`--epoch best`); skip all-epoch
checkpoints and optimizer state to shrink the payload. Analysis *outputs*
(`{exp}/{phase}/best/…`) are written fresh on the cluster — do not transfer them.

```bash
# From the GPU server (paths as on that host)
SRC=/home/eunsupark/Storage/geoindex
DST=cluster:$SCRATCH/geoindex          # adjust to the cluster path

# Data bundle
rsync -a  "$SRC/datasets/data.parquet" \
          "$SRC/datasets/total_ap" "$SRC/datasets/total_hp" \
          "$SRC/datasets/cv5" "$SRC/datasets/cv5_hp" \
          "$DST/datasets/"

# Checkpoints only (best epoch), preserving the {exp}/checkpoint/ layout
rsync -a --prune-empty-dirs \
      --include='*/' \
      --include='checkpoint/model_best.pth' \
      --include='checkpoint/table_stats.pkl' \
      --exclude='*' \
      "$SRC/results/" "$DST/results/"
```

## 2. Cluster environment

- **conda env** from `requirements.txt` (torch, numpy, pandas, matplotlib,
  h5py, hydra-core, omegaconf, opencv-python, pyyaml, tqdm, pyarrow). There is
  no `environment.yml`; build from `requirements.txt`.
- **torch**: install the build matching the cluster's CUDA/driver (usually
  after `module load cuda/<ver>`), not a stray pip default.
- **opencv-python** imports `cv2`, which needs the system `libGL` shared lib.
  If the compute nodes lack it, either `module load` a mesa/libGL, or swap to
  `opencv-python-headless` (safe if `cv2` is not actually exercised at runtime).
- **No DB client** needed.

## 3. Path handling (temporary, no repo changes)

`server_ap.yaml` hardcodes `/home/eunsupark/Storage/geoindex/{datasets,results}`.
Cleanest temporary fix — make that path resolve on the cluster via a symlink, so
`server_ap` / `server_hp` work **unmodified** (keeps the `ap_`/`hp_` prefix and
the hp `~…gnn_variable_groups.ap30` drop intact):

```bash
mkdir -p "$SCRATCH/geoindex"
ln -s "$SCRATCH/geoindex" /home/eunsupark/Storage/geoindex   # if home path matches
```

If you cannot recreate that home path, do **not** invent a new config profile
(that would require editing the `case "$CONFIG_NAME"` blocks in every runner).
Instead call the Python entrypoints directly with Hydra overrides
`environment.data_root=… environment.save_root=…` (see the job-array template).

## 4. Stats-file race (MUST fix for a job array)

`server_ap` and `server_hp` both inherit `stat_file: "table_stats.pkl"` from
`base.yaml`, i.e. they share `datasets/table_stats.pkl`. But ap needs `ap30`
and hp needs `hp30`, so `compute_statistics_table` recomputes and **overwrites**
the shared file whenever the variable set does not match. Sequential runs cope;
**hundreds of concurrent array tasks reading/writing the same file will race.**

Statistics are deterministic (train-window mean/min/max), so recomputation
matches training exactly — correctness is fine; only the concurrent file
read/write is the hazard. Fix (pick one):

- **Recommended:** give each target its own stats file via override —
  `data.timeseries.stat_file=total_ap/table_stats.pkl` for ap,
  `total_hp/table_stats.pkl` for hp. Seed both **once** before the array (a
  single ap task + single hp task), after which every array task only **reads**
  them (all vars present → load, no rewrite).
- Or: run one seeding job per target to completion first, then launch the array.

## 5. Execution model

The stock `*.sh` runners use bash `&` (`MAX_JOBS`) parallelism — bound to **one
node, one GPU** (every process uses `device="cuda"` = GPU 0). They do **not**
spread across GPUs.

### Option A — minimal (1 node, 1 GPU)

Reuse the scripts as-is inside one allocation. Simple; does not exploit the
cluster's scale.

```bash
#!/bin/bash
#SBATCH --job-name=geoidx-analysis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
# #SBATCH --partition=<queue>

module load cuda/<ver>          # cluster-specific
source activate geoindex        # or conda activate
cd "$SLURM_SUBMIT_DIR"

RESULTS=/home/eunsupark/Storage/geoindex/results
./run_pending.sh --config-name server_ap --max-jobs 4
SAVE_ROOT=$RESULTS ./run_pending.sh --config-name server_hp --max-jobs 4
```

### Option B — job array (recommended for scale)

One experiment per array task, each with its own GPU. This is where a large
Slurm cluster pays off (hours → minutes). Sketch of what the launcher must do:

1. **Build the task list** once (a text file, one line per task):
   `phase io model target` for every combination —
   - validation: 24 io × 14 model, targets ap + hp
   - mcd:        24 io × 14 model, targets ap + hp
   - attention:  24 io × **4** attention models (`transformer`, `patchtst`,
     `gnn_transformer`, `gnn_patchtst`), targets ap + hp
   (saliency excluded).
2. Map `SLURM_ARRAY_TASK_ID` → the N-th line.
3. **Skip if the marker exists** so requeued/failed tasks resume:
   `results/{prefix}{io}_{model}/{phase}/best/…` — `validation_results.csv`
   for validation, `{phase}/best/npz.zip` for mcd/attention.
4. Call the matching entrypoint with overrides:
   - config-name `server_ap` (ap) or `server_hp` (hp)
   - `+io=<io> +model=<model> experiment.name=<prefix><io>_<model>`
   - `<phase>.epoch=best`
   - hp only: `~data.timeseries.gnn_variable_groups.ap30`
   - per-target stats: `data.timeseries.stat_file=total_<ap|hp>/table_stats.pkl`
   - if not using the symlink: `environment.data_root=… environment.save_root=…`

Reference skeleton (`sbatch --array=0-$((N-1))%<concurrent> run_array.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=geoidx-array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
# #SBATCH --partition=<queue>

module load cuda/<ver>
source activate geoindex
cd "$SLURM_SUBMIT_DIR"

# tasks.txt: one "phase io model target" per line (generated beforehand)
read -r PHASE IO MODEL TGT < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks.txt)

case "$TGT" in
  ap) CN=server_ap; PFX=ap_; EXTRA=() ;;
  hp) CN=server_hp; PFX=hp_; EXTRA=("~data.timeseries.gnn_variable_groups.ap30") ;;
esac
EXP="${PFX}${IO}_${MODEL}"
ROOT=/home/eunsupark/Storage/geoindex/results

# Marker-based skip (idempotent on requeue)
case "$PHASE" in
  validation) MARK="$ROOT/$EXP/validation/best/validation_results.csv"
              RUNNER=scripts/validate.py ;;
  mcd)        MARK="$ROOT/$EXP/mcd/best/npz.zip"
              RUNNER=analysis/run_mcd.py ;;
  attention)  MARK="$ROOT/$EXP/attention/best/npz.zip"
              RUNNER=analysis/run_attention.py ;;
esac
[[ -f "$MARK" ]] && { echo "skip $EXP/$PHASE (exists)"; exit 0; }

python "$RUNNER" --config-name="$CN" \
    +io="$IO" +model="$MODEL" experiment.name="$EXP" \
    "${PHASE}.epoch=best" \
    data.timeseries.stat_file="total_${TGT}/table_stats.pkl" \
    "${EXTRA[@]}"
```

Tune `%<concurrent>` to the GPU allocation you are granted. Attention on a
non-attention model is a no-op/skip inside `run_attention.py`, but the task
list above already restricts attention to the 4 attention models to avoid
wasting array slots.

## 6. Retrieve results

```bash
rsync -a --prune-empty-dirs \
      --include='*/' \
      --include='validation/***' --include='mcd/***' --include='attention/***' \
      --exclude='*' \
      cluster:$SCRATCH/geoindex/results/ ./results/
```

Then aggregate locally (`scripts/aggregate_cv_results.py`, model-comparison
report, etc.).

## 7. Pre-flight checklist (fill in cluster specifics)

- [ ] Partition/queue name, GPU type, per-task time & CPU limits
- [ ] Home path is `/home/eunsupark` (symlink OK) **or** decide on
      `data_root`/`save_root` overrides
- [ ] `module load` / conda activation recipe (cuda version, anaconda module)
- [ ] `opencv-python` import works on compute nodes (else `-headless`)
- [ ] Per-target `table_stats.pkl` seeded before launching the array
- [ ] `logs/` dir exists for `%A_%a.out`
