# Retrain Sweep — August 2026 (seed-paired direct + recursive)

Two sweeps on the GPU server, both ap30-side only (no hp30), both with
`experiment.seed=250104` (already fixed in `configs/base.yaml` and applied by
`setup_seed` to python/numpy/torch/cuda; cudnn deterministic).

| Sweep | Profile | Experiments | Target head | Loss |
|---|---|---|---|---|
| A — direct, short-horizon | `server_ap` | in {6h,12h,18h,1d} × out {1h..6h} × 14 models = **336**, names `ap_{io}_{model}` | ap30 (1 ch) | solar_wind_weighted |
| B — recursive | `server_ap_recursive` | 6 in-lengths × out6h × 14 models = **84**, names `ap_recursive_{io}_{model}` | all 22 input channels | mse |

Sweep A (revised 2026-08-12) is the short-horizon grid: 20 new io configs
`in{6h,12h,18h,1d}_out{1..5}h` plus the existing `*_out6h` four. `train.sh`
and `run_pending.sh` apply this grid automatically for `server_ap`; the
legacy long-horizon grid (out12h/18h/24h, in2d/3d) remains available via an
explicit `--filter`, and stays the scan matrix for `server_hp`.

Sweep B (`configs/server_ap_recursive.yaml`) predicts every input variable
over a 6-h chunk so the chunk can be fed back for an iterated rollout to
longer leads (E1 deep arm of the preregistration). `train.sh` and
`run_pending.sh` automatically restrict it to the `*_out6h` io windows and
prefix its experiment names with `ap_recursive_`, so it never collides with
the direct `ap_*` results.

## 0. Before launching sweep A — archive the previous mainline results

With the short-horizon grid, sweep A collides with existing results only on
the four `*_out6h` io combos (`ap_in{6h,12h,18h,1d}_out6h_{model}`, 56 dirs)
under `/home/eunsupark/Projects/GeoIndex/results`. Archive those before
launching (CV fold dirs `*_fold[1-5]` stay; the legacy long-horizon results
that are not being retrained also stay):

```bash
cd /home/eunsupark/Projects/GeoIndex/results
mkdir -p _archive_pre-retrain-2026-08
for io in in6h_out6h in12h_out6h in18h_out6h in1d_out6h; do
    for d in ap_${io}_*; do
        [[ "$d" == *_fold* ]] && continue  # keep CV fold results
        [[ -d "$d" ]] && mv "$d" _archive_pre-retrain-2026-08/
    done
done
ls _archive_pre-retrain-2026-08 | wc -l    # expect 56
```

This is a rename inside the cloud-synced tree (cheap for the sync client)
and fully reversible.

## 1. Launch (GPU server, `conda activate geoindex`)

```bash
cd ~/GitHub/njit-geoindex/geoindex-model   # or the server checkout path

# Sweep A — 336 direct models
./train.sh --config-name server_ap --max-jobs 4

# Sweep B — 84 recursive models
./train.sh --config-name server_ap_recursive --max-jobs 4
```

`--max-jobs 4` matches `environment.num_workers: 4` in `server_ap.yaml`
(4 jobs × 4 loader workers on the single RTX 3090). Sanity-check the queue
first with `--dry-run` (A prints 336 configs, B prints 84).

Logs: `~/tmp/train_logs/{experiment}.log`.

## 2. After training — validation

```bash
./run_pending.sh --config-name server_ap           --epoch best --max-jobs 4
./run_pending.sh --config-name server_ap_recursive --epoch best --max-jobs 4
```

## Notes and caveats

- **Stats cache.** Both sweeps read `table_stats_ap.pkl` (existing, computed
  over the same 22 variables); nothing is recomputed or overwritten.
- **Loss for sweep B.** `solar_wind_weighted` denormalizes with
  `target_variables[0]`'s stats and applies ap-tier weighting to the whole
  target tensor — wrong for a 22-channel mixed-normalization target, hence
  plain MSE in normalized space (equal per-channel weighting).
- **Validation headline for sweep B.** The per-variable metrics cover all 22
  channels, but anything keyed to `target_variables[0]` (e.g. the MC-dropout
  calibration block) refers to `v_avg`, not ap30 — ap30 is the **last**
  channel (index 21). Rollout evaluation should read the ap30 channel
  explicitly.
- **Channel order.** Recursive targets are in exactly the input-variable
  order, so a rollout appends the predicted chunk to the input window with
  no reindexing.
- **Smoke test.** All 14 architectures build, forward, and backprop with the
  22-channel head at in6h/in2d (verified 2026-08-12 on CPU).
