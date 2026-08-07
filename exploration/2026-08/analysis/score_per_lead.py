"""Score validation runs on per-lead rho and peak rho.

compare_loss_variants.py reports MAE, tail reproduction and dispersion, which
is what the damping work needed. The discrimination questions -- does a change
make the model rank anchors better -- need per-lead correlation and the
correlation of the predicted 12 h maximum with the observed one, and those are
the numbers the rest of this investigation quotes.

Reads validation_results.csv, whose file_name column is
"<anchor>_group<lead>_<target>", so the lead index comes straight out of it.

Usage:
    python exploration/2026-08/analysis/score_per_lead.py \
        --results-dir ~/Projects/GeoIndex/results \
        --prefix auxprobe_ap_in12h_out12h_gnn_transformer
    # group seeds of the same variant and report mean +- spread:
    python .../score_per_lead.py --results-dir ... --prefix ... --group-seeds
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def score_one(csv_path):
    """Return per-lead rho array, plus curve and peak statistics."""
    df = pd.read_csv(csv_path,
                     usecols=['file_name', 'target_raw', 'prediction_raw'])
    parts = df['file_name'].str.extract(r'^(?P<anchor>\d+)_group(?P<lead>\d+)_')
    df['anchor'] = parts['anchor']
    df['lead'] = parts['lead'].astype(int)

    obs = df.pivot(index='anchor', columns='lead', values='target_raw')
    prd = df.pivot(index='anchor', columns='lead', values='prediction_raw')
    obs, prd = obs.to_numpy(), prd.to_numpy()

    rho = np.array([np.corrcoef(prd[:, k], obs[:, k])[0, 1]
                    for k in range(obs.shape[1])])
    peak_o, peak_p = obs.max(1), prd.max(1)
    return {
        'n': obs.shape[0],
        'leads': obs.shape[1],
        'per_lead': rho,
        'mean_rho': rho.mean(),
        'rho_6h': rho[11] if len(rho) > 11 else np.nan,
        'rho_12h': rho[-1],
        'peak_rho': np.corrcoef(peak_p, peak_o)[0, 1],
        'mae': np.abs(prd - obs).mean(),
        'rmse': np.sqrt(((prd - obs) ** 2).mean()),
        'sd_ratio': np.mean([prd[:, k].std() / obs[:, k].std()
                             for k in range(obs.shape[1])]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--epoch', default='best')
    ap.add_argument('--group-seeds', action='store_true',
                    help='average runs whose names differ only by a _s<N> suffix')
    args = ap.parse_args()

    pattern = os.path.join(os.path.expanduser(args.results_dir),
                           f'{args.prefix}*', 'validation', args.epoch,
                           'validation_results.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'no validation CSVs matched {pattern}')

    rows = {}
    for path in paths:
        run = path.split(os.sep)[-4]
        name = run[len(args.prefix):].lstrip('_')
        print(f'scoring {name} ...', flush=True)
        rows[name] = score_one(path)

    if args.group_seeds:
        groups = {}
        for name, res in rows.items():
            base = re.sub(r'_s\d+$', '', name)
            groups.setdefault(base, []).append(res)
        print(f'\n{"variant":>12}{"seeds":>7}{"mean rho":>20}{"rho@12h":>20}'
              f'{"peak rho":>20}{"MAE":>10}')
        order = sorted(groups, key=lambda g: -np.mean([r['mean_rho']
                                                       for r in groups[g]]))
        for base in order:
            rs = groups[base]
            def ms(key):
                v = np.array([r[key] for r in rs])
                return f'{v.mean():.4f} +-{v.std():.4f}'
            print(f'{base:>12}{len(rs):>7}{ms("mean_rho"):>20}'
                  f'{ms("rho_12h"):>20}{ms("peak_rho"):>20}'
                  f'{np.mean([r["mae"] for r in rs]):>10.3f}')
        return

    print(f'\n{"run":>16}{"n":>8}{"mean rho":>10}{"rho@6h":>9}{"rho@12h":>9}'
          f'{"peak rho":>10}{"MAE":>9}{"RMSE":>9}{"sd ratio":>10}')
    for name in sorted(rows, key=lambda n: -rows[n]['mean_rho']):
        r = rows[name]
        print(f'{name:>16}{r["n"]:>8}{r["mean_rho"]:>10.4f}{r["rho_6h"]:>9.4f}'
              f'{r["rho_12h"]:>9.4f}{r["peak_rho"]:>10.4f}{r["mae"]:>9.3f}'
              f'{r["rmse"]:>9.3f}{r["sd_ratio"]:>10.3f}')


if __name__ == '__main__':
    main()
