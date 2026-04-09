"""Storm-period filtered performance evaluation.
폭풍 구간 필터링 성능 평가.

Computes MAE, RMSE, and bias for each NOAA G-Scale tier
from denormalized validation predictions.

Usage:
    python analysis/evaluate_storm_performance.py \
        --results-dir /path/to/results \
        --output-dir ./storm_analysis \
        --filter out12h
"""

import argparse
import csv
import io
import os
import zipfile
from collections import defaultdict

import numpy as np


# NOAA G-Scale AP tiers (raw ap30 values)
AP_TIERS = {
    'none':    (0, 29),
    'g1':      (30, 49),
    'g2':      (50, 99),
    'g3_plus': (100, float('inf')),
}

TIER_LABELS = {
    'none':    'None (Kp<5)',
    'g1':      'G1 Minor (Kp=5)',
    'g2':      'G2 Moderate (Kp=6)',
    'g3_plus': 'G3+ Strong (Kp≥7)',
}


def load_validation_npz(results_dir, experiment):
    """Load all validation npz from a zip archive.

    Returns list of (targets, predictions) arrays, both denormalized.
    """
    zip_path = os.path.join(results_dir, experiment, 'validation', 'best.zip')
    if not os.path.exists(zip_path):
        return None

    all_targets = []
    all_preds = []

    with zipfile.ZipFile(zip_path) as z:
        npz_files = [n for n in z.namelist() if n.endswith('.npz')]
        for name in npz_files:
            data = np.load(io.BytesIO(z.read(name)))
            targets = data['targets'].squeeze()    # (target_len,)
            preds = data['predictions'].squeeze()  # (target_len,)
            all_targets.append(targets)
            all_preds.append(preds)

    if not all_targets:
        return None

    return np.concatenate(all_targets), np.concatenate(all_preds)


def compute_metrics(targets, predictions):
    """Compute MAE, RMSE, bias from arrays."""
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    return mae, rmse, bias


def evaluate_experiment(results_dir, experiment):
    """Evaluate one experiment across all AP tiers.

    Returns dict of {tier: {mae, rmse, bias, count}}.
    """
    result = load_validation_npz(results_dir, experiment)
    if result is None:
        return None

    targets, preds = result
    metrics = {}

    # Overall metrics
    mae, rmse, bias = compute_metrics(targets, preds)
    metrics['all'] = {
        'mae': mae, 'rmse': rmse, 'bias': bias, 'count': len(targets)
    }

    # Per-tier metrics
    for tier_name, (lo, hi) in AP_TIERS.items():
        mask = (targets >= lo) & (targets <= hi)
        count = mask.sum()
        if count > 0:
            mae, rmse, bias = compute_metrics(targets[mask], preds[mask])
            metrics[tier_name] = {
                'mae': mae, 'rmse': rmse, 'bias': bias, 'count': int(count)
            }
        else:
            metrics[tier_name] = {
                'mae': 0, 'rmse': 0, 'bias': 0, 'count': 0
            }

    # Storm aggregate (G1+)
    storm_mask = targets >= 30
    storm_count = storm_mask.sum()
    if storm_count > 0:
        mae, rmse, bias = compute_metrics(targets[storm_mask], preds[storm_mask])
        metrics['storm_all'] = {
            'mae': mae, 'rmse': rmse, 'bias': bias, 'count': int(storm_count)
        }
    else:
        metrics['storm_all'] = {'mae': 0, 'rmse': 0, 'bias': 0, 'count': 0}

    return metrics


def parse_experiment_name(name):
    """Parse experiment name into input, output, model."""
    parts = name.split('_')
    inp = parts[0]   # in1d, in2d, in3d
    out = parts[1]   # out6h, out12h, out24h
    model = '_'.join(parts[2:]) if len(parts) > 2 else 'linear'
    return inp, out, model


def main():
    parser = argparse.ArgumentParser(
        description='Storm-period filtered performance evaluation'
    )
    parser.add_argument('--results-dir', required=True,
                        help='Root results directory')
    parser.add_argument('--output-dir', default='./storm_analysis',
                        help='Output directory')
    parser.add_argument('--filter', default='',
                        help='Filter pattern for experiment names')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect experiments
    experiments = sorted([
        d for d in os.listdir(args.results_dir)
        if d.startswith('in') and os.path.isdir(
            os.path.join(args.results_dir, d)
        )
    ])

    if args.filter:
        experiments = [e for e in experiments if args.filter in e]

    print(f"Evaluating {len(experiments)} experiments...")
    print()

    # Evaluate all
    all_results = []
    for exp in experiments:
        metrics = evaluate_experiment(args.results_dir, exp)
        if metrics is None:
            print(f"  [SKIP] {exp} — no validation zip")
            continue

        inp, out, model = parse_experiment_name(exp)
        for tier, m in metrics.items():
            all_results.append({
                'experiment': exp,
                'input': inp,
                'output': out,
                'model': model,
                'tier': tier,
                'mae': m['mae'],
                'rmse': m['rmse'],
                'bias': m['bias'],
                'count': m['count'],
            })

        # Print summary for this experiment
        all_m = metrics['all']
        storm_m = metrics['storm_all']
        storm_pct = storm_m['count'] / all_m['count'] * 100 if all_m['count'] > 0 else 0
        print(f"  {exp:45s}  all_MAE={all_m['mae']:6.2f}  "
              f"storm_MAE={storm_m['mae']:6.2f}  "
              f"storm%={storm_pct:4.1f}%")

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'storm_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'input', 'output', 'model', 'tier',
            'mae', 'rmse', 'bias', 'count'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    # Generate summary table (grouped by output + model)
    summary_path = os.path.join(args.output_dir, 'storm_summary.txt')
    with open(summary_path, 'w') as f:
        for out in ['out6h', 'out12h', 'out24h']:
            f.write(f"\n{'='*80}\n")
            f.write(f"  {out} — Storm Performance (ap30 ≥ 30)\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Model':<28s} {'Input':>5s}  "
                    f"{'All MAE':>8s} {'Storm MAE':>10s} {'Storm RMSE':>11s} "
                    f"{'Storm Bias':>11s} {'Storm %':>8s}\n")
            f.write('-' * 95 + '\n')

            subset = [r for r in all_results
                      if r['output'] == out and r['tier'] in ('all', 'storm_all')]

            # Group by model + input
            groups = defaultdict(dict)
            for r in subset:
                key = (r['model'], r['input'])
                groups[key][r['tier']] = r

            for (model, inp), tiers in sorted(groups.items()):
                if 'all' in tiers and 'storm_all' in tiers:
                    a = tiers['all']
                    s = tiers['storm_all']
                    pct = s['count'] / a['count'] * 100 if a['count'] > 0 else 0
                    f.write(f"{model:<28s} {inp:>5s}  "
                            f"{a['mae']:8.2f} {s['mae']:10.2f} {s['rmse']:11.2f} "
                            f"{s['bias']:11.2f} {pct:7.1f}%\n")

    print(f"Saved: {summary_path}")


if __name__ == '__main__':
    main()
