"""Monte Carlo Dropout uncertainty analysis.

Evaluates prediction uncertainty: 95% CI coverage, storm-period
coverage, std vs target correlation, and calibration diagrams.

Usage:
    python analysis/evaluate_mcd.py \
        --results-dir /path/to/results \
        --output-dir ./mcd_analysis \
        --filter out12h
"""

import argparse
import csv
import io
import os
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


def load_mcd_npz(results_dir, experiment):
    """Load all MCD npz from a zip archive.

    Returns arrays: all_means, all_stds, all_targets (concatenated).
    """
    zip_path = os.path.join(results_dir, experiment, 'mcd', 'best', 'npz.zip')
    if not os.path.exists(zip_path):
        return None

    all_means = []
    all_stds = []
    all_targets = []

    with zipfile.ZipFile(zip_path) as z:
        npz_files = [n for n in z.namelist() if n.endswith('.npz')]
        for name in npz_files:
            data = np.load(io.BytesIO(z.read(name)))
            all_means.append(data['mean'])      # (target_len,)
            all_stds.append(data['std'])         # (target_len,)
            all_targets.append(data['target'])   # (target_len,)

    if not all_means:
        return None

    return (
        np.concatenate(all_means),
        np.concatenate(all_stds),
        np.concatenate(all_targets),
    )


def compute_coverage(means, stds, targets, n_std=2.0):
    """Compute fraction of targets within ±n_std × std of mean."""
    lower = means - n_std * stds
    upper = means + n_std * stds
    covered = (targets >= lower) & (targets <= upper)
    return covered.mean()


def compute_calibration(means, stds, targets,
                         confidence_levels=None):
    """Compute calibration: expected vs actual coverage at each level.

    Returns list of (expected, actual) tuples.
    """
    if confidence_levels is None:
        confidence_levels = [50, 60, 70, 80, 90, 95, 99]

    results = []
    for conf in confidence_levels:
        z = scipy_stats.norm.ppf((1 + conf / 100) / 2)
        actual = compute_coverage(means, stds, targets, n_std=z)
        results.append((conf / 100, actual))

    return results


def plot_calibration(calibrations, labels, save_path):
    """Plot calibration diagram for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    for cal, label in zip(calibrations, labels):
        expected = [c[0] for c in cal]
        actual = [c[1] for c in cal]
        ax.plot(expected, actual, 'o-', markersize=5, label=label)

    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('MCD Calibration Diagram')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_std_vs_target(stds, targets, label, save_path):
    """Plot std vs target ap30 scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(targets, stds, alpha=0.02, s=1, color='steelblue')

    # Bin averages
    bins = np.arange(0, max(targets.max(), 100), 5)
    bin_means_t = []
    bin_means_s = []
    for i in range(len(bins) - 1):
        mask = (targets >= bins[i]) & (targets < bins[i + 1])
        if mask.sum() > 10:
            bin_means_t.append((bins[i] + bins[i + 1]) / 2)
            bin_means_s.append(stds[mask].mean())

    ax.plot(bin_means_t, bin_means_s, 'ro-', markersize=4,
            linewidth=2, label='Bin average')

    r, p = scipy_stats.pearsonr(targets, stds)
    ax.text(0.02, 0.98, f'r = {r:.3f} (p = {p:.1e})',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Target ap30')
    ax.set_ylabel('Prediction Std')
    ax.set_title(f'Uncertainty vs Target — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return r


def parse_experiment_name(name):
    """Parse experiment name into input, output, model."""
    parts = name.split('_')
    inp = parts[0]
    out = parts[1]
    model = '_'.join(parts[2:]) if len(parts) > 2 else 'linear'
    return inp, out, model


def main():
    parser = argparse.ArgumentParser(
        description='MCD uncertainty analysis'
    )
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output-dir', default='./mcd_analysis')
    parser.add_argument('--filter', default='')
    parser.add_argument('--n-std', type=float, default=2.0,
                        help='Number of std for CI (default: 2.0 ≈ 95.4%%)')
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

    print(f"Analyzing MCD for {len(experiments)} experiments...\n")

    all_results = []
    calibrations_by_group = {}

    for exp in experiments:
        result = load_mcd_npz(args.results_dir, exp)
        if result is None:
            print(f"  [SKIP] {exp} — no MCD zip")
            continue

        means, stds, targets = result
        inp, out, model = parse_experiment_name(exp)

        # Overall coverage
        total_cov = compute_coverage(means, stds, targets, args.n_std)

        # Storm coverage (ap30 >= 30)
        storm_mask = targets >= 30
        storm_count = storm_mask.sum()
        if storm_count > 10:
            storm_cov = compute_coverage(
                means[storm_mask], stds[storm_mask],
                targets[storm_mask], args.n_std
            )
        else:
            storm_cov = float('nan')

        # Std-target correlation
        r, _ = scipy_stats.pearsonr(targets, stds)

        all_results.append({
            'experiment': exp,
            'input': inp,
            'output': out,
            'model': model,
            'total_coverage': total_cov,
            'storm_coverage': storm_cov,
            'std_target_corr': r,
            'mean_std': stds.mean(),
            'total_points': len(targets),
            'storm_points': int(storm_count),
        })

        print(f"  {exp:45s}  cov={total_cov:.3f}  "
              f"storm_cov={storm_cov:.3f}  r={r:.3f}")

        # Calibration per output group
        group_key = f"{inp}_{out}"
        if group_key not in calibrations_by_group:
            calibrations_by_group[group_key] = []
        cal = compute_calibration(means, stds, targets)
        calibrations_by_group[group_key].append((cal, model))

        # Std vs target scatter (only for selected models)
        if model in ('gnn_transformer', 'transformer', 'linear'):
            scatter_path = os.path.join(
                args.output_dir, f'mcd_scatter_{exp}.png'
            )
            plot_std_vs_target(stds, targets, exp, scatter_path)

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'mcd_coverage.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'input', 'output', 'model',
            'total_coverage', 'storm_coverage', 'std_target_corr',
            'mean_std', 'total_points', 'storm_points'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    # Calibration plots per group
    for group_key, cal_list in calibrations_by_group.items():
        cals = [c[0] for c in cal_list]
        labels = [c[1] for c in cal_list]
        save_path = os.path.join(
            args.output_dir, f'mcd_calibration_{group_key}.png'
        )
        plot_calibration(cals, labels, save_path)

    print(f"Calibration plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
