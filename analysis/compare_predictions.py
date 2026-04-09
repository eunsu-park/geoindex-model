"""Multi-model prediction comparison plots.
다중 모델 예측 비교 플롯.

Overlays predictions from multiple models on the same event,
highlighting storm events and quiet periods.

Usage:
    python analysis/compare_predictions.py \
        --results-dir /path/to/results \
        --output-dir ./prediction_plots \
        --config-base in2d_out12h \
        --top-k 5
"""

import argparse
import io
import os
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Model display names and colors
MODEL_STYLES = {
    'linear':           ('Linear', '#999999', '-'),
    'transformer':      ('Transformer', '#1f77b4', '-'),
    'tcn':              ('TCN', '#ff7f0e', '-'),
    'patchtst':         ('PatchTST', '#2ca02c', '-'),
    'timesnet':         ('TimesNet', '#d62728', '-'),
    'gnn_transformer':  ('GNN+Trans', '#9467bd', '-'),
    'gnn_tcn':          ('GNN+TCN', '#8c564b', '--'),
    'gnn_bilstm':       ('GNN+BiLSTM', '#e377c2', '--'),
    'gnn_patchtst':     ('GNN+PatchTST', '#17becf', '--'),
}


def get_model_suffix(experiment, config_base):
    """Extract model suffix from experiment name."""
    if experiment == config_base:
        return 'linear'
    prefix = config_base + '_'
    if experiment.startswith(prefix):
        return experiment[len(prefix):]
    return None


def load_npz_from_zip(zip_path, npz_name):
    """Load a single npz from within a zip archive."""
    with zipfile.ZipFile(zip_path) as z:
        return np.load(io.BytesIO(z.read(npz_name)))


def find_storm_events(zip_path, top_k=5, quiet_k=3):
    """Find top-k storm events and quiet events by max target ap30.

    Returns list of (npz_name, max_ap30) tuples.
    """
    events = []
    with zipfile.ZipFile(zip_path) as z:
        npz_files = [n for n in z.namelist() if n.endswith('.npz')]
        for name in npz_files:
            data = np.load(io.BytesIO(z.read(name)))
            max_ap = data['targets'].max()
            events.append((name, float(max_ap)))

    events.sort(key=lambda x: x[1], reverse=True)

    storm = events[:top_k]
    quiet_candidates = [e for e in events if e[1] < 10]
    if len(quiet_candidates) >= quiet_k:
        step = len(quiet_candidates) // quiet_k
        quiet = quiet_candidates[::step][:quiet_k]
    else:
        quiet = quiet_candidates

    return storm, quiet


def plot_comparison(target, predictions_dict, input_ap30, event_name,
                    save_path, event_type='storm'):
    """Create comparison plot for one event.

    Args:
        target: (target_len,) ground truth ap30
        predictions_dict: {model_name: (target_len,) predictions}
        input_ap30: (input_len,) input ap30 values
        event_name: string for title
        save_path: output path
        event_type: 'storm' or 'quiet'
    """
    input_len = len(input_ap30)
    target_len = len(target)

    # Time axis: input is negative, target is positive
    t_input = np.arange(-input_len, 0)
    t_target = np.arange(0, target_len)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Left: Full view
    ax = axes[0]
    ax.plot(t_input, input_ap30, color='steelblue', alpha=0.5,
            linewidth=1, label='Input (ap30)')
    ax.plot(t_target, target, color='black', linewidth=2,
            marker='o', markersize=2, label='Ground Truth')

    for model_key, pred in predictions_dict.items():
        if model_key in MODEL_STYLES:
            label, color, ls = MODEL_STYLES[model_key]
        else:
            label, color, ls = model_key, None, '-'
        ax.plot(t_target, pred, color=color, linestyle=ls,
                linewidth=1.2, alpha=0.8, label=label)

    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Timestep (relative to T=0)')
    ax.set_ylabel('ap30')
    ax.set_title(f'{event_name} — Full View ({event_type})')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right: Peak zoom (target region only)
    ax2 = axes[1]
    ax2.plot(t_target, target, color='black', linewidth=2,
             marker='o', markersize=3)

    for model_key, pred in predictions_dict.items():
        if model_key in MODEL_STYLES:
            _, color, ls = MODEL_STYLES[model_key]
        else:
            color, ls = None, '-'
        mae = np.mean(np.abs(pred - target))
        ax2.plot(t_target, pred, color=color, linestyle=ls,
                 linewidth=1.2, alpha=0.8)

    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('ap30')
    ax2.set_title('Target Region (zoom)')
    ax2.grid(True, alpha=0.3)

    # Add MAE text box
    mae_text = '\n'.join([
        f"{MODEL_STYLES.get(k, (k,))[0]}: {np.mean(np.abs(v - target)):.1f}"
        for k, v in predictions_dict.items()
    ])
    ax2.text(0.98, 0.98, mae_text, transform=ax2.transAxes,
             fontsize=6, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Multi-model prediction comparison plots'
    )
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output-dir', default='./prediction_plots')
    parser.add_argument('--config-base', default='in2d_out12h',
                        help='Base config name (e.g., in2d_out12h)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of storm events to plot')
    parser.add_argument('--models', default='',
                        help='Comma-separated model list (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find experiments matching config_base
    all_dirs = sorted(os.listdir(args.results_dir))
    experiments = {}
    for d in all_dirs:
        if not d.startswith(args.config_base):
            continue
        model = get_model_suffix(d, args.config_base)
        if model is not None:
            zip_path = os.path.join(args.results_dir, d, 'validation', 'best.zip')
            if os.path.exists(zip_path):
                experiments[model] = zip_path

    if args.models:
        filter_models = set(args.models.split(','))
        experiments = {k: v for k, v in experiments.items() if k in filter_models}

    print(f"Config base: {args.config_base}")
    print(f"Models found: {list(experiments.keys())}")

    if not experiments:
        print("No experiments found!")
        return

    # Find storm/quiet events from first experiment
    first_zip = list(experiments.values())[0]
    storm_events, quiet_events = find_storm_events(first_zip, args.top_k)

    print(f"Storm events: {len(storm_events)} (max ap30: "
          f"{storm_events[0][1]:.0f} ~ {storm_events[-1][1]:.0f})")
    print(f"Quiet events: {len(quiet_events)}")
    print()

    # Plot each selected event
    for event_list, event_type in [(storm_events, 'storm'), (quiet_events, 'quiet')]:
        for npz_name, max_ap in event_list:
            timestamp = os.path.basename(npz_name).replace('.npz', '')

            # Load ground truth and input from first model
            ref_data = load_npz_from_zip(first_zip, npz_name)
            target = ref_data['targets'].squeeze()
            inputs = ref_data['inputs']
            # Extract ap30 from input (last variable, index 21)
            input_ap30 = inputs[:, -1]  # ap30 is last input variable

            # Load predictions from all models
            predictions = {}
            for model_key, zip_path in experiments.items():
                try:
                    data = load_npz_from_zip(zip_path, npz_name)
                    predictions[model_key] = data['predictions'].squeeze()
                except (KeyError, FileNotFoundError):
                    pass

            if not predictions:
                continue

            save_path = os.path.join(
                args.output_dir,
                f'compare_{args.config_base}_{timestamp}_{event_type}.png'
            )
            plot_comparison(target, predictions, input_ap30,
                           f'{timestamp} (max={max_ap:.0f})',
                           save_path, event_type)
            print(f"  [{event_type}] {timestamp} — max ap30={max_ap:.0f}")

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
