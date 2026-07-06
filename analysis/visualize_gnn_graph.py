"""GNN learned adjacency matrix visualization.

Extracts and visualizes the adaptive adjacency matrix from GNN
checkpoints without requiring full model/config reconstruction.

Usage:
    python analysis/visualize_gnn_graph.py \
        --results-dir /path/to/results \
        --output-dir ./gnn_graphs \
        --configs in2d_out6h,in2d_out12h,in2d_out24h
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


NODE_LABELS = ['V', 'Np', 'T', 'Bx', 'By', 'Bz', 'Bt', 'ap30']

# Physical expectations for validation
EXPECTED_STRONG_EDGES = {
    ('Bz', 'ap30'): 'Southward IMF drives storms',
    ('V', 'ap30'):   'High-speed solar wind',
    ('Bt', 'ap30'):  'IMF magnitude',
}


def extract_adjacency(checkpoint_path):
    """Extract adjacency matrix from GNN checkpoint.

    Computes A = softmax(relu(E1 @ E2^T)) from node embeddings.
    Does not require model reconstruction.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']

    # Find node embedding keys
    e1_key = None
    e2_key = None
    for key in state:
        if 'node_embed1' in key:
            e1_key = key
        if 'node_embed2' in key:
            e2_key = key

    if e1_key is None or e2_key is None:
        raise ValueError("No GNN node embeddings found in checkpoint")

    e1 = state[e1_key]  # (num_nodes, embed_dim)
    e2 = state[e2_key]  # (num_nodes, embed_dim)

    adj = F.softmax(F.relu(e1 @ e2.T), dim=1)
    return adj.numpy()


def plot_single_heatmap(adj, title, save_path):
    """Plot single adjacency matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(adj, cmap='YlOrRd', vmin=0, vmax=adj.max())
    ax.set_xticks(range(len(NODE_LABELS)))
    ax.set_yticks(range(len(NODE_LABELS)))
    ax.set_xticklabels(NODE_LABELS, fontsize=10)
    ax.set_yticklabels(NODE_LABELS, fontsize=10)

    # Annotate values
    for i in range(len(NODE_LABELS)):
        for j in range(len(NODE_LABELS)):
            color = 'white' if adj[i, j] > adj.max() * 0.6 else 'black'
            ax.text(j, i, f'{adj[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, label='Edge Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(adjs, configs, save_path):
    """Plot side-by-side adjacency matrices for comparison."""
    n = len(adjs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    vmax = max(a.max() for a in adjs)

    for ax, adj, cfg in zip(axes, adjs, configs):
        im = ax.imshow(adj, cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(NODE_LABELS)))
        ax.set_yticks(range(len(NODE_LABELS)))
        ax.set_xticklabels(NODE_LABELS, fontsize=8)
        ax.set_yticklabels(NODE_LABELS, fontsize=8)

        for i in range(len(NODE_LABELS)):
            for j in range(len(NODE_LABELS)):
                color = 'white' if adj[i, j] > vmax * 0.6 else 'black'
                ax.text(j, i, f'{adj[i, j]:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

        # Extract output length from config name
        out = cfg.split('_')[1] if '_' in cfg else cfg
        ax.set_title(out, fontsize=11)

    fig.suptitle('GNN Learned Adjacency', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def validate_physics(adj, config_name):
    """Check if learned graph matches physical expectations."""
    lines = [f"\n=== Physical Validation: {config_name} ===\n"]

    for (src_name, tgt_name), desc in EXPECTED_STRONG_EDGES.items():
        src_idx = NODE_LABELS.index(src_name)
        tgt_idx = NODE_LABELS.index(tgt_name)
        weight = adj[src_idx, tgt_idx]

        # Rank among all edges from src
        rank = (adj[src_idx] >= weight).sum()
        lines.append(f"  {src_name:>4s} → {tgt_name:<5s}: "
                     f"weight={weight:.4f}, rank={rank}/{len(NODE_LABELS)}  "
                     f"({desc})")

    # Top 5 strongest edges overall
    lines.append(f"\n  Top 5 strongest edges:")
    flat = [(adj[i, j], NODE_LABELS[i], NODE_LABELS[j])
            for i in range(len(NODE_LABELS))
            for j in range(len(NODE_LABELS)) if i != j]
    flat.sort(reverse=True)
    for w, s, t in flat[:5]:
        lines.append(f"    {s:>4s} → {t:<5s}: {w:.4f}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='GNN learned adjacency matrix visualization'
    )
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output-dir', default='./gnn_graphs')
    parser.add_argument('--configs',
                        default='in2d_out6h_gnn_transformer,'
                                'in2d_out12h_gnn_transformer,'
                                'in2d_out24h_gnn_transformer',
                        help='Comma-separated experiment names '
                             '(full names, e.g., in2d_out12h_gnn_transformer)')
    parser.add_argument('--epoch', default='best',
                        help='Checkpoint epoch: "best", "final", or number '
                             '(e.g., 10 → model_epoch_0010.pth)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = [c.strip() for c in args.configs.split(',')]
    adjs = []
    valid_configs = []
    stats_lines = []

    # Resolve checkpoint filename from --epoch
    if args.epoch == 'best':
        ckpt_filename = 'model_best.pth'
    elif args.epoch == 'final':
        ckpt_filename = 'model_final.pth'
    else:
        ckpt_filename = f'model_epoch_{int(args.epoch):04d}.pth'

    for exp_name in configs:
        ckpt_path = os.path.join(
            args.results_dir, exp_name, 'checkpoint', ckpt_filename
        )

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {exp_name} — checkpoint not found")
            continue

        print(f"  Loading: {exp_name}")
        adj = extract_adjacency(ckpt_path)
        adjs.append(adj)
        valid_configs.append(exp_name)

        # Individual heatmap
        save_path = os.path.join(args.output_dir, f'gnn_graph_{exp_name}.png')
        plot_single_heatmap(adj, f'GNN Graph — {exp_name}', save_path)

        # Physical validation
        stats_lines.append(validate_physics(adj, exp_name))

    if not adjs:
        print("No checkpoints found!")
        return

    # Comparison plot
    if len(adjs) > 1:
        save_path = os.path.join(args.output_dir, 'gnn_graph_comparison.png')
        plot_comparison(adjs, valid_configs, save_path)
        print(f"\nComparison plot: {save_path}")

    # Save stats
    stats_path = os.path.join(args.output_dir, 'gnn_graph_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats_lines))
    print(f"Stats: {stats_path}")


if __name__ == '__main__':
    main()
