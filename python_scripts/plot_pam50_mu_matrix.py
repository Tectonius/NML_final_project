import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib import cm
import argparse
import os

SUBTYPE_ORDER = [
    'LumA',
    'LumB',
    'Her2',
    'Basal',
    'claudin-low',
    'Normal',
]
SUBTYPE_COLORS = {
    'LumA':        '#4575b4',
    'LumB':        '#e6ab02',
    'Her2':        '#e7298a',
    'Basal':       '#00BFFF',
    'claudin-low': '#1b9e77',
    'Normal':      '#999999',
}


def compute_prototype_distances(lattice):
    rows, cols, _ = lattice.shape
    max_d = 0
    for i in range(rows):
        for j in range(cols):
            if i < rows - 1:
                d = np.linalg.norm(lattice[i, j, :] - lattice[i + 1, j, :])
                if d > max_d:
                    max_d = d
            if j < cols - 1:
                d = np.linalg.norm(lattice[i, j, :] - lattice[i, j + 1, :])
                if d > max_d:
                    max_d = d
    return max_d


def main():
    parser = argparse.ArgumentParser(
        description='Plot mU-matrix with PAM50 + Claudin-low bar chart overlay')
    parser.add_argument('run_dir', help='Run directory (e.g., run_201)')
    parser.add_argument('--original-tsv',
                        default='data/metabric_complete_merged_data_original.tsv',
                        help='Path to the original METABRIC TSV')
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip('/')
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', run_dir)

    lattice = np.load(os.path.join(run_dir, 'som_lattice.npy'))
    assignments = pd.read_csv(os.path.join(run_dir, 'som_cluster_assignments.csv'))

    original_tsv = args.original_tsv
    if not os.path.isabs(original_tsv):
        original_tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', original_tsv)
    original = pd.read_csv(original_tsv, sep='\t', low_memory=False)
    merged = assignments.merge(original[['Sample_ID', 'Pam50 + Claudin-low subtype']], on='Sample_ID', how='left')

    rows, cols, _ = lattice.shape
    n_types = len(SUBTYPE_ORDER)

    prop_grid = np.zeros((rows, cols, n_types))
    count_grid = np.zeros((rows, cols), dtype=int)

    for (r, c), group in merged.groupby(['SOM_Row', 'SOM_Col']):
        r, c = int(r), int(c)
        vals = group['Pam50 + Claudin-low subtype'].dropna()
        vals = vals[vals != 'NC']
        n = len(vals)
        count_grid[r, c] = n
        if n > 0:
            for k, stype in enumerate(SUBTYPE_ORDER):
                prop_grid[r, c, k] = (vals == stype).sum() / n
    max_d = compute_prototype_distances(lattice)
    max_assignments = np.max(count_grid)
    fig, ax = plt.subplots(figsize=(12, 12))

    for i in range(rows):
        for j in range(cols):
            count = count_grid[i, j]
            intensity = count / max_assignments if max_assignments > 0 else 0
            color = cm.Reds(intensity)
            rect = Rectangle((j, rows - 1 - i), 1, 1, facecolor=color,
                              edgecolor='none', zorder=1)
            ax.add_patch(rect)

    bar_height = 0.35
    bar_width = 0.80
    for i in range(rows):
        for j in range(cols):
            if count_grid[i, j] == 0:
                continue
            y_cell = rows - 1 - i
            x_left = j + (1 - bar_width) / 2
            y_bottom = y_cell + 0.5 - bar_height
            single_bar_w = bar_width / n_types
            for k in range(n_types):
                prop = prop_grid[i, j, k]
                if prop <= 0:
                    continue
                bx = x_left + k * single_bar_w
                bh = prop * bar_height
                rect = Rectangle((bx, y_bottom + (bar_height - bh)),
                                 single_bar_w * 0.85, bh,
                                 facecolor=SUBTYPE_COLORS[SUBTYPE_ORDER[k]],
                                 edgecolor='none', alpha=0.85, zorder=2)
                ax.add_patch(rect)

    for i in range(rows - 1):
        for j in range(cols - 1):
            current = lattice[i, j, :]
            right = lattice[i, j + 1, :]
            d = np.linalg.norm(current - right)
            intensity = d / max_d if max_d > 0 else 0
            color = cm.Greys(intensity)
            y_top = rows - 1 - i
            ax.plot([j + 1, j + 1], [y_top, y_top + 1], color=color,
                    linewidth=3, zorder=3)
            below = lattice[i + 1, j, :]
            d = np.linalg.norm(current - below)
            intensity = d / max_d if max_d > 0 else 0
            color = cm.Greys(intensity)
            ax.plot([j, j + 1], [rows - 1 - i, rows - 1 - i], color=color,
                    linewidth=3, zorder=3)
    for j in range(cols - 1):
        current = lattice[rows - 1, j, :]
        right = lattice[rows - 1, j + 1, :]
        d = np.linalg.norm(current - right)
        intensity = d / max_d if max_d > 0 else 0
        color = cm.Greys(intensity)
        ax.plot([j + 1, j + 1], [0, 1], color=color,
                linewidth=3, zorder=3)
    for i in range(rows - 1):
        current = lattice[i, cols - 1, :]
        below = lattice[i + 1, cols - 1, :]
        d = np.linalg.norm(current - below)
        intensity = d / max_d if max_d > 0 else 0
        color = cm.Greys(intensity)
        ax.plot([cols - 1, cols], [rows - 1 - i, rows - 1 - i], color=color,
                linewidth=3, zorder=3)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title('mU-Matrix with Pam50 + Claudin-low Overlay')
    ax.set_xlabel('SOM Column')
    ax.set_ylabel('SOM Row')

    legend_elements = [Patch(facecolor=SUBTYPE_COLORS[s], label=s) for s in SUBTYPE_ORDER]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              title='Pam50 + Claudin-low', title_fontsize=9,
              framealpha=0.9, edgecolor='gray',
              bbox_to_anchor=(1.02, 1.0))
    out_path = os.path.join(run_dir, 'mU_matrix_pam50_claudinlow_bars.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
