import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.patches import Rectangle
from matplotlib import cm
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_metabric_som import load_metabric_data, normalize_data


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


def plot_mu_matrix_with_means(run_dir, data, all_sample_ids, feature_names,
                              figsize=(12, 12), border_width=3, dpi=300):
    lattice = np.load(os.path.join(run_dir, 'som_lattice.npy'))
    assignments = pd.read_csv(os.path.join(run_dir, 'som_cluster_assignments.csv'))
    rows, cols, n_features = lattice.shape
    count_grid = np.zeros((rows, cols), dtype=int)
    for (r, c), group in assignments.groupby(['SOM_Row', 'SOM_Col']):
        count_grid[int(r), int(c)] = len(group)
    max_assignments = np.max(count_grid)
    max_d = compute_prototype_distances(lattice)
    fig, ax = plt.subplots(figsize=figsize)
    margin = 0.05
    for i in range(rows):
        for j in range(cols):
            count = count_grid[i, j]
            intensity = count / max_assignments if max_assignments > 0 else 0
            color = cm.Reds(intensity)
            rect = Rectangle((j, rows - 1 - i), 1, 1, facecolor=color, edgecolor='none', zorder=1)
            ax.add_patch(rect)
    feat_x_local = np.linspace(margin, 1 - margin, n_features)

    for i in range(rows):
        for j in range(cols):
            vals = lattice[i, j, :]
            vals = np.clip(vals, 0, 1)
            x0 = j
            y0 = rows - 1 - i
            line_x = x0 + feat_x_local
            line_y = y0 + margin + vals * (1 - 2 * margin)

            ax.plot(line_x, line_y, color='black', linewidth=0.5, zorder=2, solid_capstyle='round')

    for i in range(rows - 1):
        for j in range(cols - 1):
            current = lattice[i, j, :]
            right = lattice[i, j + 1, :]
            d = np.linalg.norm(current - right)
            intensity = d / max_d if max_d > 0 else 0
            color = cm.Greys(intensity)
            y_top = rows - 1 - i
            ax.plot([j + 1, j + 1], [y_top, y_top + 1], color=color, linewidth=border_width, zorder=3)
            below = lattice[i + 1, j, :]
            d = np.linalg.norm(current - below)
            intensity = d / max_d if max_d > 0 else 0
            color = cm.Greys(intensity)
            ax.plot([j, j + 1], [rows - 1 - i, rows - 1 - i], color=color, linewidth=border_width, zorder=3)

    for j in range(cols - 1):
        current = lattice[rows - 1, j, :]
        right = lattice[rows - 1, j + 1, :]
        d = np.linalg.norm(current - right)
        intensity = d / max_d if max_d > 0 else 0
        color = cm.Greys(intensity)
        ax.plot([j + 1, j + 1], [0, 1], color=color, linewidth=border_width, zorder=3)

    for i in range(rows - 1):
        current = lattice[i, cols - 1, :]
        below = lattice[i + 1, cols - 1, :]
        d = np.linalg.norm(current - below)
        intensity = d / max_d if max_d > 0 else 0
        color = cm.Greys(intensity)
        ax.plot([cols - 1, cols], [rows - 1 - i, rows - 1 - i], color=color, linewidth=border_width, zorder=3)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title('mU-matrix with Weight Vectors')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    out_path = os.path.join(run_dir, 'mU_matrix_with_weights.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot mU-matrix with mean feature line profiles per prototype')
    parser.add_argument('run_dir', help='Run directory (e.g., run_195)')
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_dir = os.path.join(project_root, run_dir)

    if not os.path.isdir(run_dir):
        sys.exit(1)

    data, all_sample_ids, feature_names = load_metabric_data()
    data, _ = normalize_data(data)

    plot_mu_matrix_with_means(run_dir, data, all_sample_ids, feature_names)
