import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch

NEG_COLOR = '#F7F7F7'
POS_COLOR = '#D6604D'

DATA_TYPES = ['EXPR', 'MUT', 'CNA']

GENE_LOCATIONS = 'data/gene_locations.txt'

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


def make_cmap():
    cmap = LinearSegmentedColormap.from_list('her2_style', [NEG_COLOR, POS_COLOR], N=256)
    cmap.set_bad(color='black')
    return cmap


def compute_subtype_proportions(merged, lattice_rows, lattice_cols):
    n_types = len(SUBTYPE_ORDER)
    prop_grid = np.zeros((lattice_rows, lattice_cols, n_types))
    count_grid = np.zeros((lattice_rows, lattice_cols), dtype=int)

    col = 'Pam50 + Claudin-low subtype'
    for (r, c), group in merged.groupby(['SOM_Row', 'SOM_Col']):
        r, c = int(r), int(c)
        vals = group[col].dropna()
        vals = vals[vals != 'NC']
        n = len(vals)
        count_grid[r, c] = n
        if n > 0:
            for k, stype in enumerate(SUBTYPE_ORDER):
                prop_grid[r, c, k] = (vals == stype).sum() / n
    return prop_grid, count_grid


def compute_mean_matrix(merged, lattice_rows, lattice_cols, col):
    mat = np.full((lattice_rows, lattice_cols), np.nan)
    cnt = np.zeros((lattice_rows, lattice_cols), dtype=int)
    for (r, c), group in merged.groupby(['SOM_Row', 'SOM_Col']):
        r, c = int(r), int(c)
        vals = group[col].dropna()
        cnt[r, c] = len(vals)
        if len(vals) > 0:
            mat[r, c] = vals.mean()
    return mat, cnt


def plot_heatmap_with_bars(matrix, count_matrix, prop_grid, title, save_path,
                           cbar_label='Value'):
    rows, cols = matrix.shape
    cmap = make_cmap()
    n_types = len(SUBTYPE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor('black')

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    vmin = np.nanmin(matrix) if np.any(~np.isnan(matrix)) else 0
    vmax = np.nanmax(matrix) if np.any(~np.isnan(matrix)) else 1
    if vmin == vmax:
        vmax = vmin + 1

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')

    bar_height = 0.35
    bar_width = 0.80

    for i in range(rows):
        for j in range(cols):
            if count_matrix[i, j] == 0:
                continue
            x_left = j - bar_width / 2
            y_bottom = i + 0.5 - bar_height

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
                                 edgecolor='none', alpha=0.85, zorder=3)
                ax.add_patch(rect)

    ax.set_xlabel('SOM Column')
    ax.set_ylabel('SOM Row')
    ax.set_title(title, fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)

    legend_elements = [Patch(facecolor=SUBTYPE_COLORS[s], label=s) for s in SUBTYPE_ORDER]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              title='Pam50 + Claudin-low', title_fontsize=8,
              framealpha=0.9, edgecolor='gray',
              bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def parse_chromosome_arm(cytoband):
    m = re.match(r'^([0-9XY]+)([pq])', cytoband)
    if m:
        return f"chr{m.group(1)}{m.group(2)}"
    return None


def build_arm_gene_map(gene_class_path):
    df = pd.read_csv(gene_class_path, sep='\t')
    arm_genes = {}
    for _, row in df.iterrows():
        gene = row['Gene']
        loc = str(row['Chromosome_Location'])
        arm = parse_chromosome_arm(loc)
        if arm:
            arm_genes.setdefault(arm, []).append(gene)
    return arm_genes


def main():
    parser = argparse.ArgumentParser(
        description='Plot summed matrices with PAM50/Claudin-low overlay by chromosome arm')
    parser.add_argument('run_dir', help='Path to run directory (e.g. run_201)')
    parser.add_argument('--original-tsv',
                        default='data/metabric_complete_merged_data_original.tsv',
                        help='Path to the original METABRIC TSV')
    parser.add_argument('--gene-locations',
                        default=GENE_LOCATIONS,
                        help='Path to gene locations file')
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip('/')
    assignments_path = os.path.join(run_dir, 'som_cluster_assignments.csv')
    if not os.path.isfile(assignments_path):
        raise FileNotFoundError(f"No assignments file at {assignments_path}")

    loc_dir = os.path.join(run_dir, 'location_analysis')
    if not os.path.isdir(loc_dir):
        raise FileNotFoundError(f"No location_analysis directory at {loc_dir}")

    assignments = pd.read_csv(assignments_path)
    original = pd.read_csv(args.original_tsv, sep='\t', low_memory=False)
    merged = assignments.merge(original, on='Sample_ID', how='left')

    lattice_rows = int(merged['SOM_Row'].max()) + 1
    lattice_cols = int(merged['SOM_Col'].max()) + 1

    prop_grid, subtype_count = compute_subtype_proportions(merged, lattice_rows, lattice_cols)

    arm_genes = build_arm_gene_map(args.gene_locations)

    arm_dirs = sorted([d for d in os.listdir(loc_dir) if os.path.isdir(os.path.join(loc_dir, d))])

    available_cols = set(original.columns)

    total_plots = 0
    for arm_dir in arm_dirs:
        arm_path = os.path.join(loc_dir, arm_dir)
        genes = arm_genes.get(arm_dir, [])
        if not genes:
            continue

        for dtype in DATA_TYPES:
            cols = [f"{dtype}_{g}" for g in genes if f"{dtype}_{g}" in available_cols]
            if not cols:
                continue

            summed = np.zeros((lattice_rows, lattice_cols))
            count_mat = np.zeros((lattice_rows, lattice_cols), dtype=int)
            any_data = np.zeros((lattice_rows, lattice_cols), dtype=bool)

            for col in cols:
                mat, cnt = compute_mean_matrix(merged, lattice_rows, lattice_cols, col)
                valid = ~np.isnan(mat)
                any_data |= valid
                summed += np.where(valid, mat, 0)
                count_mat = np.maximum(count_mat, cnt)

            summed[~any_data] = np.nan

            gene_names = [c.replace(f"{dtype}_", "") for c in cols]
            gene_count = len(cols)
            title = f"{arm_dir} {dtype} Summed ({gene_count} genes)\n{', '.join(gene_names)}"

            if dtype == 'MUT':
                cbar_label = 'Summed Mutation Proportion'
            elif dtype == 'CNA':
                cbar_label = 'Summed Mean CNA'
            else:
                cbar_label = 'Summed Mean Expression'

            save_path = os.path.join(arm_path, f"{dtype}_summed.png")
            plot_heatmap_with_bars(summed, subtype_count, prop_grid, title, save_path, cbar_label=cbar_label)
            total_plots += 1


if __name__ == '__main__':
    main()
