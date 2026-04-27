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

SKIP_COLUMNS = {'Sample_ID', 'Cohort'}

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
    'Basal':       '#d73027',
    'claudin-low': '#1b9e77',
    'Normal':      '#999999',
}

BINARY_TEXT = {
    'Type of Breast Surgery': 'MASTECTOMY',
    'Chemotherapy': 'YES',
    'ER status measured by IHC': 'Positve',
    'ER Status': 'Positive',
    'HER2 Status': 'Positive',
    'Hormone Therapy': 'YES',
    'PR Status': 'Positive',
    'Radio Therapy': 'YES',
    'Primary Tumor Laterality': 'Left',
}
BINARY_NUMERIC = {
    'Cellularity_High', 'Cellularity_Moderate', 'Cellularity_Low',
    'Inferred Menopausal State',
}
PARSED_BINARY = {
    'Overall Survival Status': lambda s: 1 if str(s).startswith('1:') else (0 if str(s).startswith('0:') else np.nan),
    'Relapse Free Status': lambda s: 1 if str(s).startswith('1:') else (0 if str(s).startswith('0:') else np.nan),
}
MULTI_LEVEL = {
    'Cancer Type Detailed',
    'Pam50 + Claudin-low subtype',
    'Tumor Other Histologic Subtype',
    'Integrative Cluster',
    'Oncotree Code',
    '3-Gene classifier subtype',
    "Patient's Vital Status",
}


def safe_filename(s):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s).strip('_')


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
        # Exclude 'NC' (not classified)
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


def compute_proportion_matrix(merged, lattice_rows, lattice_cols, col, pos_value):
    mat = np.full((lattice_rows, lattice_cols), np.nan)
    cnt = np.zeros((lattice_rows, lattice_cols), dtype=int)
    for (r, c), group in merged.groupby(['SOM_Row', 'SOM_Col']):
        r, c = int(r), int(c)
        vals = group[col].dropna()
        cnt[r, c] = len(vals)
        if len(vals) > 0:
            mat[r, c] = (vals == pos_value).sum() / len(vals)
    return mat, cnt


def plot_heatmap_with_bars(matrix, count_matrix, prop_grid, title, save_path, cbar_label='Value'):
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

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')

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


def process_column(col, merged, lattice_rows, lattice_cols, prop_grid, output_dir):
    count = 0

    if col in BINARY_TEXT:
        pos_val = BINARY_TEXT[col]
        mat, cnt = compute_proportion_matrix(merged, lattice_rows, lattice_cols, col, pos_val)
        fname = f"{safe_filename(col)}.png"
        title = f"{col}: Proportion {pos_val}"
        plot_heatmap_with_bars(mat, cnt, prop_grid, title, os.path.join(output_dir, fname), 'Proportion')
        count = 1

    elif col in BINARY_NUMERIC:
        mat, cnt = compute_mean_matrix(merged, lattice_rows, lattice_cols, col)
        fname = f"{safe_filename(col)}.png"
        title = f"{col}: Proportion = 1"
        plot_heatmap_with_bars(mat, cnt, prop_grid, title, os.path.join(output_dir, fname), 'Proportion')
        count = 1

    elif col in PARSED_BINARY:
        parse_fn = PARSED_BINARY[col]
        merged[f'_parsed_{col}'] = merged[col].apply(parse_fn)
        mat, cnt = compute_mean_matrix(merged, lattice_rows, lattice_cols, f'_parsed_{col}')
        merged.drop(columns=[f'_parsed_{col}'], inplace=True)
        fname = f"{safe_filename(col)}.png"
        if 'Survival' in col:
            title = f"{col}: Proportion Deceased"
        else:
            title = f"{col}: Proportion Recurred"
        plot_heatmap_with_bars(mat, cnt, prop_grid, title, os.path.join(output_dir, fname), 'Proportion')
        count = 1

    elif col in MULTI_LEVEL:
        unique_vals = sorted(merged[col].dropna().unique())
        for val in unique_vals:
            mat, cnt = compute_proportion_matrix(merged, lattice_rows, lattice_cols, col, val)
            fname = f"{safe_filename(col)}_{safe_filename(str(val))}.png"
            title = f"{col}: Proportion {val}"
            plot_heatmap_with_bars(mat, cnt, prop_grid, title, os.path.join(output_dir, fname), 'Proportion')
            count += 1

    else:
        if not np.issubdtype(merged[col].dtype, np.number):
            coerced = pd.to_numeric(merged[col], errors='coerce')
            if coerced.notna().sum() == 0:
                return 0
            merged[col] = coerced

        mat, cnt = compute_mean_matrix(merged, lattice_rows, lattice_cols, col)
        fname = f"{safe_filename(col)}.png"
        title = f"{col}: Mean per Prototype"
        plot_heatmap_with_bars(mat, cnt, prop_grid, title, os.path.join(output_dir, fname), 'Mean')
        count = 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description='Plot response graphs with Pam50 + Claudin-low bar chart overlay')
    parser.add_argument('run_dir', help='Path to run directory (e.g. run_195)')
    parser.add_argument('--original-tsv', default='data/metabric_complete_merged_data_original.tsv', help='Path to the original METABRIC TSV')
    parser.add_argument('--single', default=None, help='Only plot this one column (for testing)')
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip('/')
    assignments_path = os.path.join(run_dir, 'som_cluster_assignments.csv')
    if not os.path.isfile(assignments_path):
        raise FileNotFoundError(f"No assignments file at {assignments_path}")

    output_dir = os.path.join(run_dir, 'pam50_overlay_response_graphs')
    os.makedirs(output_dir, exist_ok=True)

    assignments = pd.read_csv(assignments_path)
    original = pd.read_csv(args.original_tsv, sep='\t', low_memory=False)
    merged = assignments.merge(original, on='Sample_ID', how='left')

    lattice_rows = int(merged['SOM_Row'].max()) + 1
    lattice_cols = int(merged['SOM_Col'].max()) + 1

    prop_grid, _ = compute_subtype_proportions(merged, lattice_rows, lattice_cols)

    if args.single:
        col = args.single
        if col not in merged.columns:
            return
        process_column(col, merged, lattice_rows, lattice_cols, prop_grid, output_dir)
    else:
        columns = [c for c in original.columns if c not in SKIP_COLUMNS]
        for _, col in enumerate(columns, 1):
            process_column(col, merged, lattice_rows, lattice_cols, prop_grid, output_dir)


if __name__ == '__main__':
    main()
