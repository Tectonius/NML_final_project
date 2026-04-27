import numpy as np
import pandas as pd
from SOM import SOM, width_decay, linear_decay, piecewise_linear_decay, gaussian_neighborhood
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


def load_metabric_data(filepath='data/metabric_complete_merged_data_input.tsv'):
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    if 'Sample_ID' in df.columns:
        sample_ids = df['Sample_ID'].values
        df = df.drop(columns=['Sample_ID'])
    else:
        sample_ids = df.index.values
    data = df.values
    feat_names = df.columns.values
    return data, sample_ids, feat_names


def normalize_data(data):
    dmin = data.min(axis=0)
    dmax = data.max(axis=0)
    drange = dmax - dmin
    drange[drange == 0] = 1.0
    normed = (data - dmin) / drange
    return normed


def generate_snapshot_schedule(training_steps, n_snapshots=50):
    steps = np.unique(np.geomspace(1000, training_steps, n_snapshots).astype(int))
    steps = np.unique(np.concatenate([[0], steps, [training_steps - 1]]))
    return set(steps.tolist())


def make_snapshot_callback(snapshot_dir, data):
    os.makedirs(snapshot_dir, exist_ok=True)

    def callback(som, step, data):
        path = os.path.join(snapshot_dir, f'mU_matrix_step_{step:07d}.png')
        som.plot_mU_matrix(data, border_width=3, figsize=(12, 12),
                          title=f'mU-Matrix at step {step:,}',
                          save_path=path)
    return callback


def train_metabric_som(data, lattice_shape=(30, 30), training_steps=200000,
                       max_lr=0.1, min_lr=0.01, initial_width=15,
                       verbose=True, snapshot_interval=None, lr_schedule=None,
                       snapshot_dir=None, n_snapshots=50):
    som = SOM(lattice_shape=lattice_shape)
    som.make_lattice(data)

    w_func = width_decay(w_0=initial_width, t_max=training_steps)
    if lr_schedule:
        lr_func = piecewise_linear_decay(lr_schedule, max_t=training_steps)
    else:
        lr_func = linear_decay(max_lr=max_lr, min_lr=min_lr, max_t=training_steps)

    mon_steps = None
    if snapshot_interval:
        mon_steps = list(range(0, training_steps, snapshot_interval))

    snap_cb = None
    snap_steps = None
    if snapshot_dir:
        snap_steps = generate_snapshot_schedule(training_steps, n_snapshots)
        snap_cb = make_snapshot_callback(snapshot_dir, data)
    som.train(
        data=data,
        training_steps=training_steps,
        lr_func=lr_func,
        neigh_func=gaussian_neighborhood,
        width_func=w_func,
        monitor_steps=mon_steps,
        verbose=verbose,
        snapshot_callback=snap_cb,
        snapshot_steps=snap_steps
    )

    som.compute_quantization_error(data)
    return som


def analyze_som_clusters(som, data, sample_ids):
    assignments = som.get_cluster_assignments(data)
    rows, cols = som.lattice_shape
    pe_counts = np.zeros((rows, cols))
    for i, j in assignments:
        pe_counts[i, j] += 1

    df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'SOM_Row': assignments[:, 0],
        'SOM_Col': assignments[:, 1],
        'SOM_Cluster': [f"{i}_{j}" for i, j in assignments]
    })
    return df, pe_counts


def plot_pe_density(pe_counts, figsize=(10, 8), save_path='pe_density.png'):
    plt.figure(figsize=figsize)
    im = plt.imshow(pe_counts, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Number of Samples')
    plt.title('Sample Distribution Across SOM Lattice')
    plt.xlabel('Column')
    plt.ylabel('Row')
    rows, cols = pe_counts.shape
    for i in range(rows):
        for j in range(cols):
            n = int(pe_counts[i, j])
            if n > 0:
                color = 'white' if pe_counts[i, j] > pe_counts.max() * 0.6 else 'black'
                plt.text(j, i, str(n), ha='center', va='center',
                        color=color, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def render_mu_matrix_snapshots(som, data, output_dir):
    if not som.training_history:
        return

    snap_dir = os.path.join(output_dir, 'mu_matrix_snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    orig_lattice = som.lattice.copy()

    for entry in som.training_history:
        step = entry['step']
        som.lattice = entry['lattice']
        path = os.path.join(snap_dir, f'mU_matrix_step_{step:06d}.png')
        som.plot_mU_matrix(data, border_width=3, figsize=(12, 12),title=f'mU-Matrix at step {step:,}',save_path=path)

    som.lattice = orig_lattice

    path = os.path.join(snap_dir, f'mU_matrix_step_{200000:06d}.png')
    som.plot_mU_matrix(data, border_width=3, figsize=(12, 12),
                      title='mU-Matrix at step 200,000 (final)',
                      save_path=path)

def main():
    parser = argparse.ArgumentParser(description='Train SOM on METABRIC breast cancer data')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    parser.add_argument('--lattice-rows', type=int, default=30,
                        help='Number of rows in SOM lattice (default: 30)')
    parser.add_argument('--lattice-cols', type=int, default=30,
                        help='Number of columns in SOM lattice (default: 30)')
    parser.add_argument('--training-steps', type=int, default=200000,
                        help='Number of training steps (default: 200000)')
    parser.add_argument('--max-lr', type=float, default=0.1,
                        help='Maximum learning rate (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=0.01,
                        help='Minimum learning rate (default: 0.01)')
    parser.add_argument('--initial-width', type=float, default=15,
                        help='Initial neighborhood width (default: 15)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--snapshot-interval', type=int, default=None,
                        help='Save mU-matrix snapshot every N training steps to memory (default: disabled)')
    parser.add_argument('--snapshot-dir', type=str, default=None,
                        help='Directory to save periodic mU-matrix PNGs during training (default: disabled)')
    parser.add_argument('--n-snapshots', type=int, default=50,
                        help='Number of geometrically-spaced snapshots when using --snapshot-dir (default: 50)')
    parser.add_argument('--lr-schedule', type=str, default=None,
                        help='Piecewise linear LR schedule as frac:lr,frac:lr,... '
                             'e.g. "0:0.8,0.1:0.3,0.5:0.05,1.0:0.01". '
                             'Overrides --max-lr/--min-lr when provided.')

    args = parser.parse_args()

    lr_sched = None
    if args.lr_schedule:
        lr_sched = []
        for pair in args.lr_schedule.split(','):
            frac_str, lr_str = pair.split(':')
            lr_sched.append((float(frac_str), float(lr_str)))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    data, sample_ids, feat_names = load_metabric_data('data/metabric_complete_merged_data_input.tsv')
    normed = normalize_data(data)

    snap_dir = args.snapshot_dir
    if snap_dir and not os.path.isabs(snap_dir):
        snap_dir = os.path.join(args.output_dir, snap_dir)

    som = train_metabric_som(
        data=normed,
        lattice_shape=(args.lattice_rows, args.lattice_cols),
        training_steps=args.training_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        initial_width=args.initial_width,
        verbose=True,
        snapshot_interval=args.snapshot_interval,
        lr_schedule=lr_sched,
        snapshot_dir=snap_dir,
        n_snapshots=args.n_snapshots
    )

    df_assign, pe_counts = analyze_som_clusters(som, normed, sample_ids)

    assign_path = os.path.join(args.output_dir, 'som_cluster_assignments.csv')
    df_assign.to_csv(assign_path, index=False)

    pe_path = os.path.join(args.output_dir, 'pe_density.png')
    plot_pe_density(pe_counts, save_path=pe_path)

    mu_path = os.path.join(args.output_dir, 'mU_matrix.png')
    som.plot_mU_matrix(normed, border_width=3, figsize=(12, 12),
                      title='Modified U-Matrix: METABRIC SOM',
                      save_path=mu_path)

    if args.snapshot_interval:
        render_mu_matrix_snapshots(som, normed, args.output_dir)

    lattice_path = os.path.join(args.output_dir, 'som_lattice.npy')
    np.save(lattice_path, som.lattice)

    params_path = os.path.join(args.output_dir, 'training_params.txt')
    with open(params_path, 'w') as f:
        f.write(f"Lattice shape: {args.lattice_rows} x {args.lattice_cols}\n")
        f.write(f"Training steps: {args.training_steps}\n")
        if args.lr_schedule:
            f.write(f"Learning rate schedule: {args.lr_schedule}\n")
        else:
            f.write(f"Learning rate: {args.max_lr} -> {args.min_lr} (linear decay)\n")
        f.write(f"Initial neighborhood width: {args.initial_width}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Data samples: {len(sample_ids)}\n")
        f.write(f"Features: {len(feat_names)}\n")

if __name__ == '__main__':
    main()
