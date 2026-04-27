import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd


class SOM:
    def __init__(self, lattice_shape, input_dim=None):
        self.lattice_shape = lattice_shape
        self.input_dim = input_dim
        self.lattice = None
        self.training_history = []

    def make_lattice(self, data):
        self.input_dim = data.shape[1]
        lattice = np.zeros((*self.lattice_shape, self.input_dim))
        for i in range(self.input_dim):
            lo, hi = np.min(data[:, i]), np.max(data[:, i])
            lattice[:, :, i] = np.random.uniform(lo, hi, self.lattice_shape)
        self.lattice = lattice
        return lattice

    def feedforward(self, x):
        rows, cols, feat = self.lattice.shape
        flat = self.lattice.reshape(rows * cols, feat)
        dists = np.linalg.norm(flat - x, axis=1)
        return np.unravel_index(np.argmin(dists), (rows, cols))

    def update_weights(self, x, winner, t, lr_func, neigh_func, width_func):
        new_lattice = np.zeros_like(self.lattice)
        lr = lr_func(t)
        rows, cols, _ = self.lattice.shape

        for i in range(rows):
            for j in range(cols):
                proto = self.lattice[i, j, :]
                h = neigh_func(np.array([i, j]), np.array(winner), t, width_func)
                new_lattice[i, j, :] = proto + (lr * h * (x - proto))

        self.lattice = new_lattice
        return new_lattice

    def train(self, data, training_steps, lr_func, neigh_func, width_func,
              monitor_steps=None, labels=None, verbose=True,
              snapshot_callback=None, snapshot_steps=None):
        if self.lattice is None:
            self.make_lattice(data)

        n = data.shape[0]
        for t in range(training_steps):
            idx = np.random.randint(0, n)
            x = data[idx, :]
            winner = self.feedforward(x)
            self.update_weights(x, winner, t, lr_func, neigh_func, width_func)

            if monitor_steps and t in monitor_steps:
                self.training_history.append({
                    'step': t,
                    'lattice': self.lattice.copy()
                })

            if snapshot_callback and snapshot_steps and t in snapshot_steps:
                snapshot_callback(self, t, data)
        return self.lattice

    def get_cluster_assignments(self, data):
        res = np.zeros((data.shape[0], 2), dtype=int)
        for k in range(data.shape[0]):
            res[k] = self.feedforward(data[k, :])
        return res

    def compute_quantization_error(self, data):
        total = 0.0
        for k in range(data.shape[0]):
            w = self.feedforward(data[k, :])
            total += np.linalg.norm(data[k, :] - self.lattice[w[0], w[1], :])
        return total / data.shape[0]

    def plot_mU_matrix(self, data, labels=None, border_width=3, figsize=(10, 10), title="Modified U-Matrix", save_path=None):
        rows, cols, feat = self.lattice.shape

        counts = np.zeros((rows, cols))
        for k in range(data.shape[0]):
            w = self.feedforward(data[k, :])
            counts[w[0], w[1]] += 1
        max_count = np.max(counts)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(rows):
            for j in range(cols):
                intensity = counts[i, j] / max_count if max_count > 0 else 0
                rect = Rectangle((j, rows - 1 - i), 1, 1,
                                 facecolor=cm.Reds(intensity), edgecolor='none')
                ax.add_patch(rect)

        max_d = 0
        for i in range(rows):
            for j in range(cols):
                cur = self.lattice[i, j, :]
                if i < rows - 1:
                    d = np.linalg.norm(cur - self.lattice[i + 1, j, :])
                    if d > max_d: max_d = d
                if j < cols - 1:
                    d = np.linalg.norm(cur - self.lattice[i, j + 1, :])
                    if d > max_d: max_d = d

        for i in range(rows - 1):
            for j in range(cols - 1):
                cur = self.lattice[i, j, :]

                d = np.linalg.norm(cur - self.lattice[i, j + 1, :])
                c = cm.Greys(d / max_d if max_d > 0 else 0)
                y_top = rows - 1 - i
                ax.plot([j + 1, j + 1], [y_top, y_top + 1], color=c, linewidth=border_width)

                d = np.linalg.norm(cur - self.lattice[i + 1, j, :])
                c = cm.Greys(d / max_d if max_d > 0 else 0)
                ax.plot([j, j + 1], [rows - 1 - i, rows - 1 - i], color=c, linewidth=border_width)

        for j in range(cols - 1):
            d = np.linalg.norm(self.lattice[rows - 1, j, :] - self.lattice[rows - 1, j + 1, :])
            c = cm.Greys(d / max_d if max_d > 0 else 0)
            ax.plot([j + 1, j + 1], [0, 1], color=c, linewidth=border_width)

        for i in range(rows - 1):
            d = np.linalg.norm(self.lattice[i, cols - 1, :] - self.lattice[i + 1, cols - 1, :])
            c = cm.Greys(d / max_d if max_d > 0 else 0)
            ax.plot([cols - 1, cols], [rows - 1 - i, rows - 1 - i], color=c, linewidth=border_width)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('SOM Column')
        ax.set_ylabel('SOM Row')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_component_planes(self, feature_names=None, figsize=(15, 10), save_path=None):
        _, _, n_feat = self.lattice.shape
        nc = min(5, n_feat)
        nr = int(np.ceil(n_feat / nc))

        _, axes = plt.subplots(nr, nc, figsize=figsize)
        axes = np.array(axes).flatten()

        for f in range(n_feat):
            ax = axes[f]
            im = ax.imshow(self.lattice[:, :, f], cmap='viridis', aspect='auto')
            label = feature_names[f] if feature_names and f < len(feature_names) else f'Feature {f}'
            ax.set_title(label, fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx in range(n_feat, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Component Planes', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def width_decay(w_0, t_max):
    lam = t_max / np.log(w_0)
    def f(t):
        return w_0 * np.exp(-t / lam)
    return f


def linear_decay(max_lr, min_lr, max_t):
    def f(t):
        return max_lr - (max_lr - min_lr) * (t / max_t)
    return f


def piecewise_linear_decay(breakpoints, max_t):
    breakpoints = sorted(breakpoints, key=lambda x: x[0])
    fracs = [b[0] for b in breakpoints]
    lrs = [b[1] for b in breakpoints]

    def f(t):
        frac = t / max_t
        if frac <= fracs[0]:
            return lrs[0]
        if frac >= fracs[-1]:
            return lrs[-1]
        for i in range(len(fracs) - 1):
            if fracs[i] <= frac <= fracs[i + 1]:
                span = fracs[i + 1] - fracs[i]
                alpha = (frac - fracs[i]) / span if span > 0 else 1.0
                return lrs[i] + alpha * (lrs[i + 1] - lrs[i])
        return lrs[-1]
    return f

def exp_decay(initial_lr, k):
    def f(t):
        return initial_lr * np.exp(k * t)
    return f

def gaussian_neighborhood(point, winner, t, width_func):
    d = np.linalg.norm(point - winner)
    w = width_func(t)
    return np.exp(-(d**2) / (2 * w**2))

def map_to_range(x, min_0, max_0, min_1, max_1):
    return min_1 + (x - min_0) * ((max_1 - min_1) / (max_0 - min_0))
