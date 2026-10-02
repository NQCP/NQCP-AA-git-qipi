"""
Combined queries-to-accuracy plot for H2, LiH and BeH2 in one figure
(3 columns, shared y-axis, single shared colorbar).

The per-molecule query data is cached in QIPI/data/queries_<molecule>.pkl so
re-running for styling tweaks is instant. Pass --regen to recompute.
"""

import os
import sys
import pickle

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import chebyt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from QIPI.hamiltonian import build as build_hamiltonian

plt.style.use(os.path.join(_REPO_ROOT, 'plotstylefile.mplstyle'))
plotdir = os.path.join(_REPO_ROOT, 'paper_plots')
DATA_DIR = os.path.join(_THIS_DIR, 'data')

chem_acc = 1.59e-3
numlines = 201
numiter = 30
deg = 8


def optpoly_eigenstatefiltering(x, l, Delta):
    arg_top = -1 + 2 * ((x ** 2 - Delta ** 2) / (1 - Delta ** 2))
    arg_bot = -1 + 2 * ((-Delta ** 2) / (1 - Delta ** 2))
    cheb_l = scipy.special.chebyt(l)
    return cheb_l(arg_top) / cheb_l(arg_bot)


vpoly = np.vectorize(optpoly_eigenstatefiltering)


def generate_random_state_with_overlap(target_state, target_overlap):
    """Identical to the helper in hamiltonian.py."""
    if not 0 <= target_overlap <= 1:
        raise ValueError("Target overlap must be between 0 and 1")
    if abs(np.linalg.norm(target_state) - 1) > 1e-10:
        raise ValueError("Target state must be normalized")
    random_state = np.random.randn(len(target_state)) + 1j * np.random.randn(len(target_state))
    random_state -= np.vdot(target_state, random_state) * target_state
    random_state /= np.linalg.norm(random_state)
    guess_state = np.sqrt(target_overlap) * target_state + np.sqrt(1 - target_overlap) * random_state
    return guess_state


def compute_queries(molecule):
    ham = build_hamiltonian(molecule)
    renorm_shifted = ham['renorm_shifted']
    targetEvec = ham['targetEvec']
    targetEval = ham['targetEval']
    scale = ham['scale']
    lam = ham['lam']
    delta_scaled = ham['delta_scaled']

    U, S, Vh = np.linalg.svd(renorm_shifted, full_matrices=True)
    pS = vpoly(S, deg, delta_scaled)
    pM = np.dot(U * pS, Vh)

    overlaps = np.linspace(0, 1, numlines + 1)
    queries_to_accuracy = np.zeros(numlines + 1)
    found = np.zeros(numlines + 1, dtype=bool)

    for i in range(len(overlaps)):
        transfVec = generate_random_state_with_overlap(targetEvec, overlaps[i])
        found_accuracy = False

        for iteration in range(numiter):
            energy = (np.dot(transfVec.T, renorm_shifted.dot(transfVec))) * (scale) + lam

            if abs(energy - targetEval) <= chem_acc and not found_accuracy:
                queries_to_accuracy[i] = iteration * deg * 2
                found_accuracy = True
                break

            transfVecUnnorm = pM.dot(transfVec)
            transfVec = transfVecUnnorm / np.linalg.norm(transfVecUnnorm)

        if not found_accuracy:
            queries_to_accuracy[i] = numiter * deg * 2
        found[i] = found_accuracy

    w = np.linalg.eigvalsh(renorm_shifted)
    p_vals = vpoly(w, deg, delta_scaled)
    E_orig = w * scale + lam
    lam_t_rs = (targetEval - lam) / scale
    ti = int(np.argmin(np.abs(w - lam_t_rs)))
    E_t = E_orig[ti]
    N = len(w)

    theory_overlaps = np.linspace(0.005, 1.0, 500)
    theory_queries = np.empty_like(theory_overlaps)
    for n, c2 in enumerate(theory_overlaps):
        base = np.full(N, (1 - c2) / (N - 1))
        base[ti] = c2
        k_found = numiter
        for k in range(numiter + 1):
            wk = base * (p_vals ** (2 * k))
            Ek = np.sum(wk * E_orig) / np.sum(wk)
            if abs(Ek - E_t) <= chem_acc:
                k_found = k
                break
        theory_queries[n] = k_found * deg * 2

    return {
        'molecule': molecule,
        'overlaps': overlaps,
        'queries': queries_to_accuracy,
        'found': found,
        'theory_overlaps': theory_overlaps,
        'theory_queries': theory_queries,
        'scale': scale,
        'delta_scaled': delta_scaled,
    }


def get_data(molecule, regen=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f'queries_{molecule}.pkl')
    if os.path.exists(path) and not regen:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if 'theory_queries' in data and 'scale' in data:
            return data
        print(f"{molecule}: cached data lacks theory/scaling info, recomputing ...")
    print(f"Computing queries-to-accuracy data for {molecule} ...")
    data = compute_queries(molecule)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {path}")
    return data


MOLECULE_ORDER = ['H2', 'LiH', 'BeH2']
MOLECULE_LABELS = {
    'H2': r'$\mathrm{H}_2$',
    'LiH': r'$\mathrm{LiH}$',
    'BeH2': r'$\mathrm{BeH}_2$',
}
XLABEL = r'$ \gamma^2 = |\langle \psi_{t} | \Psi_{0} \rangle|^2$'
YLABEL = "Queries to " + r'$U_H$'


def make_figure(data, layout, out_path, ylabel=YLABEL):
    cmap = matplotlib.colormaps['magma_r']

    if layout == 'cols':
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=100, sharey=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(9, 18), dpi=100, sharex=True)

    for idx, molecule in enumerate(MOLECULE_ORDER):
        ax = axes[idx]
        d = data[molecule]
        mask = d['found']
        ax.scatter(
            d['overlaps'][mask],
            d['queries'][mask],
            c=d['overlaps'][mask],
            cmap=cmap,
            vmin=0,
            vmax=1,
            marker='o',
        )
        ax.set_title(MOLECULE_LABELS[molecule])
        ax.grid(True)
        ax.set_box_aspect(1)

    if layout == 'cols':
        for ax in axes:
            ax.set_xlabel(XLABEL)
        axes[0].set_ylabel(ylabel)
    else:
        for ax in axes:
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
        axes[-1].set_xlabel(XLABEL)

    mappable = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)
    cbar_label = 'Initial overlap: ' + r'$\gamma^2$'

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def bbox_frac(ax):
        bb = ax.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return x0, y0, x1, y1

    right = max(bbox_frac(ax)[2] for ax in axes)
    ref_ax = axes[1] if layout == 'rows' else axes[0]
    _, ry0, _, ry1 = bbox_frac(ref_ax)

    fig_w = fig.get_size_inches()[0]
    cbar_width = 0.3 / fig_w
    cbar_pad = 0.4 / fig_w
    cax = fig.add_axes([right + cbar_pad, ry0, cbar_width, ry1 - ry0])
    fig.colorbar(mappable, cax=cax, label=cbar_label)

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


def main():
    regen = '--regen' in sys.argv[1:]
    data = {m: get_data(m, regen=regen) for m in MOLECULE_ORDER}

    make_figure(
        data,
        layout='cols',
        out_path=os.path.join(plotdir, 'figure12.pdf'),
        ylabel="Queries to " + r'$U_{\tilde{H}}$',
    )
    plt.show()


if __name__ == "__main__":
    main()
