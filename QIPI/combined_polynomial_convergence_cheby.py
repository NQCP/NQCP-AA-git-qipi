"""
Chebyshev matrix-inversion convergence plot for a single molecule.

Left panel : the Chebyshev approximation G_N(x) of 1/x for a few degrees N,
             overlaid on the (renormalized, shifted) Hamiltonian spectrum.
Right panel: overlap with the target eigenvector under repeated application of
             the polynomial-approximated inverse, compared with the exact inverse.

Usage:
    python QIPI/combined_polynomial_convergence_cheby.py
    python -m QIPI.combined_polynomial_convergence_cheby
"""

import os
import sys
import warnings
from decimal import Decimal

import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from QIPI.hamiltonian import build, overlap_square, init_state

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Chebyshev approximation of 1/x  (Childs-Kothari-Somma style coefficients)
# ---------------------------------------------------------------------------

def computeChebCoeffs(nCheb, b0):
    chebCoeffs = np.zeros(nCheb)
    for j in range(0, nCheb):
        sumval = Decimal(0.0)
        for i in range(j + 1, b0 + 1):
            sumval += Decimal(scipy.special.binom((2 * b0), b0 + i))
        sign = 1 if (j % 2 == 0) else -1
        chebCoeffs[j] = sign * sumval / Decimal((2 ** Decimal(2 * b0)))
    return chebCoeffs


def chebyshevApprox(x, n, coeffs):
    sumval = 0
    for j in range(0, n):
        sumval += coeffs[j] * scipy.special.chebyt(2 * j + 1)(x)
    return sumval * 4


def main():
    # ===== Chebyshev approximation parameters =====
    condNum = 6
    eps = 1e-4                                  # Precision
    b = condNum ** 2 * np.log(condNum / eps)    # Parameter b
    b0 = int(np.ceil(b))
    j0 = np.sqrt(b * np.log(4 * b / eps))
    nCheb = int(np.ceil(j0))                    # Number of Chebyshev coefficients
    chebCoeffs = computeChebCoeffs(nCheb, b0)

    # ===== Plot styling =====
    plt.style.use(os.path.join(_REPO_ROOT, 'plotstylefile.mplstyle'))
    plotdir = os.path.join(_REPO_ROOT, 'paper_plots')
    os.makedirs(plotdir, exist_ok=True)

    color1 = '#3929b7'  # blue
    color2 = '#901a1e'  # red
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [color1, color2], N=n_bins)

    # ===== Parameters =====
    molecule = 'H2'
    degrees = [2, 7, 9]                 # Degrees shown in the legend
    cheby_orders = [2, 7, 9]            # Chebyshev orders N to plot
    num_iter = 5                        # Number of convergence iterations
    target_overlap = 0.2
    use_rotation_decomposition = False  # No rotation-decomposition data in this plot

    # ===== Build the Hamiltonian  =====
    ham = build(molecule)
    renorm_shifted = ham['renorm_shifted']
    rSeigenValues = ham['rSeigenValues']
    targetEvec = ham['targetEvec']
    targetIdx = ham['targetIdx']

    condNum = np.abs(np.max(np.abs(rSeigenValues)) / np.min(np.abs(rSeigenValues)))

    U, S, Vh = np.linalg.svd(renorm_shifted, full_matrices=True)
    V = Vh.T

    # Reseed for a reproducible initial guess state.
    np.random.seed(5737)
    initial_state = init_state(target_state=targetEvec, target_overlap=target_overlap)

    # ===== Classical exact-inverse convergence =====
    exactInv = np.linalg.inv(renorm_shifted)
    pMexact = exactInv
    vec_norm = initial_state
    transfVec = vec_norm
    oveOfIter = [overlap_square(transfVec, targetEvec)]
    iterVec = [0]

    for iteration in range(num_iter):
        iterVec.append(iteration + 1)
        transfVecUnnorm = pMexact.dot(transfVec)
        transfVec = transfVecUnnorm / np.linalg.norm(transfVecUnnorm)
        oveOfIter.append(overlap_square(transfVec, targetEvec))

    # ===== Figure =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    # ---------- FIRST SUBPLOT: polynomial plots ----------
    print("Generating polynomial plots...")

    rescaled_eigvals = rSeigenValues
    degree_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    colors = custom_cmap(np.linspace(0, 1, int(len(rSeigenValues) * (1.2))))

    for i in range(0, len(rSeigenValues)):
        color = colors[i]
        label = '$\\tilde{\\lambda}_{' + str(i) + '}~\\mathrm{of}~\\tilde{H}$'
        if i == targetIdx:
            ax1.axvline(rSeigenValues[i], linewidth=1.5, color=color, label=label)
        else:
            ax1.axvline(rSeigenValues[i], linewidth=1, linestyle='-', color=color, alpha=0.5, label=label)

    xvals = np.linspace(-1, 1, 1000)

    for numCheby in cheby_orders:
        gx = chebyshevApprox(xvals, numCheby, chebCoeffs)
        ax1.plot(xvals, gx, label=r'$N=$' + str(numCheby))

    ax1.plot(xvals, 1 / xvals, '--', color="black")
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$G_N(x)$')

    # Build the legend elements for the eigenvalue lines / degrees / exact inverse.
    legend_elements_ax1 = []
    for i in range(0, len(rSeigenValues)):
        color = colors[i]
        label = '$\\tilde{\\lambda}_{' + str(i) + '}~\\mathrm{of}~\\tilde{H}$'
        if i == targetIdx:
            line = plt.Line2D([0], [0], color=color, linewidth=1.5, label=label)
        else:
            line = plt.Line2D([0], [0], color=color, linestyle='-', linewidth=1, label=label)
        legend_elements_ax1.append((line, label))

    for degree_idx, degree in enumerate(degrees):
        rect = mpatches.Rectangle((0, 0), 1, 1, facecolor=degree_colors[degree_idx],
                                  label=f'$N_{{\\mathrm{{max}}}} = {2 * degree + 1}$')
        legend_elements_ax1.append((rect, f'$N_{{\\mathrm{{max}}}} = {2 * degree + 1}$'))

    exact_inverse_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
                                    label='Exact Inverse')
    legend_elements_ax1.append((exact_inverse_line, 'Exact Inverse'))

    # ---------- SECOND SUBPLOT: convergence plots ----------
    print("Generating convergence plots...")

    ax2.plot(iterVec, np.abs(oveOfIter), '--', color='black', linewidth=1.2, label="Exact Inverse")
    for nCheb in cheby_orders:
        pS = chebyshevApprox(S, nCheb, chebCoeffs)
        pM = np.dot(U * pS, Vh)
        transfVec = vec_norm
        oveOfIter = [overlap_square(transfVec, targetEvec)]
        iterVec = [0]

        for iteration in range(num_iter):
            iterVec.append(iteration + 1)
            transfVecUnnorm = pM.dot(transfVec)
            transfVec = transfVecUnnorm / np.linalg.norm(transfVecUnnorm)
            oveOfIter.append(overlap_square(transfVec, targetEvec))
        ax2.plot(iterVec, np.abs(oveOfIter), '-', marker='o', markersize=7,
                 markeredgecolor='black', markeredgewidth=0.8, alpha=0.9,
                 label=r'$N_{\mathrm{max}}=$' + str(2 * nCheb + 1))

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Overlap')
    ax2.set_ylim(-0.01, 1.1)

    # ---------- Shared legends below the figure ----------
    eigenvalue_handles, eigenvalue_labels = [], []
    delta_handles, delta_labels = [], []
    precision_handles, precision_labels = [], []
    degree_handles, degree_labels = [], []
    exact_handles, exact_labels = [], []

    for handle, label in legend_elements_ax1:
        if 'lambda' in label and '\\tilde{H}' in label:
            eigenvalue_handles.append(handle)
            eigenvalue_labels.append(label)
        elif 'Delta' in label or '\\pm' in label:
            delta_handles.append(handle)
            delta_labels.append(label)
        elif 'epsilon' in label:
            precision_handles.append(handle)
            precision_labels.append(label)
        elif '\\mathrm{max}' in label:
            degree_handles.append(handle)
            degree_labels.append(label)
        elif 'Exact Inverse' in label:
            exact_handles.append(handle)
            exact_labels.append(label)

    legends = []

    if eigenvalue_handles:
        leg1 = fig.legend(eigenvalue_handles, eigenvalue_labels, loc='center',
                          bbox_to_anchor=(0.33, -0.15), title='Eigenvalues',
                          frameon=True, fancybox=True, shadow=True)
        legends.append(leg1)

    if delta_handles or exact_handles:
        combined_handles = delta_handles + exact_handles
        combined_labels = delta_labels + exact_labels
        leg2 = fig.legend(combined_handles, combined_labels, loc='center',
                          bbox_to_anchor=(0.51, -0.15),
                          frameon=True, fancybox=True, shadow=True)
        legends.append(leg2)

    if precision_handles and use_rotation_decomposition:
        leg3 = fig.legend(precision_handles, precision_labels, loc='center',
                          bbox_to_anchor=(0.50, -0.15), title='Precision',
                          frameon=True, fancybox=True, shadow=True)
        legends.append(leg3)

    if degree_handles:
        leg4 = fig.legend(degree_handles, degree_labels, loc='center',
                          bbox_to_anchor=(0.69, -0.15), title='Degree',
                          frameon=True, fancybox=True, shadow=True)
        legends.append(leg4)

    for leg in legends:
        fig.add_artist(leg)

    fig.tight_layout(pad=2.0)

    filepath = os.path.join(plotdir, 'figure3.pdf')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved {filepath}")
    plt.show()

    print("Plot generation completed!")


if __name__ == "__main__":
    main()
