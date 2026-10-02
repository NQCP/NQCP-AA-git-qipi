import os
import sys
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from QIPI.chebyshev_qsp_phase_factors import get_phase_factors
from QIPI.hamiltonian import build as build_hamiltonian
from QIPI.qsvt import *

plt.style.use(os.path.join(_REPO_ROOT, 'plotstylefile.mplstyle'))

color1 = '#3929b7'
color2 = '#901a1e'

n_bins = 256
colors = [color1, color2]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

np.random.seed(5737)
warnings.filterwarnings('ignore')

degree = 8
M = degree // 2
num_data_qubits = 2

ham = build_hamiltonian("H2")
delta_scaled = ham["delta_scaled"]
rescaled_eigvals = ham["rSeigenValues"]
targetIdx = ham["targetIdx"]

A = ham["renorm_shifted"]

x = np.linspace(-1, 1, 500)

U_list = []
for i in range(len(x)):
    U = block_encode(np.array([[x[i], 0], [0, x[i]]]))
    U_gate = UnitaryGate(U)
    U_list.append(U_gate)

print(f"Processing polynomial: degree {degree}")
phi_list = get_phase_factors(M=M, a=delta_scaled, plot=False)

first_element = []
for i in range(len(x)):
    current_qc = qsvt_circuit(U_gate=U_list[i], phi_list=phi_list, num_data_qubits=2)
    current_qc.h(2)
    current_qc.h(1)
    unitary = Operator(current_qc.to_gate())
    matrix = unitary.data[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    first_element.append(np.real(eigenvalues[0]))

polynomial_values = np.array(first_element)

poly_at_eigvals = []
for eigval in rescaled_eigvals:
    idx = np.argmin(np.abs(x - eigval))
    poly_at_eigvals.append(polynomial_values[idx])

fig, ax = plt.subplots(figsize=(10, 7.5), dpi=100)

ax.plot(x, polynomial_values, color='#1f77b4', linewidth=1.5, alpha=0.9)

ax.axvspan(-delta_scaled, delta_scaled, color='grey', alpha=0.4)
ax.text(
    0,
    1.02,
    r'$\leftarrow\Delta = ' + str(np.round(delta_scaled, 2)) + r'\rightarrow$',
    ha='center',
    va='bottom',
    transform=ax.get_xaxis_transform(),
)

eigenvalue_colors = custom_cmap(np.linspace(0, 1, int(len(rescaled_eigvals) * 1.2)))
eigenvalue_handles = []
eigenvalue_labels = []

for i, (eigval, poly_val) in enumerate(zip(rescaled_eigvals, poly_at_eigvals)):
    color = eigenvalue_colors[i]
    eig_label = '$\\tilde{\\lambda}_{' + str(i) + '}\\,\\mathrm{of}\\,\\tilde{H}$'
    if i == targetIdx:
        eig_line = ax.axvline(
            eigval,
            color=color,
            linewidth=1.6,
            alpha=0.95,
            label=eig_label,
        )
    else:
        eig_line = ax.axvline(
            eigval,
            color=color,
            linewidth=1.1,
            alpha=0.6,
            label=eig_label,
        )
    eigenvalue_handles.append(eig_line)
    eigenvalue_labels.append(eig_label)
    if i == targetIdx:
        ax.plot(
            eigval,
            poly_val,
            'o',
            color=color,
            markersize=9,
            markeredgecolor='black',
            markeredgewidth=1.3,
            zorder=20,
        )
    else:
        ax.plot(
            eigval,
            poly_val,
            'o',
            color=color,
            markersize=7,
            markeredgecolor='black',
            markeredgewidth=1.0,
            alpha=0.8,
            zorder=20,
        )

delta_idx = np.argmin(np.abs(x - delta_scaled))
p_delta_val = polynomial_values[delta_idx]

inside_delta_mask = np.abs(x) <= delta_scaled
max_inside_idx = np.argmax(polynomial_values[inside_delta_mask])
max_inside_x = x[inside_delta_mask][max_inside_idx]
max_inside_val = polynomial_values[inside_delta_mask][max_inside_idx]

hline_xspan = 2
amp_line_center, = ax.plot(
    [-hline_xspan, hline_xspan],
    [max_inside_val, max_inside_val],
    color='green',
    linewidth=0.9,
    linestyle=':',
    zorder=15,
    label=r'$R_8(0,\Delta)$',
)
amp_line_max, = ax.plot(
    [-hline_xspan, hline_xspan],
    [p_delta_val, p_delta_val],
    color='red',
    linewidth=0.9,
    linestyle=':',
    zorder=15,
    label=r'$\max_{x \in \mathcal{D}_{\Delta}} R_8(x,\Delta)$',
)

arrow_y_low = min(p_delta_val, max_inside_val)
arrow_y_high = max(p_delta_val, max_inside_val)
if np.isclose(arrow_y_high, arrow_y_low, rtol=0.0, atol=1e-6):
    y_span = float(np.max(polynomial_values) - np.min(polynomial_values))
    pad = max(0.05 * y_span, 0.05)
    arrow_y_low -= 0.5 * pad
    arrow_y_high += 0.5 * pad
ax.annotate(
    '',
    xy=(-delta_scaled, arrow_y_high),
    xytext=(-delta_scaled, arrow_y_low),
    arrowprops=dict(arrowstyle='<->', lw=2.0, color='black'),
    zorder=30,
    annotation_clip=False,
)

mid_point = (arrow_y_low + arrow_y_high) / 2
ax.text(
    -delta_scaled - 0.015,
    mid_point,
    'Amplification',
    va='center',
    ha='right',
    rotation=90,
    zorder=30,
)

ax.set_xlim(-1.01, 1)
ax.set_xlabel('$x$')
ax.set_ylabel(r'$R_d(x,\Delta)$')

leg_amp = fig.legend(
    [amp_line_center, amp_line_max],
    [r'$R_8(0,\Delta)$', r'$\max_{x \in \mathcal{D}_{\Delta}}$' + ' ' + r'$R_8(x,\Delta)$'],
    loc='center',
    bbox_to_anchor=(0.72, 0.06),
    frameon=True,
    fancybox=True,
    shadow=True,
)
fig.add_artist(leg_amp)

eig_ncol = 2
eig_n = len(eigenvalue_handles)
eig_nrow = int(np.ceil(eig_n / eig_ncol))
_eig_grid_h = [[None] * eig_ncol for _ in range(eig_nrow)]
_eig_grid_l = [[None] * eig_ncol for _ in range(eig_nrow)]
for idx in range(eig_n):
    r, c = idx // eig_ncol, idx % eig_ncol
    _eig_grid_h[r][c] = eigenvalue_handles[idx]
    _eig_grid_l[r][c] = eigenvalue_labels[idx]
eig_handles_rowwise = []
eig_labels_rowwise = []
for c in range(eig_ncol):
    for r in range(eig_nrow):
        if _eig_grid_h[r][c] is not None:
            eig_handles_rowwise.append(_eig_grid_h[r][c])
            eig_labels_rowwise.append(_eig_grid_l[r][c])

leg_eig = fig.legend(
    eig_handles_rowwise,
    eig_labels_rowwise,
    loc='center',
    bbox_to_anchor=(0.35, 0.06),
    ncol=eig_ncol,
    title='Eigenvalues',
    frameon=True,
    fancybox=True,
    shadow=True,
)
fig.add_artist(leg_eig)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig(os.path.join(_REPO_ROOT, 'paper_plots', 'figure6.pdf'), dpi=300, bbox_inches='tight')

print("Polynomial transformation plot generated!")
