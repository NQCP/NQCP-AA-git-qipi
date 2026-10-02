"""Three-panel LiH plot: full Jordan-Wigner spectrum vs the accessible (singlet) spectrum.

Target: eigenstate 121 of the full 12-qubit JW Hamiltonian (a singlet), shift lam = E_t + 0.001 in both panels.

Panels:
  (a) R_d(x, Delta) over the full JW spectrum, x = (E - lam)/scale, scale = 1-norm of H, Delta = (1/sqrt(12))/scale,
  (b) R_d(x, Delta) over the accessible spectrum only (the N = 4, S = 0 singlets), with
        s_acc = max(|max E_CSF - lam|, |min E_CSF - lam|)   (E_CSF = <CSF|H|CSF> over the singlet CSFs),
      enlarged if needed so that every singlet has |x| <= 1, and Delta = (1/sqrt(12))/s_acc,
  (c) QIPI convergence: random initial state on the full spectrum, spin adapted singlet CSF on the
      accessible spectrum, each with its exact inverse iteration as reference.

The JW Hamiltonian is built here; only helper functions are taken from the existing modules.

Figure 14.

Usage:
    python QIPI/chemical_simulations/spectrum_comparison.py
    python -m QIPI.chemical_simulations.spectrum_comparison
"""

import sys
from pathlib import Path
import warnings

_ROOT = Path(__file__).resolve().parents[2]  # repository root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_algorithms import NumPyEigensolver
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper

from QIPI.chebyshev_qsp_phase_factors import get_phase_factors
from QIPI.hamiltonian import generate_random_state_with_overlap
from QIPI.chemical_simulations.cache_utils import _cache_path, _operator_key, _load_cache, _save_cache
from QIPI.chemical_simulations.prepStateg import stateprep_states
from QIPI.qsvt import block_encode, qsvt_circuit

plt.style.use(_ROOT / 'plotstylefile.mplstyle')
plt.rcParams['mathtext.fontset'] = 'cm'  # LaTeX (Computer Modern) look for all math in this figure
font_scale = 1.35  # enlarge every font by the same factor (keeping the style file's proportions) for print legibility
for key in ['font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']:
    plt.rcParams[key] = plt.rcParams[key] * font_scale  # titles and legend titles are relative ('large') and follow font.size
plotdir = _ROOT / 'paper_plots'
warnings.filterwarnings('ignore')

# Parameters
molecule = 'LiH'
inter_dist = 3
targetIdx = 121
shift = 0.001               # lam = E_t + shift
M = 16                      # degree d = 2M
num_iter = 15
num_grid = 1000             # grid points for the polynomial curves
block_en_ancilla = 1
qsvt_ancilla = 1
system_qubits = 12
num_data_qubits = system_qubits + block_en_ancilla
num_qubits = num_data_qubits + qsvt_ancilla
random_overlap = 0.011       # matches the spin adapted c_0/c_1 = 5.40 (target / next dominant overlap)
random_seed = 5737
delta_default = 1 / np.sqrt(12)

custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#3929b7', '#901a1e'], N=256)
qipi_color = '#2ca02c'      # green: full-spectrum polynomial
qipi_color_acc = '#0a4a0a'  # darker green: accessible-spectrum polynomial
random_color = qipi_color          # initial states keep the colour of the polynomial of their spectrum
spin_adapted_color = qipi_color_acc


def overlap(a, b):
    return np.abs(np.vdot(b, a))**2


# ---------------- JW Hamiltonian and its (cached) eigendecomposition ----------------
problem = PySCFDriver(atom=f'Li 0 0 0; H 0 0 {inter_dist}', basis='sto3g', charge=0, spin=0,
                      unit=DistanceUnit.ANGSTROM).run()
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()[0])
mat = qubit_op.to_matrix()
assert qubit_op.num_qubits == system_qubits

# same cache key scheme as lih_hamiltonian.py, so the JW diagonalisation is cached between runs
_cache_file = _cache_path(molecule, _operator_key(qubit_op.simplify()), type(mapper).__name__, problem.num_particles)
_cache = _load_cache(_cache_file)
if _cache is not None:
    spin_0, spin_0_eigenvectors = _cache['spin_0'], _cache['spin_0_eigenvectors']
    eigenValues, eigenVectors = _cache['eigenValues'], _cache['eigenVectors']
else:
    def singlet_filter(eigenstate, eigenvalue, aux_values):
        return (np.isclose(np.round(aux_values['AngularMomentum'][0], 10), 0)
                and np.isclose(aux_values['ParticleNumber'][0], 4))
    algo = NumPyEigensolver(k=1000000)
    algo.filter_criterion = singlet_filter
    result = ExcitedStatesEigensolver(mapper, algo).solve(problem)
    spin_0 = np.asarray(result.eigenvalues)
    spin_0_eigenvectors = np.array([Statevector(vec[0]).data for vec in result.eigenstates])
    eigenValues, eigenVectors = np.linalg.eigh(mat)
    _save_cache(_cache_file, spin_0=spin_0, spin_0_eigenvectors=spin_0_eigenvectors,
                eigenValues=eigenValues, eigenVectors=eigenVectors)

sortIdx = np.argsort(eigenValues)
eigenValues, eigenVectors = eigenValues[sortIdx], eigenVectors[:, sortIdx]
targetEval = eigenValues[targetIdx]
targetEvec = eigenVectors[:, targetIdx] / np.linalg.norm(eigenVectors[:, targetIdx])
lam = targetEval + shift

# ---------------- (a) full JW spectrum: scale = 1-norm of H ----------------
scale = np.linalg.norm(mat, ord=1)
x_full = (eigenValues - lam) / scale
assert np.max(np.abs(x_full)) <= 1, 'scale does not bound the shifted spectrum'
delta_full = delta_default / scale
A_full = (mat - lam * np.eye(mat.shape[0])) / scale
# block_encode is unitary by construction for ||A|| <= 1; skip qiskit's O(n^3) check on the 8192 x 8192 matrix
U_A_full = UnitaryGate(block_encode(A_full), check_input=False)

np.random.seed(random_seed)
random_state = generate_random_state_with_overlap(targetEvec, random_overlap)
random_state /= np.linalg.norm(random_state)

# ---------------- (b) accessible spectrum: singlets only ----------------
singlet_csfs = [np.asarray(t) / np.linalg.norm(t)
                for t in stateprep_states(problem, mapper_name='jw', n_qubits=system_qubits)[0]]
E_csf = np.array([np.real(np.vdot(t, mat @ t)) for t in singlet_csfs])
spin_adapted_state = singlet_csfs[int(np.argmax([overlap(t, targetEvec) for t in singlet_csfs]))]
s_acc_csf = max(abs(E_csf.max() - lam), abs(E_csf.min() - lam))

E_sing = np.real(np.asarray(spin_0))
V_sing = np.array(spin_0_eigenvectors).T
# The exact singlet spectrum can reach beyond the CSF energy range, so s_acc_csf alone can leave
# |x| > 1, which cannot be block encoded. Enlarge s just enough to keep every singlet in [-1, 1].
s_acc_min = np.max(np.abs(E_sing - lam))
s_acc = max(s_acc_csf, s_acc_min)
if s_acc > s_acc_csf:
    print(f'WARNING: s from CSF energies = {s_acc_csf:.4f} gives max|x| = {s_acc_min / s_acc_csf:.4f} > 1; '
          f'using s = {s_acc:.4f} instead')
x_acc = (E_sing - lam) / s_acc
delta_acc = delta_default / s_acc

# (H - lam)/s_acc projected onto the singlet sector; the spin adapted state has no weight outside it
A_acc = (V_sing * x_acc) @ V_sing.conj().T
U_A_acc = UnitaryGate(block_encode(A_acc), check_input=False)

target_sing = int(np.argmin(np.abs(E_sing - targetEval)))
print(f'E_t = {targetEval:.6f}, lam = {lam:.6f}')
print(f'full:       scale = {scale:.4f}, Delta = {delta_full:.4f}, x_t = {x_full[targetIdx]:+.2e}')
print(f'accessible: s = {s_acc:.4f}, Delta = {delta_acc:.4f}, x_t = {x_acc[target_sing]:+.2e}, '
      f'|<target|singlet {target_sing}>|^2 = {overlap(V_sing[:, target_sing], targetEvec):.4f}')
print(f'random overlap = {overlap(random_state, targetEvec):.4f}, '
      f'spin adapted overlap = {overlap(spin_adapted_state, targetEvec):.4f}, '
      f'spin adapted weight outside singlets = {1 - np.linalg.norm(V_sing.conj().T @ spin_adapted_state)**2:.1e}')

phi_full = get_phase_factors(M=M, a=delta_full, plot=False)
phi_acc = get_phase_factors(M=M, a=delta_acc, plot=False)


# ---------------- polynomial curves (same 2x2 construction as the spin script) ----------------
def polynomial_curve(phi_list, xs):
    values = []
    for xv in xs:
        U_gate = UnitaryGate(block_encode(np.array([[xv, 0], [0, xv]])))
        qc = qsvt_circuit(U_gate=U_gate, phi_list=phi_list, num_data_qubits=2)
        qc.h(2)  # qsvt ancilla
        qc.h(1)  # last data qubit
        values.append(np.real(np.linalg.eigvals(Operator(qc.to_gate()).data[:2, :2])[0]))
    return np.array(values)


x_grid = np.linspace(-1, 1, num_grid)
print('Generating polynomial curves...')
poly_full = polynomial_curve(phi_full, x_grid)
poly_acc = polynomial_curve(phi_acc, x_grid)


# ---------------- convergence ----------------
def qipi_convergence(U_A_gate, phi_list, state):
    qsvt_A = qsvt_circuit(U_gate=U_A_gate, phi_list=phi_list, num_data_qubits=num_data_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.append(qsvt_A, range(num_qubits))
    qc.h(num_qubits - 1)  # qsvt ancilla
    overlaps = [overlap(state, targetEvec)]
    current = state.copy()
    for i in range(num_iter):
        init_vec = np.zeros(2**num_qubits, dtype=complex)
        init_vec[:2**system_qubits] = current
        out = Statevector(init_vec).evolve(qc).data
        # post-select the qsvt ancilla and the block-encoding ancilla (two most significant qubits) on 0
        post = out[:2**system_qubits]
        current = post / np.linalg.norm(post)
        overlaps.append(overlap(current, targetEvec))
        if (i + 1) % 5 == 0:
            print(f'    iteration {i + 1} of {num_iter}: overlap {overlaps[-1]:.4f}')
    return np.array(overlaps)


def exact_inverse_convergence(A_inv, state):
    overlaps = [overlap(state, targetEvec)]
    current = state.copy()
    for _ in range(num_iter):
        current = A_inv @ current
        current /= np.linalg.norm(current)
        overlaps.append(overlap(current, targetEvec))
    return np.array(overlaps)


A_inv_full = (eigenVectors / x_full) @ eigenVectors.conj().T
A_inv_acc = (V_sing / x_acc) @ V_sing.conj().T

print('QIPI, full JW spectrum, random state')
qipi_full = qipi_convergence(U_A_full, phi_full, random_state)
print('QIPI, accessible spectrum, spin adapted state')
qipi_acc = qipi_convergence(U_A_acc, phi_acc, spin_adapted_state)
exact_full = exact_inverse_convergence(A_inv_full, random_state)
exact_acc = exact_inverse_convergence(A_inv_acc, spin_adapted_state)


# ---------------- plot ----------------
def draw_spectrum_panel(ax, xs, target, poly, delta, poly_color):
    line_colors = custom_cmap(np.linspace(0, 1, int(len(xs) * 1.2)))
    for i in np.argsort(xs):
        if i == target:
            ax.axvline(xs[i], linewidth=1.5, color=line_colors[i], zorder=3)
        else:
            ax.axvline(xs[i], linewidth=1, linestyle=':', color=line_colors[i], alpha=0.5)
    ax.axvspan(-delta, delta, color='grey', alpha=0.4)
    ax.text(0, 1.02, r'$\leftarrow\Delta = ' + str(np.round(delta, 3)) + r'\rightarrow$', ha='center', va='bottom',
            transform=ax.get_xaxis_transform())
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
    ax.plot(x_grid, poly, color=poly_color, linewidth=1.2, alpha=0.9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$R_d(x,\Delta)$')


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), dpi=100)  # 7 in: room for the 1.35x fonts at the original panel shape

draw_spectrum_panel(ax1, x_full, targetIdx, poly_full, delta_full, qipi_color)
ax1.set_title('(a) Full spectrum', pad=40 * font_scale)
draw_spectrum_panel(ax2, x_acc, target_sing, poly_acc, delta_acc, qipi_color_acc)
ax2.set_title('(b) Accessible spectrum', pad=40 * font_scale)

iters = np.arange(num_iter + 1)
ax3.plot(iters, exact_full, '--', color=random_color, linewidth=1.2)
ax3.plot(iters, exact_acc, '--', color=spin_adapted_color, linewidth=1.2)
ax3.plot(iters, qipi_full, color=random_color, linewidth=1.2, marker='^', markersize=7,
         markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)
ax3.plot(iters, qipi_acc, color=spin_adapted_color, linewidth=1.2, marker='o', markersize=7,
         markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)
ax3.set_xlabel('Iteration $k$')
ax3.set_ylabel('Overlap ' + r'$|\langle \Psi_{t} | \Psi_k \rangle|^2$')
ax3.set_ylim(0, 1.1)
ax3.set_title('(c) Convergence', pad=40 * font_scale)

target_color = custom_cmap(np.linspace(0, 1, int(len(x_full) * 1.2)))[targetIdx]  # as drawn in panel (a)
# legends: target eigenvalue, the two polynomials and exact inverse (one black dashed line; the curves use each
# state's colour), and the QIPI curves by initial state
spectrum_handles = [
    plt.Line2D([0], [0], color=target_color, linewidth=1.5, label='Target eigenvalue ' + r'$\tilde{\lambda}_t$'),
    plt.Line2D([0], [0], color=qipi_color, linewidth=1.2,
               label=f'$R_{{{2 * M}}}(x, \\Delta = {np.round(delta_full, 3)})$'),
    plt.Line2D([0], [0], color=qipi_color_acc, linewidth=1.2,
               label=f'$R_{{{2 * M}}}(x, \\Delta = {np.round(delta_acc, 3)})$'),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Exact inverse'),
]
qipi_handles = [
    plt.Line2D([0], [0], color=spin_adapted_color, marker='o', markersize=7, markeredgecolor='black', markeredgewidth=0.8,
               label=r'Spin-adapted ($|\Psi_{\mathrm{SA}}\rangle$)'),
    plt.Line2D([0], [0], color=random_color, marker='^', markersize=7, markeredgecolor='black', markeredgewidth=0.8,
               label='Random'),
]
bottom_legends = []  # placed left to right with equal gaps after tight_layout (see below)
for handles, title in [(spectrum_handles, None), (qipi_handles, 'Initial state')]:
    bottom_legends.append(fig.legend(handles=handles, title=title, loc='upper left', bbox_to_anchor=(0, 0.0),
                                     frameon=True, fancybox=True, shadow=True))
    fig.add_artist(bottom_legends[-1])

plt.tight_layout(pad=2.0)
# bottom legends: equal gaps between the boxes, the group centred on the saved image. The image is cropped
# (bbox_inches='tight') to the panels including their tick and axis labels, so centre on that extent.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_gap = 0.03  # in figure widths
legend_widths = [leg.get_window_extent(renderer).width / fig.bbox.width for leg in bottom_legends]
panels_extent = matplotlib.transforms.Bbox.union([ax.get_tightbbox(renderer) for ax in (ax1, ax2, ax3)])
panels_centre = (panels_extent.x0 + panels_extent.x1) / 2 / fig.bbox.width
x_left = panels_centre - (sum(legend_widths) + legend_gap * (len(bottom_legends) - 1)) / 2
for leg in bottom_legends:
    leg.set_bbox_to_anchor((x_left, 0.0), transform=fig.transFigure)
    x_left += leg.get_window_extent(renderer).width / fig.bbox.width + legend_gap
filepath = plotdir / 'figure14.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print('saved', filepath)
plt.show()
