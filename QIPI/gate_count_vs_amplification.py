import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from scipy.signal import find_peaks

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from QIPI.chebyshev_qsp_phase_factors import get_phase_factors
from QIPI.qsvt import *


# Set random seed for reproducibility
np.random.seed(5737)

plt.style.use(os.path.join(_REPO_ROOT, 'plotstylefile.mplstyle'))

plotdir = os.path.join(_REPO_ROOT, "paper_plots") + "/"


# Parameters
degrees = [4, 8, 16]  # Degrees to plot
precisions = [0.1, 0.01, 0.005, 0.001]  # Precisions to plot
delta_scaled_values = [0.05, 0.12, 0.17, 0.25]  # Different delta_scaled values to plot
x_eval = np.linspace(-0.5, 0.5, 100)  # Range of x values to evaluate polynomial


# Define linestyles for different precisions
precision_linestyles = ['-', '--', '-.', ':']  # Solid for 1e-1, dashed for 1e-2, dotted for 1e-3


def calculate_amplification_ratio(degree, precision, delta_scaled_val):
    """Calculate amplification ratio: highest peak (at x=0) / second highest peak"""
    phi_list = get_phase_factors(M=degree, a=delta_scaled_val, plot=False)

    polynomial_values = []
    for x_val in x_eval:
        U = block_encode(np.array([[x_val, 0], [0, x_val]]))
        U_gate = UnitaryGate(U)

        current_qc = qsvt_circuit_rot_synthesis(
            U_gate=U_gate, phi_list=phi_list, num_data_qubits=2, precision=precision
        )
        current_qc.h(2)
        current_qc.h(1)


        unitary = Operator(current_qc.to_gate())
        matrix = unitary.data[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        polynomial_values.append(np.real(eigenvalues[0]))

    polynomial_values = np.array(polynomial_values)

    origin_idx = len(x_eval) // 2
    origin_value = np.abs(polynomial_values[origin_idx])

    peaks, properties = find_peaks(
        np.abs(polynomial_values),
        height=0.001,
        distance=10,
    )

    if len(peaks) >= 2:
        peak_values = np.abs(polynomial_values[peaks])

        origin_tolerance = 10
        non_origin_peaks = []
        for i, peak_idx in enumerate(peaks):
            if abs(peak_idx - origin_idx) > origin_tolerance:
                non_origin_peaks.append(peak_values[i])

        if len(non_origin_peaks) > 0:
            second_max_peak = np.max(non_origin_peaks)
            amplification_ratio = origin_value / second_max_peak
            return amplification_ratio

    non_origin_values = np.abs(polynomial_values)
    non_origin_values[origin_idx - 20:origin_idx + 20] = 0
    max_non_origin = np.max(non_origin_values)
    if max_non_origin > 0:
        return origin_value / max_non_origin
    else:
        return 1


t_counts = np.zeros((len(delta_scaled_values), len(degrees), len(precisions)))
amplifications = np.zeros((len(delta_scaled_values), len(degrees), len(precisions)))

for delta_idx, delta_scaled_val in enumerate(delta_scaled_values):
    print(f"Processing delta_scaled = {delta_scaled_val}")

    for degree_idx, degree in enumerate(degrees):
        print(f"  Processing degree {degree}")

        for precision_idx, precision in enumerate(precisions):
            print(f"    Processing precision {precision}")

            phi_list = get_phase_factors(M=degree, a=delta_scaled_val, plot=False)
            t_count_list = []
            for phase_factor in phi_list:
                rot_decomp = rot_decomposition(2 * phase_factor, precision)
                t_count_list.append(rot_decomp.count('T'))
            total_t_count = sum(t_count_list)

            amplification_ratio = calculate_amplification_ratio(degree, precision, delta_scaled_val)

            t_counts[delta_idx, degree_idx, precision_idx] = total_t_count
            amplifications[delta_idx, degree_idx, precision_idx] = amplification_ratio

fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=100)
fig.patch.set_facecolor('white')

degree_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
delta_linestyles = ['-', '--', '-.', ':']
precision_markers = ['*', 's', '^', 'o']
precision_markersizes = [10, 6, 6, 6]

for delta_idx, delta_scaled_val in enumerate(delta_scaled_values):
    for degree_idx, degree in enumerate(degrees):
        ax.plot(
            t_counts[delta_idx, degree_idx, :],
            amplifications[delta_idx, degree_idx, :],
            color=degree_colors[degree_idx],
            linewidth=2,
            alpha=0.8,
            linestyle=delta_linestyles[delta_idx],
        )

        for precision_idx, precision in enumerate(precisions):
            ax.plot(
                t_counts[delta_idx, degree_idx, precision_idx],
                amplifications[delta_idx, degree_idx, precision_idx],
                color=degree_colors[degree_idx],
                linewidth=0,
                marker=precision_markers[precision_idx],
                markersize=precision_markersizes[precision_idx],
                markeredgecolor='black',
                markeredgewidth=1,
                alpha=0.9,
            )

ax.set_xlabel(r'$\mathrm{T}$ gate count')
ax.set_ylabel(r'Amplification ratio $\rho_{amp}$')

degree_legend_elements = []
for degree_idx, degree in enumerate(degrees):
    line = plt.Line2D([0], [0], color=degree_colors[degree_idx], linewidth=2, label=f'd = {2 * degree}')
    degree_legend_elements.append(line)

delta_legend_elements = []
for delta_idx, delta_scaled_val in enumerate(delta_scaled_values):
    line = plt.Line2D(
        [0], [0], color='black', linewidth=2, linestyle=delta_linestyles[delta_idx], label=f'Δ = {delta_scaled_val}'
    )
    delta_legend_elements.append(line)

precision_legend_elements = []
for precision_idx, precision in enumerate(precisions):
    line = plt.Line2D(
        [0],
        [0],
        marker=precision_markers[precision_idx],
        color='black',
        markersize=precision_markersizes[precision_idx],
        markeredgecolor='black',
        markeredgewidth=1,
        linewidth=0,
        label=f'ε = {precision:.0e}',
    )
    precision_legend_elements.append(line)

legend1 = fig.legend(
    handles=degree_legend_elements,
    loc='center',
    bbox_to_anchor=(0.25, -0.05),
    title='Degree',
    frameon=True,
    fancybox=True,
    shadow=True,
)
legend2 = fig.legend(
    handles=delta_legend_elements,
    loc='center',
    bbox_to_anchor=(0.485, -0.08),
    title='Δ',
    frameon=True,
    fancybox=True,
    shadow=True,
)
legend3 = fig.legend(
    handles=precision_legend_elements,
    loc='center',
    bbox_to_anchor=(0.75, -0.08),
    title='Precision',
    frameon=True,
    fancybox=True,
    shadow=True,
)

fig.add_artist(legend1)
fig.add_artist(legend2)
fig.add_artist(legend3)

ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.2)
ax.tick_params(axis='x', which='minor', direction='out', length=3, width=1.0)
ax.tick_params(axis='y', which='minor', direction='out', length=3, width=1.0)

plt.tight_layout(pad=2.0)
plt.yscale('log')
plt.savefig(
    plotdir + 'figure11.pdf',
    dpi=300,
    bbox_inches='tight',
)
plt.show()

print("Plot generation completed!")
print(f"T counts shape: {t_counts.shape}")
print(f"Amplifications shape: {amplifications.shape}")
print(f"T counts: {t_counts}")
print(f"Amplifications: {amplifications}")
