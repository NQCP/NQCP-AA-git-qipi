"""
Figure 13: LiH initial state comparison (spin adapted vs cooled vs random).

Left panel : R_d(x, Delta) over the spectrum of the parity-mapped LiH Hamiltonian (lih_hamiltonian.py),
             shifted by the mean-field energy <Psi_SA|H|Psi_SA> of the spin adapted CSF.
Right panel: QIPI convergence for the spin adapted CSF, a cooled state and a random state,
             each with its exact inverse iteration as reference.

Usage:
    python QIPI/chemical_simulations/initial_state_comparison.py
    python -m QIPI.chemical_simulations.initial_state_comparison
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repository root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector
import matplotlib.pyplot as plt
from QIPI.chebyshev_qsp_phase_factors import get_phase_factors
from QIPI.hamiltonian import overlap_square
from QIPI.qsvt import block_encode, qsvt_circuit, init_state
from QIPI.chemical_simulations.lih_hamiltonian import (mat, scale, problemOfInterest, targetEvec, targetIdx, targetEval,
                                                       eigenValues, eigenVectors, delta_scaled)
import warnings
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib
from QIPI.chemical_simulations.prepStateg import stateprep_states

plt.style.use(_ROOT / 'plotstylefile.mplstyle')
plt.rcParams['mathtext.fontset'] = 'cm'  # LaTeX (Computer Modern) look for all math in this figure
plotdir = str(_ROOT / 'paper_plots') + '/'

# Define the colors for the gradient
color1 = '#3929b7'  # blue
color2 = '#901a1e'  # red

# Create the colormap
n_bins = 256  # Number of color gradients
colors = [color1, color2]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)
cmap = matplotlib.colormaps['plasma']

# Set random seed for reproducibility
np.random.seed(5737)
warnings.filterwarnings('ignore')
molecule = 'LiH'

# Persistent color mapping for degrees
# Default matplotlib color cycle (from plotstylefile.mplstyle)
default_color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', 
                       '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Initialize degree color mapping with predefined colors
degree_color_map = {
    2: '#1f77b4',  # blue
    4: '#ff7f0e',  # orange
    8: '#2ca02c',  # green
}

def get_degree_colors(degrees_list):
    """
    Get colors for degrees, maintaining consistency across plots.
    - Uses predefined colors for degrees 2, 4, 8
    - Assigns new colors from default cycle for new degrees
    - Preserves existing color assignments
    """
    global degree_color_map
    
    # Find which degrees need new colors
    new_degrees = [d for d in degrees_list if d not in degree_color_map]
    
    # Get set of already used colors
    used_colors = set(degree_color_map.values())
    
    # Assign colors to new degrees in order from the default cycle
    for degree in new_degrees:
        # Find the next color in the cycle that's not already used
        for color in default_color_cycle:
            if color not in used_colors:
                degree_color_map[degree] = color
                used_colors.add(color)
                break
        else:
            # If all default colors are used, cycle through them
            # Find the highest index of any used color and use the next one
            max_used_idx = max([default_color_cycle.index(c) for c in used_colors if c in default_color_cycle], default=-1)
            next_idx = (max_used_idx + 1) % len(default_color_cycle)
            degree_color_map[degree] = default_color_cycle[next_idx]
            used_colors.add(default_color_cycle[next_idx])
    
    # Return colors in the order of degrees_list
    return [degree_color_map[d] for d in degrees_list]

# Parameters
degrees = [16]  # Degrees to plot
legend_degrees = [2, 4, 8, 16, 32]  # All degrees to show in legen
num_iter = 50  # Number of iterations for convergence plot
block_en_ancilla = 1
qsvt_ancilla = 1
system_qubits = 10 # beh2:12, h2:2. lih:10
num_data_qubits = system_qubits + block_en_ancilla
num_qubits = num_data_qubits + qsvt_ancilla
target_overlap = 0.2

# initial_state = init_state(target_state=targetEvec, target_overlap=target_overlap)
# create a new initial state as zero on all places except 4028 place. 
# initial_state = np.zeros(len(targetEvec))
# initial_state[occupation] = 1

initial_state = stateprep_states(problemOfInterest, mapper_name="parity", n_qubits=system_qubits)[1]
print('initial_state', initial_state)
# check the shape of the initial state
print(len(initial_state))
# calculate the overlap of the initial state with the target state
overlap_list = []
for i in range(len(initial_state)):
    overlap = overlap_square(np.array([initial_state[i]]), targetEvec)
    overlap_list.append(overlap)
print('overlap of the initial state with the target state:', np.array(overlap_list))
# choose the initial state with the highest overlap with the target state
initial_state = initial_state[np.argmax(overlap_list)]
print('initial state:', initial_state)
# check if the initial state has 
# overlap of the initial state with the target state
overlap = overlap_square(initial_state, targetEvec)
print('overlap of the initial state with the target state:', overlap)

# Shift omega = mean-field energy <Psi_SA|H|Psi_SA> of the spin adapted CSF, instead of lih_hamiltonian.py's
# lam = E_t + 0.001; the scale stays the one-norm, so delta_scaled is unchanged
lam = np.real(np.vdot(initial_state, mat @ initial_state))
print('shift lam = <Psi_SA|H|Psi_SA> =', lam, ', lam - E_t =', lam - targetEval)
renorm_shifted = (mat - lam * np.eye(mat.shape[0])) / scale
assert np.max(np.abs((eigenValues - lam) / scale)) <= 1, 'scale does not bound the shifted spectrum'
rSeigenValues = (eigenValues - lam) / scale   # same (sorted) order as eigenValues, so targetIdx still applies
rSeigenVectors = eigenVectors
A = renorm_shifted
# lih_hamiltonian.py's lam differs from this one, so the block encoding is rebuilt here
U_A = block_encode(A)
U_A_gate = UnitaryGate(U_A, check_input=False)  # unitary by construction for ||A|| <= 1

# Same Gaussian cooling as cooled_initial_state in the original hamiltonian.py, but built only from the N = 4, S = 0 singlets (spin_0)
def cooled_initial_state_at_target(sigma):
    coefficients = 1

    # weights = np.exp(-(np.real(spin_0) - np.real(lam))**2 / (4 * sigma**2))
    weights = np.exp(-(eigenValues - np.real(lam))**2 / (4 * sigma**2))

    cooled_coefficients = coefficients * weights

    cooled_coefficients /= np.linalg.norm(cooled_coefficients)

    # cooled_state = np.array(spin_0_eigenvectors).T @ cooled_coefficients
    cooled_state = rSeigenVectors @ cooled_coefficients
    # summ of ampltitude squered should be 1, if its not, normalize it
    if np.abs(np.sum(np.abs(cooled_state)**2) - 1) > 1e-10:
        cooled_state = cooled_state / np.sqrt(np.sum(np.abs(cooled_state)**2))

    return cooled_state

# Cooled states for several sigmas: larger sigma -> lower overlap
# (singlet sector, centred on lam: sigma = 0.0059 gives ~0.61, close to the spin adapted state; sigma = 2.44 gives ~0.02)
# cooled_sigmas = [0.0059, 0.0117, 0.0553, 0.198, 0.934, 2.44]   # overlaps ~0.61, 0.40, 0.20, 0.10, 0.05, 0.02
cooled_sigmas = [0.0212]   # overlap 0.62 with the Gaussian centred on the mean-field shift (0.0187 gives 0.74)
cooled_states = {}
for sigma in cooled_sigmas:
    cooled_states[sigma] = cooled_initial_state_at_target(sigma)
    print(f'cooled state (sigma = {sigma}) overlap with target state:', overlap_square(cooled_states[sigma], targetEvec))

# Random state with overlap 0.02 with the target state
initial_state_random = init_state(target_state=targetEvec, target_overlap=0.04)
print('random initial state overlap with target state:', overlap_square(initial_state_random, targetEvec))


# print('overlap of the initial state with the target state:', overlap)
# exit()
binary_strings = [bin(int(i))[2:].zfill(num_qubits) for i in range(2**num_qubits)]

# The two initial states to compare in the convergence plot
initial_states = {'Spin adapted': initial_state}
for sigma in cooled_sigmas:
    initial_states[f'Cooled, $\\sigma$ = {sigma}'] = cooled_states[sigma]
initial_states['Random'] = initial_state_random
cooled_names = [name for name in initial_states if name.startswith('Cooled')]

state_markers = {'Spin adapted': 'o', 'Random': '^'}
state_exact_linestyles = {'Spin adapted': '--', 'Random': '--'}  # exact inverse: dashed, in the state's colour
qipi_color = '#2ca02c'  # green, for the polynomial (degree d)
state_colors = {'Spin adapted': '#1f77b4', 'Random': '#d62728'}  # one colour per initial state
cooled_colors = ['#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f'][:len(cooled_names)]
for name, c in zip(cooled_names, cooled_colors):
    state_markers[name] = 's'
    state_exact_linestyles[name] = '--'
    state_colors[name] = c

# clasical exact inverse
exactInv = np.linalg.inv(renorm_shifted)
oveOfIter_per_state = {}

for state_name, state_vec in initial_states.items():
    vec_norm = state_vec
    pMexact = exactInv
    transfVec=vec_norm
    oveOfIter = [overlap_square(transfVec, targetEvec)]
    iterVec = [0]

    for iter in range(num_iter):
        iterVec.append(iter + 1)
        transfVecUnnorm = pMexact.dot(transfVec)
        transfVec = transfVecUnnorm / np.linalg.norm(transfVecUnnorm)
        oveOfIter.append(overlap_square(transfVec, targetEvec))
    oveOfIter_per_state[state_name] = oveOfIter

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

# ===== FIRST SUBPLOT: POLYNOMIAL PLOTS =====
print("Generating polynomial plots...")

# Generate polynomials for different degrees
# x = np.linspace(-0.3, 0.3, 100)
x = np.linspace(-1, 1, 1000)
# Create U gates for each x value
U_list = []
U_dagger_list = []
for i in range(len(x)):
    U = block_encode(np.array([[x[i], 0], [0, x[i]]]))
    U_dagger = np.linalg.inv(U)
    U_gate = UnitaryGate(U)
    U_gate_dagger = UnitaryGate(U_dagger)
    U_list.append(U_gate)
    U_dagger_list.append(U_gate_dagger)

# Initialize array to store results
first_element_array = np.zeros((len(degrees), len(x)))

# Run QSVT iterations for each degree
for j in range(len(degrees)):
    M = degrees[j]
    phi_list = get_phase_factors(M=M, a=delta_scaled, plot=False)
    
    print(f"Processing polynomial: degree {M}")
    first_element = []
    
    for i in range(len(x)):
        current_qc = qsvt_circuit(U_gate=U_list[i], phi_list=phi_list, num_data_qubits=2)
        current_qc.h(2) # qsvt ancilla 
        current_qc.h(1) # last data qubit
        unitary = Operator(current_qc.to_gate())    
        matrix = unitary.data[:2,:2]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        first_element.append(np.real(eigenvalues[0]))
    
    first_element_array[j, :] = np.array(first_element)

# Plot polynomials in first subplot
rescaled_eigvals = rSeigenValues
eigenvalue_colors = custom_cmap(np.linspace(0, 1, len(rescaled_eigvals)))
# Initialize degree color mapping (populates degree_color_map for all degrees in list)
get_degree_colors(degrees)
# Also initialize colors for all legend degrees
get_degree_colors(legend_degrees)

# Plot target eigenvalues as vertical lines
# for i in range(targetIdx, targetIdx+4):
colors = custom_cmap(np.linspace(0, 1, int(len(rSeigenValues)*(1.2))))

for i in range(0, len(rSeigenValues)):
    color = colors[i]
    if (i==targetIdx):
        ax1.axvline(rSeigenValues[i], linewidth=1.5, color=color, label=r'$\tilde{{\lambda}}_{{{}}}$'.format(i) + " of " + r'A')
    else:
        ax1.axvline(rSeigenValues[i], linewidth=1, linestyle=':', color=color, alpha=0.5, label=r'$\tilde{{\lambda}}_{{{}}}$'.format(i) + " of " + r'A')



ax1.axvspan(-delta_scaled, delta_scaled, color='grey', alpha=0.4)
delta_patch = mpatches.Patch(color='grey',  alpha=0.4, label=r'$\Delta = \pm$'+ str(delta_scaled))
# Add delta annotation at the top of the gray bar
ax1.text(0, 1.02, r'$\leftarrow\Delta = ' + str(np.round(delta_scaled,2)) + r'\rightarrow$', ha='center', va='bottom', 
         transform=ax1.get_xaxis_transform())
# add straight line at x = 0
# ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.2)

# Plot polynomial curves for each degree
for degree_idx, degree in enumerate(degrees):
    ax1.plot(x, first_element_array[degree_idx, :], 
            color=qipi_color, linewidth=1.2, alpha=0.9, 
            linestyle='-') 
# xlim = (-0.3, 0.3)
# xlim = (-0.3, 0.3)
xlim = (-1, 1)
ax1.set_xlim(xlim)
ax1.set_xlabel('$x$')
# ax1.set_ylabel('$P(x)$')
ax1.set_ylabel(r'$R_d(x,\Delta)$')


# Create comprehensive legend elements for first subplot
legend_elements_ax1 = []

# Add eigenvalue lines manually
for i in range(0, len(rSeigenValues)):
    color = colors[i]
    if (i==targetIdx):
        line = plt.Line2D([0], [0], color=color, linewidth=1.5, 
                         label='Target eigenvalue ' + r'$\tilde{\lambda}_t$')#{{{}}}$'.format(i) + " of " + r'A')
        legend_elements_ax1.append((line, 'Target eigenvalue ' + r'$\tilde{\lambda}_t$'))#{{{}}}$'.format(i) + " of " + r'A'))
    else:
        if molecule == 'LiH' or molecule == 'BeH2':
            pass
        else:
            line = plt.Line2D([0], [0], color=color, linestyle='-', linewidth=1,
                         label='Target eigenvalue ' + r'$\tilde{\lambda}_t$')#{{{}}}$'.format(i) + " of " + r'A')
            legend_elements_ax1.append((line, 'Target eigenvalue ' + r'$\tilde{\lambda}_t$'))#{{{}}}$'.format(i) + " of " + r'A'))

# Add delta shift
# legend_elements_ax1.append((delta_patch, r'$\Delta = \pm$' + str(np.round(delta_scaled,2))))

# Degree legend elements will be created separately with all legend_degrees

# Add exact inverse

exact_inverse_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
                            label='Exact inverse')
legend_elements_ax1.append((exact_inverse_line, 'Exact inverse'))

# Legend will be created after both subplots are complete


# ===== SECOND SUBPLOT: CONVERGENCE PLOTS =====
print("Generating convergence plots...")
for state_name in initial_states:
    ax2.plot(iterVec, np.abs(oveOfIter_per_state[state_name]), state_exact_linestyles[state_name], color=state_colors[state_name], linewidth=1.2, label=f"Exact inverse ({state_name})")
for state_name, state_vec in initial_states.items():
    print(f'Initial state: {state_name}')
    # Run QSVT iterations for each degree
    for degree_idx, M in enumerate(degrees):
        print(f'Processing convergence: degree d = {2*M}')
    
        # Get phase factors for this degree
        phi_list = get_phase_factors(M=M, a=delta_scaled, plot=False)
        qsvt_A = qsvt_circuit(U_gate=U_A_gate, phi_list=phi_list, num_data_qubits=num_data_qubits)
    
        # Initialize lists to store results
        overlap_list = [overlap_square(state_vec, targetEvec)]
        current_state = state_vec.copy()
        
        # Run iterations
        for i in range(num_iter):
            if (i + 1) % 20 == 0:
                print(f'    iteration {i+1} of {num_iter}')
        
            # Create quantum circuit
            ancilla = QuantumRegister(qsvt_ancilla, name='anc')
            data = QuantumRegister(num_data_qubits, name='data')
            cl = ClassicalRegister(num_data_qubits, name='cl')
            qc = QuantumCircuit(data, ancilla, cl)
        
            # Start directly from |0>_qsvt-anc |0>_block-anc |current_state>_system: the same state
            # qc.initialize(current_state, range(system_qubits)) prepares, without synthesising a state-prep circuit
            init_vec = np.zeros(2**num_qubits, dtype=complex)
            init_vec[:2**system_qubits] = current_state
            qc.append(qsvt_A, range(num_qubits)) # apply the block encoded matrix to the data qubits

            qc.h(ancilla[0]) # qsvt ancilla
            # qc.h(data[-1]) # last data qubit
        
            # Get statevector
            statevector = Statevector(init_vec).evolve(qc)
            prob_amps = []
            # post selecting on the qsvt ancilla and the last data qubit being zero. 
            for j in range(len(binary_strings)):
                if binary_strings[j][0] == '0'and binary_strings[j][1] == '0':
                    prob_amps.append(statevector[j])
            prob_amps = prob_amps / np.linalg.norm(prob_amps)

            # Calculate overlap and update state
            overlap = overlap_square(prob_amps, targetEvec)
            overlap_list.append(overlap)
            current_state = prob_amps
    
        # Plot convergence for this degree
        # for iterations, use integer range
        ax2.plot(range(len(overlap_list)), overlap_list, color=state_colors[state_name], linewidth=1.2, 
                linestyle='-', marker=state_markers[state_name], markersize=7,
                markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)

# Legends: target eigenvalue, degree and exact inverse (one black line; the curves use each state's colour),
# and the QIPI curves by initial state
state_labels = {'Spin adapted': r'Spin-adapted ($|\Psi_{\mathrm{SA}}\rangle$)', 'Random': 'Random'}
state_labels.update({name: r'Cooled ($|\Psi_{\mathrm{cool}}\rangle$)' for name in cooled_names})
initial_state_handles = [plt.Line2D([0], [0], color=state_colors[state_name], linestyle='-', marker=state_markers[state_name],
                                    markersize=7, markeredgecolor='black', markeredgewidth=0.8, label=state_labels[state_name])
                         for state_name in initial_states]
degree_lines = [plt.Line2D([0], [0], color=colors[targetIdx], linewidth=1.5, label='Target eigenvalue ' + r'$\tilde{\lambda}_t$')]
degree_lines += [plt.Line2D([0], [0], color=qipi_color, linewidth=1.2, label=f'EFP with $d = {2 * degree}$') for degree in degrees]
degree_lines += [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Exact inverse')]
bottom_legends = []  # placed left to right with equal gaps after tight_layout (see below)
for handles, title in [(degree_lines, None), (initial_state_handles, 'Initial state')]:
    bottom_legends.append(fig.legend(handles=handles, title=title, loc='upper left', bbox_to_anchor=(0, 0.0),
                                     frameon=True, fancybox=True, shadow=True))
    fig.add_artist(bottom_legends[-1])

ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('Overlap ' + r'$|\langle \Psi_{t} | \Psi_k \rangle|^2$')
ax2.set_ylim(0, 1.1)

# Create separate legend boxes and combine them
import matplotlib.patches as patches

# Create separate legend elements for each category
eigenvalue_handles = []
eigenvalue_labels = []
delta_handles = []
delta_labels = []
degree_handles = []
degree_labels = []
exact_handles = []
exact_labels = []

# Separate the legend elements into categories
for handle, label in legend_elements_ax1:
    if 'lambda' in label: #and 'of A' in label:
        eigenvalue_handles.append(handle)
        eigenvalue_labels.append(label)
    elif 'Delta' in label or '\\pm' in label:
        delta_handles.append(handle)
        delta_labels.append(label)
    elif 'd =' in label:
        degree_handles.append(handle)
        degree_labels.append(label)
    elif 'Exact inverse' in label:
        exact_handles.append(handle)
        exact_labels.append(label)

# Create individual legends for each category (skip if molecule is H2)
legends = []
legend_titles = []

if molecule != 'H2' and molecule != 'LiH':
    if eigenvalue_handles and exact_handles:
        combined_handles = eigenvalue_handles + exact_handles
        combined_labels = eigenvalue_labels + exact_labels
        leg1 = fig.legend(combined_handles, combined_labels, loc='center', 
                         bbox_to_anchor=(0.30, -0.10), 
                         frameon=True, fancybox=True, shadow=True)
        legends.append(leg1)
        legend_titles.append('System')

    # if delta_handles or exact_handles:
    #     # Combine delta and exact inverse
    #     combined_handles = delta_handles + exact_handles
    #     combined_labels = delta_labels + exact_labels
    #     leg2 = fig.legend(combined_handles, combined_labels, loc='center', 
    #                      bbox_to_anchor=(0.50, -0.10), 
    #                      frameon=True, fancybox=True, shadow=True)
    #     legends.append(leg2)
    #     legend_titles.append('Target')

    # Degree legend: matplotlib fills ncol legends column-by-column (top-to-bottom per
    # column). Reorder handles so the visual layout is row-wise: first row smallest
    # three d values, next row remaining d values in increasing order.
    sorted_legend_degrees = sorted(legend_degrees)
    n_deg = len(sorted_legend_degrees)
    degree_legend_ncol = 3
    degree_legend_nrow = int(np.ceil(n_deg / degree_legend_ncol))
    _deg_grid = [[None] * degree_legend_ncol for _ in range(degree_legend_nrow)]
    for idx, degree in enumerate(sorted_legend_degrees):
        r, c = idx // degree_legend_ncol, idx % degree_legend_ncol
        _deg_grid[r][c] = degree
    degree_legend_order = []
    for c in range(degree_legend_ncol):
        for r in range(degree_legend_nrow):
            if _deg_grid[r][c] is not None:
                degree_legend_order.append(_deg_grid[r][c])

    degree_legend_handles = []
    degree_legend_labels = []
    for degree in degree_legend_order:
        rect = mpatches.Rectangle((0, 0), 1, 1, facecolor=degree_color_map[degree],
                                  label=f'$d = {2 * degree}$')
        degree_legend_handles.append(rect)
        degree_legend_labels.append(f'$d = {2 * degree}$')

    if degree_legend_handles:
        leg4 = fig.legend(degree_legend_handles, degree_legend_labels, loc='center', 
                         bbox_to_anchor=(0.65, -0.10), title='Degree', 
                         frameon=True, fancybox=True, shadow=True, ncol=degree_legend_ncol)
        legends.append(leg4)
        legend_titles.append('Degree')



    # Add all legends to the figure
    for leg in legends:
        fig.add_artist(leg)


plt.tight_layout(pad=2.0)
# bottom legends: equal gaps between the boxes, the group centred on the saved image. The image is cropped
# (bbox_inches='tight') to the panels including their tick and axis labels, so centre on that extent.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_gap = 0.03  # in figure widths
legend_widths = [leg.get_window_extent(renderer).width / fig.bbox.width for leg in bottom_legends]
panels_extent = matplotlib.transforms.Bbox.union([ax.get_tightbbox(renderer) for ax in (ax1, ax2)])
panels_centre = (panels_extent.x0 + panels_extent.x1) / 2 / fig.bbox.width
x_left = panels_centre - (sum(legend_widths) + legend_gap * (len(bottom_legends) - 1)) / 2
for leg in bottom_legends:
    leg.set_bbox_to_anchor((x_left, 0.0), transform=fig.transFigure)
    x_left += leg.get_window_extent(renderer).width / fig.bbox.width + legend_gap
filepath = plotdir + 'figure13.pdf'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.show()

print("Plot generation completed!") 
