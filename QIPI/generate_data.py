"""
Pre-generate the data for the combined QIPI plot (H2, LiH, BeH2).

"""

import os
import sys
import pickle

import numpy as np

# Make the repository root importable (qsvt, hamiltonian, chebyshev_*, nlft_qsp).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from QIPI.qsvt import block_encode, qsvt_circuit, qsvt_circuit_rot_synthesis, init_state
from QIPI.hamiltonian import overlap_square
from QIPI.chebyshev_qsp_phase_factors import get_phase_factors

# Parameterized molecule builder (lives in this folder).
from QIPI.hamiltonian import build as build_hamiltonian

import warnings
warnings.filterwarnings('ignore')

# ===== Parameters  =====
# Per-molecule degrees, given as the M parameter. The displayed polynomial
# degree is d = 2*M, so these correspond to:
#   H2  : d = [4, 8, 16]
#   LiH : d = [8, 16, 32]
#   BeH2: d = [16, 32, 64]
MOLECULE_DEGREES = {
    'H2': [2, 4, 8],
    'LiH': [4, 8, 16],
    'BeH2': [8, 16, 32],
}
PRECISIONS = [0.1, 0.01, 0.001]       # Precisions to plot
NUM_ITER = 25                         # Default number of iterations for convergence plot
# Per-molecule overrides for the number of convergence iterations (normal QSVT).
MOLECULE_NUM_ITER = {
    'H2': 5,
}
# Per-molecule convergence iterations for rotation-decomposition mode.
MOLECULE_NUM_ITER_ROT = {
    'H2': 5,
    'LiH': 15,
    'BeH2': 25,
}
# Per-molecule polynomial x-range for rotation-decomposition mode.
# Normal mode uses XRANGE_DEFAULT for all molecules.
XRANGE_DEFAULT = (-1.0, 1.0)
MOLECULE_XRANGE_ROT = {
    'H2': (-1.0, 1.0),
    'LiH': (-0.3, 0.3),
    'BeH2': (-0.3, 0.3),
}
TARGET_OVERLAP = 0.2
BLOCK_EN_ANCILLA = 1
QSVT_ANCILLA = 1

DATA_DIR = os.path.join(_THIS_DIR, 'data')

# Suffix used for the rotation-decomposition data files / plot.
ROT_SUFFIX = '_with_rot_decompose'


def generate_for_molecule(molecule, use_rotation_decomposition=False):
    """Run the full polynomial + convergence computation for one molecule.

    use_rotation_decomposition=False : normal QSVT (precision unused in the
        circuit). use_rotation_decomposition=True : QSVT with gridsynth
        rotation synthesis (the circuit depends on each precision).
    """
    tag = 'rotation decomposition' if use_rotation_decomposition else 'normal QSVT'
    degrees = MOLECULE_DEGREES[molecule]
    if use_rotation_decomposition:
        num_iter = MOLECULE_NUM_ITER_ROT.get(molecule, NUM_ITER)
        x_lo, x_hi = MOLECULE_XRANGE_ROT.get(molecule, XRANGE_DEFAULT)
    else:
        num_iter = MOLECULE_NUM_ITER.get(molecule, NUM_ITER)
        x_lo, x_hi = XRANGE_DEFAULT
    print(f"\n{'=' * 70}\nGenerating data for {molecule} ({tag}), "
          f"degrees M={degrees} (d={[2 * m for m in degrees]})\n{'=' * 70}")

    ham = build_hamiltonian(molecule)
    renorm_shifted = ham['renorm_shifted']
    targetEvec = ham['targetEvec']
    targetIdx = ham['targetIdx']
    rSeigenValues = ham['rSeigenValues']
    delta_scaled = ham['delta_scaled']
    system_qubits = ham['system_qubits']

    num_data_qubits = system_qubits + BLOCK_EN_ANCILLA
    num_qubits = num_data_qubits + QSVT_ANCILLA

    A = renorm_shifted
    U_A = block_encode(A)
    U_A_gate = UnitaryGate(U_A)

    # Initial guess state. The original sets np.random.seed(5737) once at the
    # top of the script before the single init_state call; we reseed here so
    # each molecule reproduces a fresh run of the original.
    np.random.seed(5737)
    initial_state = init_state(target_state=targetEvec, target_overlap=TARGET_OVERLAP)
    print('initial_state', initial_state)

    binary_strings = [bin(int(i))[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]

    # ----- Classical exact inverse convergence -----
    exactInv = np.linalg.inv(renorm_shifted)
    pMexact = exactInv
    transfVec = initial_state
    oveOfIter = [overlap_square(transfVec, targetEvec)]
    iterVec = [0]
    for iteration in range(num_iter):
        iterVec.append(iteration + 1)
        transfVecUnnorm = pMexact.dot(transfVec)
        transfVec = transfVecUnnorm / np.linalg.norm(transfVecUnnorm)
        oveOfIter.append(overlap_square(transfVec, targetEvec))

    # ===== Polynomial data (first_element_array) =====
    print("Generating polynomial data...")
    x = np.linspace(x_lo, x_hi, 100)

    # Create U gates for each x value
    U_list = []
    for i in range(len(x)):
        U = block_encode(np.array([[x[i], 0], [0, x[i]]]))
        U_gate = UnitaryGate(U)
        U_list.append(U_gate)

    first_element_array = np.zeros((len(degrees), len(PRECISIONS), len(x)))

    for j in range(len(degrees)):
        M = degrees[j]
        phi_list = get_phase_factors(M=M, a=delta_scaled, plot=False)

        for p_idx, precision in enumerate(PRECISIONS):
            print(f"  Processing polynomial: degree {M}, precision {precision}")
            first_element = []
            for i in range(len(x)):
                if use_rotation_decomposition:
                    current_qc = qsvt_circuit_rot_synthesis(U_gate=U_list[i], phi_list=phi_list, num_data_qubits=2, precision=precision)
                else:
                    current_qc = qsvt_circuit(U_gate=U_list[i], phi_list=phi_list, num_data_qubits=2)
                current_qc.h(2)  # qsvt ancilla
                current_qc.h(1)
                unitary = Operator(current_qc.to_gate())
                matrix = unitary.data[:2, :2]
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                first_element.append(np.real(eigenvalues[0]))
            first_element_array[j, p_idx, :] = np.array(first_element)

    # ===== Convergence data (overlap lists) =====
    print("Generating convergence data...")
    # overlap_lists[degree_idx][precision_idx] = list of overlaps (length NUM_ITER+1)
    overlap_lists = [[None for _ in PRECISIONS] for _ in degrees]

    for degree_idx, M in enumerate(degrees):
        print(f'  Processing convergence: degree d = {2 * M}')
        phi_list = get_phase_factors(M=M, a=delta_scaled, plot=False)

        for precision_idx, precision in enumerate(PRECISIONS):
            print(f'    Processing precision = {precision}')
            if use_rotation_decomposition:
                qsvt_A = qsvt_circuit_rot_synthesis(U_gate=U_A_gate, phi_list=phi_list, num_data_qubits=num_data_qubits, precision=precision)
            else:
                qsvt_A = qsvt_circuit(U_gate=U_A_gate, phi_list=phi_list, num_data_qubits=num_data_qubits)

            overlap_list = [overlap_square(initial_state, targetEvec)]
            current_state = initial_state.copy()

            for i in range(num_iter):
                if (i + 1) % 20 == 0:
                    print(f'      iteration {i + 1} of {num_iter}')

                ancilla = QuantumRegister(QSVT_ANCILLA, name='anc')
                data = QuantumRegister(num_data_qubits, name='data')
                cl = ClassicalRegister(num_data_qubits, name='cl')
                qc = QuantumCircuit(data, ancilla, cl)

                qc.initialize(current_state, range(system_qubits))
                qc.append(qsvt_A, range(num_qubits))
                qc.h(ancilla[0])  # qsvt ancilla
                qc.h(data[-1])

                statevector = Statevector.from_instruction(qc)
                prob_amps = []
                # post select on qsvt ancilla and last data qubit being zero
                for k in range(len(binary_strings)):
                    if binary_strings[k][0] == '0' and binary_strings[k][1] == '0':
                        prob_amps.append(statevector[k])
                prob_amps = prob_amps / np.linalg.norm(prob_amps)

                overlap = overlap_square(prob_amps, targetEvec)
                overlap_list.append(overlap)
                current_state = prob_amps

            overlap_lists[degree_idx][precision_idx] = overlap_list

    result = {
        'molecule': molecule,
        'use_rotation_decomposition': use_rotation_decomposition,
        'degrees': degrees,
        'precisions': PRECISIONS,
        'num_iter': num_iter,
        'x': x,
        'xlim': (x_lo, x_hi),
        'first_element_array': first_element_array,
        'rSeigenValues': rSeigenValues,
        'targetIdx': targetIdx,
        'delta_scaled': delta_scaled,
        'system_qubits': system_qubits,
        'iterVec': iterVec,
        'oveOfIter': oveOfIter,
        'overlap_lists': overlap_lists,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    suffix = ROT_SUFFIX if use_rotation_decomposition else ''
    out_path = os.path.join(DATA_DIR, f'{molecule}{suffix}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved {out_path}")
    return out_path


def main():
    # Usage: python -m QIPI.generate_data [--rot] [molecule ...]
    #   --rot  generate the rotation-decomposition data (saved as
    #          <molecule>_with_rot_decompose.pkl). Default: normal QSVT data.
    argv = sys.argv[1:]
    use_rotation_decomposition = '--rot' in argv
    molecules = [a for a in argv if a != '--rot'] or ['H2', 'LiH', 'BeH2']
    for molecule in molecules:
        generate_for_molecule(molecule, use_rotation_decomposition=use_rotation_decomposition)
    print("\nAll data generation completed!")


if __name__ == "__main__":
    main()
