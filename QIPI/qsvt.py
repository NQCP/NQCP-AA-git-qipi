import re

import mpmath
import numpy as np
from pygridsynth.gridsynth import gridsynth_gates
from pygridsynth.myplot import plot_sol
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate

from QIPI.hamiltonian import generate_random_state_with_overlap

np.random.seed(5737)

w = np.exp(1j * np.pi / 4)
W = np.array([[w, 0], [0, w]])
W_gate = UnitaryGate(W, label='W')


def sqrt_I_minus_AA(A):
    A = np.asarray(A, dtype=np.complex128)
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S = np.clip(S, 0.0, 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - S**2))
    return (U * r) @ Vh


def block_encode(A):
    A = np.asarray(A, dtype=np.complex128)
    B = sqrt_I_minus_AA(A)
    top = np.concatenate((A, 1j * B), axis=1)
    bottom = np.concatenate((1j * B, A.conj().T), axis=1)
    return np.concatenate((top, bottom), axis=0)


def apply_pi_phi(qc, ancilla, data, phi):
    """Apply projector-controlled phase gate Pi_phi."""
    qc.x(data[-1])
    qc.cx(data[-1], ancilla[0])
    qc.x(data[-1])

    qc.rz(2 * phi, ancilla[0])

    qc.x(data[-1])
    qc.cx(data[-1], ancilla[0])
    qc.x(data[-1])


def qsvt_circuit(U_gate: Gate, phi_list: list, num_data_qubits: int):
    """QSVT circuit for a block encoded matrix A."""
    ancilla = QuantumRegister(1, name='anc')
    data = QuantumRegister(num_data_qubits, name='data')

    qc = QuantumCircuit(data, ancilla)
    qc.h(ancilla[0])
    qc.h(data[-1])

    for i in range(0, (len(phi_list) - 2), 2):
        apply_pi_phi(qc, ancilla, data, phi_list[i])
        qc.append(U_gate, data)
        apply_pi_phi(qc, ancilla, data, phi_list[i + 1])
        qc.append(U_gate, data)
    apply_pi_phi(qc, ancilla, data, phi_list[-1])

    return qc


def init_state(target_state, target_overlap):
    """Initial guess state."""
    state = generate_random_state_with_overlap(target_state, target_overlap)
    print('initial statevector:', np.round(state, 4))
    state = state / np.linalg.norm(state)
    return state


def phase_align(state, target):
    phase = np.angle(np.vdot(target, state))
    return state * np.exp(-1j * phase)


def rot_decomposition(angle, precision):
    mpmath.mp.dps = 256

    epsilon = mpmath.mpmathify(str(precision))
    theta = mpmath.mpmathify(str(angle))
    gates = gridsynth_gates(theta=theta, epsilon=epsilon)

    return gates


def apply_pi_phi_rot_synthesis(qc, ancilla, data, phi, precision):
    """Apply projector-controlled phase gate Pi_phi."""
    qc.x(data[-1])
    qc.cx(data[-1], ancilla[0])
    qc.x(data[-1])

    gate_dcomp_list = rot_decomposition(2 * phi, precision)
    gate_dcomp_list = gate_dcomp_list[::-1]

    for gate in gate_dcomp_list:
        if gate == 'H':
            qc.h(ancilla[0])
        elif gate == 'T':
            qc.t(ancilla[0])
        elif gate == 'S':
            qc.s(ancilla[0])
        elif gate == 'W':
            qc.append(W_gate, [ancilla[0]])
        elif gate == 'X':
            qc.x(ancilla[0])
        elif gate == 'Y':
            qc.y(ancilla[0])
        elif gate == 'Z':
            qc.z(ancilla[0])

    qc.x(data[-1])
    qc.cx(data[-1], ancilla[0])
    qc.x(data[-1])


def qsvt_circuit_rot_synthesis(U_gate: Gate, phi_list: list, num_data_qubits: int, precision: float):
    """QSVT circuit for a block encoded matrix A."""
    ancilla = QuantumRegister(1, name='anc')
    data = QuantumRegister(num_data_qubits, name='data')

    qc = QuantumCircuit(data, ancilla)

    qc.h(ancilla[0])
    qc.h(data[-1])

    for i in range(0, (len(phi_list) - 2), 2):
        apply_pi_phi_rot_synthesis(qc, ancilla, data, phi_list[i], precision)
        qc.append(U_gate, data)
        apply_pi_phi_rot_synthesis(qc, ancilla, data, phi_list[i + 1], precision)
        qc.append(U_gate, data)
    apply_pi_phi_rot_synthesis(qc, ancilla, data, phi_list[-1], precision)

    return qc


def qasm_rotation_decomposition(file_name):
    with open(file_name, "r") as qasm_file:
        qasm_data = qasm_file.read()

    for i in range(len(qasm_data.split('\n'))):
        with open('qsvt_A_decomposition.qasm', 'a') as qasm_file:
            match = re.match(r'rz\(([^)]+)\)\s+([^;]+);?', qasm_data.split('\n')[i])
            if match:
                angle = match.group(1)

                if 'pi' in angle:
                    angle = angle.replace('pi', 'np.pi')
                    angle = eval(angle)

                angle = float(angle)
                qubit = match.group(2).strip()

                gate_dcomp_list = rot_decomposition(2 * angle, 0.001)
                gate_dcomp_list = gate_dcomp_list[::-1]
                for gate in gate_dcomp_list:
                    if gate == 'W':
                        pass
                    else:
                        qasm_file.write(f'{gate.lower()} {qubit};\n')
            else:
                qasm_file.write(qasm_data.split('\n')[i] + '\n')
