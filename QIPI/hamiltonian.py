"""
Parameterized Hamiltonian builder for the QIPI plot.

"""

import numpy as np
import warnings
import scipy

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# State / overlap helpers (ported from the old top-level hamiltonian.py and
# qsvt.py so the QIPI folder is self-contained).
# ---------------------------------------------------------------------------

def generate_random_state_with_overlap(target_state, target_overlap):
    """Generate a random quantum state with a specified overlap with a target.

    Args:
        target_state (np.ndarray): normalized target state vector.
        target_overlap (float): desired overlap in [0, 1].

    Returns:
        np.ndarray: a normalized state with the requested overlap.
    """
    if not 0 <= target_overlap <= 1:
        raise ValueError("Target overlap must be between 0 and 1")
    if abs(np.linalg.norm(target_state) - 1) > 1e-10:
        raise ValueError("Target state must be normalized")

    # Random component orthogonal to the target.
    random_state = np.random.randn(len(target_state)) + 1j * np.random.randn(len(target_state))
    random_state -= np.vdot(target_state, random_state) * target_state
    random_state /= np.linalg.norm(random_state)

    return np.sqrt(target_overlap) * target_state + np.sqrt(1 - target_overlap) * random_state


def similarity_cosine(A, B):
    """Cosine similarity dot(A, B) / (||A|| ||B||)."""
    return np.real(np.dot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B))


def overlap_square(A, B):
    """Squared overlap |<A|B>|^2 (using a plain dot product, as in the original)."""
    return np.abs(np.dot(A, B)) ** 2


def init_state(target_state, target_overlap):
    """Initial guess state with a given overlap with the target (normalized)."""
    state = generate_random_state_with_overlap(target_state, target_overlap)
    state = state / np.linalg.norm(state)
    return state


def build(molecule):
    """Build the renormalized, shifted Hamiltonian for a given molecule.

    Returns a dict with everything the QIPI plots need:
        renorm_shifted : the matrix A used in the QSVT pipeline
        targetEvec     : target eigenvector (normalized)
        targetIdx      : index of the target eigenvalue
        rSeigenValues  : sorted eigenvalues of renorm_shifted
        delta_scaled   : scaled spectral gap parameter
        system_qubits  : number of system qubits (qubit_op.num_qubits)
    """

    delta = 1 / np.sqrt(12)

    # ===== H2 =====
    if molecule == "H2":
        inter_dist = 0.7414
        atom_string = 'H 0 0 0; H 0 0 ' + str(inter_dist)
        targetIdx = 2

        driver = PySCFDriver(
            atom=atom_string,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

        full_problem = driver.run()
        mapper = ParityMapper(full_problem.num_particles)
        qubit_op = mapper.map(full_problem.second_q_ops()[0])

        problemOfInterest = full_problem

        matrix_op = qubit_op.to_matrix()
        num_qubits = qubit_op.num_qubits
        print(num_qubits)

    # ===== LiH =====
    elif molecule == "LiH":
        inter_dist = 1.6
        atom_string = 'Li 0 0 0; H 0 0 ' + str(inter_dist)
        targetIdx = 622

        driver = PySCFDriver(
            atom=atom_string,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

        full_problem = driver.run()

        # Freeze Core
        fc_transformer = FreezeCoreTransformer()
        fc_problem = fc_transformer.transform(full_problem)

        # Freeze Core with two virtuals removed
        fc_transformer = FreezeCoreTransformer(remove_orbitals=[4, 5])
        fc_problem = fc_transformer.transform(full_problem)
        as_transformer = ActiveSpaceTransformer(2, 2)
        as_problem = as_transformer.transform(full_problem)

        problemOfInterest = full_problem
        numalphas = scipy.special.binom(problemOfInterest.num_spatial_orbitals, 2)
        numbetas = scipy.special.binom(problemOfInterest.num_spatial_orbitals, 2)
        problemsize = numalphas * numbetas

        mapper = ParityMapper(problemOfInterest.num_particles)
        qubit_op = mapper.map(problemOfInterest.second_q_ops()[0])
        matrix_op = qubit_op.to_matrix()
        num_qubits = qubit_op.num_qubits
        print('num_qubits:', num_qubits)

    # ===== BeH2 =====
    elif molecule == "BeH2":
        inter_dist = 1.326
        atom_string = 'Be 0 0 0; H 0 0 ' + str(-inter_dist) + '; H 0 0 ' + str(inter_dist)
        targetIdx = 3079

        driver = PySCFDriver(
            atom=atom_string,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

        # BeH2 full electronic structure problem
        full_problem = driver.run()

        # Freeze Be 1s core + remove sigma* (highest-energy virtual MO)
        fc_transformer = FreezeCoreTransformer(remove_orbitals=[-1])
        fc_transformer = FreezeCoreTransformer()
        fc_problem = fc_transformer.transform(full_problem)
        # Freeze Core with two virtuals removed
        fc_transformer = FreezeCoreTransformer(remove_orbitals=[6, 7])
        fc_problem = fc_transformer.transform(full_problem)
        as_transformer = ActiveSpaceTransformer(2, 2)
        as_problem = as_transformer.transform(full_problem)

        problemOfInterest = full_problem

        # Parity mapping
        mapper = ParityMapper(num_particles=problemOfInterest.num_particles)
        qubit_op = mapper.map(problemOfInterest.second_q_ops()[0])

        mapper = ParityMapper(full_problem.num_particles)
        qubit_op = mapper.map(full_problem.second_q_ops()[0])

        matrix_op = qubit_op.to_matrix()
        num_qubits = qubit_op.num_qubits
        print('num_qubits:', num_qubits)

    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    print('num_particles:', problemOfInterest.num_particles)
    print('num_spatial_orbitals:', problemOfInterest.num_spatial_orbitals)

    

    mat = matrix_op

    eigenValues, eigenVectors = np.linalg.eig(mat)
    sortIdx = np.argsort(eigenValues)
    eigenValues = eigenValues[sortIdx]
    eigenVectors = eigenVectors[:, sortIdx]
    if molecule == "BeH2":
        print("\nGround state electronic energy (reduced problem):", eigenValues[0])
    print('eigenvalues near target:', eigenValues[targetIdx - 5:targetIdx + 5])

    e_min = np.min(eigenValues)
    e_max = np.max(eigenValues)

    targetEvec = eigenVectors[:, targetIdx]
    targetEval = eigenValues[targetIdx]
    print('targetEval:', np.round(targetEval, 4))
    alpha = np.linalg.norm(mat, ord=2)
    targetEvec = targetEvec / np.linalg.norm(targetEvec)

    if molecule == "BeH2":
        lam = targetEval + 0.0025 
    else:
        lam = targetEval + 0.025

    scale = max(np.abs(e_min - lam), np.abs(e_max - lam))
    delta_scaled = delta / (scale)
    print('delta_scaled:', delta_scaled)
    print('lam:', np.round(lam, 4))
    minEne = 0

    shifted_mat = mat + (minEne - lam) * np.eye(mat.shape[0])
    renorm_shifted = shifted_mat / (scale)
    print('renorm_shifted is Hermitian:', np.allclose(renorm_shifted, renorm_shifted.T.conj()))
    rSeigenValues, rSeigenVectors = np.linalg.eig(renorm_shifted)
    sortIdx = np.argsort(rSeigenValues)
    rSeigenValues = rSeigenValues[sortIdx]
    print('rSeigenValues:', rSeigenValues)

    return {
        'molecule': molecule,
        'renorm_shifted': renorm_shifted,
        'targetEvec': targetEvec,
        'targetIdx': targetIdx,
        'targetEval': targetEval,
        'rSeigenValues': rSeigenValues,
        'delta_scaled': delta_scaled,
        'system_qubits': num_qubits,
        'scale': scale,
        'lam': lam,
    }
