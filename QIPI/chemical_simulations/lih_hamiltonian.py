"""
LiH Hamiltonian for the spin-adapted initial state comparison (figure 13).

LiH at 3 Angstrom (sto3g), parity mapped with two-qubit reduction (10 qubits). The target is
eigenstate 7 of the full parity-mapped Hamiltonian, shifted by lam = E_t + 0.001 and rescaled by
the 1-norm of H:  renorm_shifted = (H - lam) / ||H||_1.

This is the LiH setup of the original hamiltonian.py, kept separate from QIPI/hamiltonian.py
(which builds the LiH case of figures 8-12). Importing the module builds everything at module level;
the diagonalisations are cached in QIPI/chemical_simulations/data/ (see cache_utils.py).
"""

import warnings

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_algorithms import NumPyEigensolver
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit

from QIPI.chemical_simulations.cache_utils import _cache_path, _operator_key, _load_cache, _save_cache

warnings.filterwarnings('ignore')

molecule = 'LiH'
inter_dist = 3
atom_string = 'Li 0 0 0; H 0 0 ' + str(inter_dist)
targetIdx = 7  # for parity mapped spin adapted simulation
delta = 1 / np.sqrt(12)

driver = PySCFDriver(
    atom=atom_string,
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
problemOfInterest = driver.run()

mapper = ParityMapper(problemOfInterest.num_particles)
qubit_op = mapper.map(problemOfInterest.second_q_ops()[0])
matrix_op = qubit_op.to_matrix()
print('num_qubits:', qubit_op.num_qubits)
print('num_particles:', problemOfInterest.num_particles)
print('num_spatial_orbitals:', problemOfInterest.num_spatial_orbitals)

expected_angularm0 = 0
expected_angularm2 = 2
expected_num_electrons = 4


# ---------------- spin sector filtering functions ----------------
def filter_criterion_custom_0(eigenstate, eigenvalue, aux_values):
    num_particles_aux = aux_values["ParticleNumber"][0]
    total_angular_momentum_aux = np.round(aux_values["AngularMomentum"][0], 10)
    return (np.isclose(total_angular_momentum_aux, expected_angularm0)
            and np.isclose(num_particles_aux, expected_num_electrons))


def filter_criterion_custom_1(eigenstate, eigenvalue, aux_values):
    num_particles_aux = aux_values["ParticleNumber"][0]
    total_angular_momentum_aux = np.round(aux_values["AngularMomentum"][0], 10)
    return (np.isclose(total_angular_momentum_aux, expected_angularm2)
            and np.isclose(num_particles_aux, expected_num_electrons))


# ---------------- classical diagonalisation ----------------
# key for everything below that depends only on the qubit Hamiltonian (not on lam)
_main_cache_path = _cache_path(molecule, _operator_key(qubit_op.simplify()),
                               type(mapper).__name__, problemOfInterest.num_particles)
_main_cache = _load_cache(_main_cache_path)

if _main_cache is not None:
    spin_0 = _main_cache['spin_0']
    spin_0_eigenvectors = list(_main_cache['spin_0_eigenvectors'])
    spin_1 = _main_cache['spin_1']
    spin_1_eigenvectors = list(_main_cache['spin_1_eigenvectors'])
else:
    algo = NumPyEigensolver(k=1000000)
    algo.filter_criterion = filter_criterion_custom_0
    result = ExcitedStatesEigensolver(mapper, algo).solve(problemOfInterest)
    spin_0 = result.eigenvalues
    spin_0_eigenvectors = [Statevector(vec[0]).data for vec in result.eigenstates]

    algo = NumPyEigensolver(k=1000000)
    algo.filter_criterion = filter_criterion_custom_1
    result = ExcitedStatesEigensolver(mapper, algo).solve(problemOfInterest)
    spin_1 = result.eigenvalues
    spin_1_eigenvectors = [Statevector(vec[0]).data for vec in result.eigenstates]
print('count singlets', len(spin_0))

mat = matrix_op
if _main_cache is not None:
    eigenValues, eigenVectors = _main_cache['eigenValues'], _main_cache['eigenVectors']
else:
    eigenValues, eigenVectors = np.linalg.eigh(mat)
_eigh_values_raw, _eigh_vectors_raw = eigenValues, eigenVectors  # unsorted, as returned by eigh (for the cache)

sortIdx = np.argsort(eigenValues)
eigenValues = eigenValues[sortIdx]
eigenVectors = eigenVectors[:, sortIdx]
print('eigenvalues near target:', eigenValues[max(targetIdx - 5, 0):targetIdx + 5])

targetEvec = eigenVectors[:, targetIdx]
targetEval = eigenValues[targetIdx]
print('targetEval:', np.round(targetEval, 4))
alpha = _main_cache['alpha'][()] if _main_cache is not None else np.linalg.norm(mat, ord=2)
targetEvec = targetEvec / np.linalg.norm(targetEvec)

lam = targetEval + 0.00100
full_eigs = _main_cache['full_eigs'] if _main_cache is not None else np.linalg.eigvalsh(mat)
if _main_cache is None:
    _save_cache(_main_cache_path,
                spin_0=np.asarray(spin_0), spin_0_eigenvectors=np.array(spin_0_eigenvectors),
                spin_1=np.asarray(spin_1), spin_1_eigenvectors=np.array(spin_1_eigenvectors),
                eigenValues=_eigh_values_raw, eigenVectors=_eigh_vectors_raw,
                alpha=np.asarray(alpha), full_eigs=full_eigs)
# scale = 1-norm of H
scale = np.linalg.norm(mat, ord=1)
delta_scaled = delta / scale
print('lam:', np.round(lam, 4))
minEne = 0

shifted_mat = mat + (minEne - lam) * np.eye(mat.shape[0])
renorm_shifted = shifted_mat / scale
print('renorm_shifted is Hermitian:', np.allclose(renorm_shifted, renorm_shifted.T.conj()))
# lam/scale-dependent, so it gets its own cache key (only the latest lam is kept on disk)
_rs_cache_path = _cache_path(molecule + '_rs', _main_cache_path.name, round(float(np.real(lam)), 10),
                             round(float(np.imag(lam)), 10), round(float(scale), 10))
_rs_cache = _load_cache(_rs_cache_path)
if _rs_cache is not None:
    rSeigenValues, rSeigenVectors = _rs_cache['rSeigenValues'], _rs_cache['rSeigenVectors']
else:
    rSeigenValues, rSeigenVectors = np.linalg.eig(renorm_shifted)
    _save_cache(_rs_cache_path, keep_only_latest=True, rSeigenValues=rSeigenValues, rSeigenVectors=rSeigenVectors)

sortIdx = np.argsort(rSeigenValues)
rSeigenValues = rSeigenValues[sortIdx]
rSeigenVectors = rSeigenVectors[:, sortIdx]
print('rSeigenVectors:', rSeigenVectors.shape)
