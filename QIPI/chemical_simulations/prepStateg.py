"""Spin-adapted (singlet / triplet) CSF state preparation in the qubit space of a given mapper.

Copied unchanged from the original prepStateg.py; used by figure 13 and figure 14.
"""


import numpy as np
import itertools
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from numpy.random import Generator, PCG64

import sys
sys.path.append("py-files")

from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
 
def build_mapper(mapper_name: str, problem) -> object:
    if isinstance(mapper_name, str):
        if mapper_name == "parity":
            if problem.num_spin_orbitals <= 4:
                mapper = ParityMapper()
            else:
                mapper = ParityMapper(num_particles=problem.num_particles)
        elif mapper_name == "jw":
            mapper = JordanWignerMapper()
        else:
            raise ValueError(f"Unknown mapper: {mapper_name}")
        return mapper


def _determinant_reduced_state(occ_indices, mapper, n_spin, n_qubits):
    """
    Prepare the qubit statevector of a single Slater determinant (a product of
    creation operators on the vacuum) in WHATEVER space `mapper` produces,
    including a two-qubit-reduced ParityMapper.

    This uses the same recipe qiskit-nature uses for HartreeFock initial states:
    a single determinant maps to a single computational basis state, whose index
    is read off from the X-part of the mapped Pauli operator. The mapped
    coefficient carries the fermionic sign/phase.

    Returns a length-2**n_qubits complex vector (a basis state, possibly phased).
    """
    label = " ".join(f"+_{i}" for i in occ_indices)
    qop = mapper.map(FermionicOp({label: 1.0}, num_spin_orbitals=n_spin))

    xbits = qop.paulis.x[0]
    idx = 0
    for q, bit in enumerate(xbits):
        if bit:
            idx |= (1 << q)

    vec = np.zeros(2 ** n_qubits, dtype=complex)
    phase = qop.coeffs[0]
    vec[idx] = phase
    return vec


def _csf_reduced_state(determinants, mapper, n_spin, n_qubits):
    """
    Build a configuration state function (CSF) as a coefficient-weighted sum of
    determinants, each prepared in the (possibly reduced) qubit space.

    determinants : list of (coeff, [occupied spin-orbital indices])
    Returns the normalized statevector, or None if it vanishes.
    """
    vec = np.zeros(2 ** n_qubits, dtype=complex)
    for coeff, occ in determinants:
        vec += coeff * _determinant_reduced_state(occ, mapper, n_spin, n_qubits)
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        return None
    return vec / nrm


def stateprep_states(problem, mapper_name, n_qubits):
    """
    Build singlet (S=0) and triplet (S=1) CSF STATEVECTORS directly in the
    mapper's qubit space (works for full JW, full parity, and reduced parity).

    Returns (singlet_states, triplet_states) as lists of complex vectors.
    """
    mapper = build_mapper(mapper_name, problem)

    n_spatial = problem.num_spatial_orbitals
    n_spin = problem.num_spin_orbitals
    n_particles = sum(problem.num_particles)

    singlet_states = []
    triplet_states = []

    remaining = n_particles - 2
    if remaining < 0 or remaining % 2 != 0:
        return [], []
    k = remaining // 2  # number of doubly-occupied inactive spatial pairs

    s = 1.0 / np.sqrt(2.0)

    for p in range(n_spatial):
        for q in range(p, n_spatial):
            pa, pb = p, p + n_spatial
            qa, qb = q, q + n_spatial

            occupied_spatial = {p, q}
            inactive_pairs = [
                (r, r + n_spatial)
                for r in range(n_spatial)
                if r not in occupied_spatial
            ]
            if len(inactive_pairs) < k:
                continue

            for combo in itertools.combinations(inactive_pairs, k):
                inactive_occ = [x for pair in combo for x in pair]

                if p == q:
                    # closed shell: doubly-occupied spatial orbital -> singlet
                    occ = [pa, pb] + inactive_occ
                    st = _csf_reduced_state([(1.0, occ)], mapper, n_spin, n_qubits)
                    if st is not None:
                        singlet_states.append(st)
                    continue

                # open-shell determinants on spatial p, q:
                detA = [pa, qb] + inactive_occ   # p_alpha q_beta
                detB = [qa, pb] + inactive_occ   # q_alpha p_beta

                # singlet (S=0):     (A + B)/sqrt2
                # triplet (M_S=0):   (A - B)/sqrt2
                sing = _csf_reduced_state(
                    [(s, detA), (s, detB)], mapper, n_spin, n_qubits
                )
                trip0 = _csf_reduced_state(
                    [(s, detA), (-s, detB)], mapper, n_spin, n_qubits
                )
                if sing is not None:
                    singlet_states.append(sing)
                if trip0 is not None:
                    triplet_states.append(trip0)

                # NOTE: we do NOT add the M_S=+1 (both alpha) / M_S=-1 (both
                # beta) SINGLE determinants. When there are other occupied
                # (inactive) orbitals, a single same-spin determinant is not an
                # S^2 eigenstate -- it mixes S=1 with higher multiplicities and
                # shows up as S^2 = 1 (etc.), which is the spin-sector leakage.
                # The M_S=0 antisymmetric combination above already seeds the
                # triplet sector cleanly (S^2 = 2), so we rely on that alone.

    return singlet_states, triplet_states


def is_duplicate_state(state, accepted, tol=1e-8):
    for ref in accepted:
        overlap = np.abs(np.vdot(ref, state))
        if overlap > 1 - tol:
            return True
    return False


def get_guess_vectors(
    method,
    p,
    hamiltonian,
    problem=None,
    qubit_hamiltonian=None,
    energy_center=None,
    target_s2="singlet",
    mapper_name=None,
    seed=None,
):
    n = hamiltonian.shape[0]

    if method == "random":
        rng = Generator(PCG64(seed=seed))
        A = rng.random(size=(n, p))
        Q, _ = np.linalg.qr(A)
        return Q

    elif method == "s2":
        # number of qubits implied by the (already-mapped) Hamiltonian
        n_qubits = int(round(np.log2(n)))
        if 2 ** n_qubits != n:
            raise ValueError(f"hamiltonian dimension {n} is not a power of two.")

        singlet_states, triplet_states = stateprep_states(
            problem, mapper_name, n_qubits
        )

        if target_s2 == "singlet":
            target_states = singlet_states
        elif target_s2 == "triplet":
            target_states = triplet_states
        else:
            raise ValueError("target_s2 must be either 'singlet' or 'triplet'")

        if len(target_states) == 0:
            raise RuntimeError(
                f"No '{target_s2}' states produced. Check the active space."
            )

        # states already live in the SAME space as `hamiltonian`, so the energy
        # ranking is well defined regardless of parity reduction.
        distance_to_center = []
        for state in target_states:
            e = np.vdot(state, hamiltonian @ state)
            distance_to_center.append(np.abs(e - energy_center))

        order = sorted(range(len(target_states)), key=lambda i: distance_to_center[i])

        unique_states = []
        for i in order:
            state = target_states[i]
            if not is_duplicate_state(state, unique_states):
                unique_states.append(state)
            if len(unique_states) == p:
                break

        if len(unique_states) < p:
            raise RuntimeError(
                f"Only found {len(unique_states)} unique '{target_s2}' states "
                f"but {p} were requested."
            )

        return np.column_stack(unique_states)

    else:
        raise ValueError("method has to be either 'random' or 's2'")