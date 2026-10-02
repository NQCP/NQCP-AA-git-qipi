"""
Disk cache for the expensive diagonalisations of the chemical simulations (figures 13 and 14).

The cache key is built from the mapped qubit operator itself (and lam/scale where they matter),
so changing the molecule, geometry, basis, mapper or lam never reuses stale results.
Set USE_HAMILTONIAN_CACHE = False to always recompute; delete the data/ folder to free disk space.
"""

import hashlib
from pathlib import Path

import numpy as np

USE_HAMILTONIAN_CACHE = True
CACHE_DIR = Path(__file__).resolve().parent / 'data'


def _cache_path(tag, *key_parts):
    key = hashlib.md5(repr(key_parts).encode()).hexdigest()[:12]
    return CACHE_DIR / f'hamiltonian_{tag}_{key}.npz'


def _operator_key(op):
    # the mapper returns the Pauli terms in a different order on each run, so sort them,
    # and round the coefficients so last-digit floating-point noise doesn't change the key
    terms = sorted(zip(op.paulis.to_labels(), np.round(np.asarray(op.coeffs), 10).tolist()))
    return hashlib.md5(repr(terms).encode()).hexdigest()


def _load_cache(path):
    if USE_HAMILTONIAN_CACHE and path.exists():
        print('loading cached', path.name)
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    return None


def _save_cache(path, keep_only_latest=False, **arrays):
    if not USE_HAMILTONIAN_CACHE:
        return
    CACHE_DIR.mkdir(exist_ok=True)
    if keep_only_latest:  # lam-dependent caches: drop older files with the same tag
        for old in CACHE_DIR.glob(path.name.rsplit('_', 1)[0] + '_*.npz'):
            old.unlink()
    np.savez(path, **arrays)
    print('saved cache', path.name)
