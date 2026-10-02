import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

from nlft_qsp.qsp import chebqsp_solve, xqsp_solve
from nlft_qsp.poly import Polynomial


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from QIPI.hamiltonian import build as build_hamiltonian

warnings.filterwarnings('ignore')

# Hamiltonian builder within the QIPI package.
delta_scaled = build_hamiltonian("H2")["delta_scaled"]
shift_factor = 0.00001
coef_list = []


def eval_chebyt(n, x):
    """Evaluate Chebyshev polynomial of the first kind at x."""
    if n < 0:
        raise ValueError("n must be non-negative")
    elif n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T0 = np.ones_like(x)
        T1 = x
        for k in range(2, n + 1):
            T2 = 2 * x * T1 - T0
            T0, T1 = T1, T2
        return T2


def optpoly_eigenstatefiltering(x, l, Delta):
    arg_top = -1 + 2 * ((x**2 - Delta**2) / (1 - Delta**2))
    arg_bot = -1 + 2 * ((-Delta**2) / (1 - Delta**2))
    cheb_l = scipy.special.chebyt(l)
    return cheb_l(arg_top) / cheb_l(arg_bot)


vpoly = np.vectorize(optpoly_eigenstatefiltering)


def eigenvalue_filtering_poly(M, a=delta_scaled, plot=False):
    """Find Chebyshev coefficients of eigenvalue filtering polynomial."""

    def r(x):
        return (
            eval_chebyt(M, (2 * (x) ** 2 - (1 + a**2)) / (1 - a**2))
            / eval_chebyt(M, -(1 + a**2) / (1 - a**2))
            - shift_factor
        )

    if plot:
        xs = np.linspace(-1, 1, 500)
        ys = r(xs)
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys)
        plt.title(f"Eigenvalue filtering polynomial, M={M}, a={a}")
        plt.xlabel("$x$")
        plt.ylabel("$r_M(x, a)$")
        plt.show()

    coef = np.polynomial.chebyshev.Chebyshev.interpolate(r, 2 * M).coef

    complex_coef = coef.copy()
    complex_coef = complex_coef[::2]
    complex_coef[0] *= 2
    complex_coef = 1 / 2 * np.concatenate([complex_coef[::-1][:-1], complex_coef])

    return coef, r


def get_phase_factors(M, a=delta_scaled, plot=False):
    """Get phase factors for a Chebyshev QSP protocol."""
    cheb_coeffs, r = eigenvalue_filtering_poly(M, a, plot)
    phase_factors = chebqsp_solve(list(cheb_coeffs))
    return phase_factors.phi


def convert_to_phi_prime(phis):
    """
    Convert a list of phi values [phi_0, ..., phi_d] to phi_prime values.
    """
    d = len(phis) - 1
    phi_primes = [0] * (d + 1)

    phi_primes[0] = phis[0] + (2 * d - 1) * np.pi / 4
    phi_primes[d] = phis[d] - np.pi / 4
    for k in range(1, d):
        phi_primes[k] = phis[k] - np.pi / 2

    return phi_primes

def main():
    M = 16
    a = 0.05

    phase_factors = get_phase_factors(M, a, plot=True)
    print('len(phase_factors)', len(phase_factors))
    print("\nPhase Factors:")
    print(f"Phi values: {phase_factors}")


if __name__ == "__main__":
    main()
