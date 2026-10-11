"""Iteration modes: hermitian, right and biorth on the same pencil.

A = tridiag(-1, 2, -1) / h**2 is the finite difference Dirichlet
Laplacian on (0, 1), stored as a SciPy sparse matrix. Its eigenvalues
are known exactly, so all three modes can be checked against them.

* mode="hermitian": Rayleigh-Ritz for A x = lambda B x, with A and B
  Hermitian and B positive definite on the search span.
  Uses scipy.linalg.eigh(a, b) which assumes the reduced pencil is Hermitian
  and that b positive definite.

* mode="right": right eigenvectors only; works for any pencil.
  Uses scipy.linalg.eig(a, b) which assumes nothing about the reduced
  pencil.

* mode="biorth": left and right eigenvectors with SVD kernel cleaning;
  a more robust choice for non-normal problems.

Each iteration first discards directions the filter has nearly
annihilated, using singular values relative to the largest:

* hermitian and right: singular values of the filtered basis, cut at
  rank_tol (result.history.basis_singular_values);

* biorth: singular values of the B-Gram matrix of the left and right
  filtered bases, cut at eta_tol (result.history.gram_singular_values).
  These decay roughly like squares of the basis values, hence the
  larger tolerance.

result.history.retained_ranks records (filtered, kept) for each step.

In this example the hermitian and right modes are mathematically
equivalent, so only round off errors can distinguish them.

Requires: numpy, scipy.
Run:      python demos/core/02_modes.py
"""

import numpy as np
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

n = 200
h = 1.0 / (n + 1)
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h**2
k = np.arange(1, n + 1)
exact = 4 / h**2 * np.sin(k * np.pi * h / 2)**2  # ~ (k pi)^2

# Encloses the three smallest eigenvalues (~9.9, 39.5, 88.8),
# excludes the fourth (~157.9).
C = Contour.circle(center=50.0, radius=45.0, npts=8)
R = MatrixResolvent(A)  # factorizations are cached and reused below
Y0 = NumpySpan.random(n, 5, seed=1)

RANK_TOL = 1e-12  # the defaults, passed explicitly to show them
ETA_TOL = 1e-8


def relative(sv):
    return np.array2string(sv / sv[0],
                           formatter={"float_kind": lambda x: f"{x:.1e}"})


summaries = []
for mode in ("hermitian", "right", "biorth"):
    result = feast(R,
                   C,
                   Y0.copy(),
                   mode=mode,
                   seed=1,
                   rank_tol=RANK_TOL,
                   eta_tol=ETA_TOL)
    lam = np.sort(result.eigenvalues.real)
    summaries.append((mode, lam, result))

    assert result.converged
    assert np.allclose(lam, exact[:3], rtol=1e-10)
    assert (result.left is None) == (mode != "biorth")

print("Solver mode comparison")
for mode, lam, result in summaries:
    print(f"  {mode:>9}: {lam}  iterations={result.iterations}")

print()
print("Rank decisions: singular values relative to the largest")
for mode, _, result in summaries:
    hist = result.history
    if mode == "biorth":
        sv_list, name, tol = hist.gram_singular_values, "eta_tol", ETA_TOL
    else:
        sv_list, name, tol = hist.basis_singular_values, "rank_tol", RANK_TOL
    print(f"  {mode} ({name} = {tol:.0e})")
    for sv, (filtered, kept) in zip(sv_list, hist.retained_ranks):
        print(f"    {relative(sv)}  kept {kept} of {filtered}")
