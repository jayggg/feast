"""Iteration modes: hermitian, right and biorth on the same pencil.

A = tridiag(-1, 2, -1) / h**2 is the finite difference Dirichlet
Laplacian on (0, 1), stored as a SciPy sparse matrix. Its eigenvalues
are known exactly, so all three modes can be checked against them.

* mode="hermitian":2 Rayleigh-Ritz for A x = lambda B x, with A and B
  Hermitian and B positive definite on the search span.
* mode="right": right eigenvectors only; works for any pencil.
* mode="biorth": left and right eigenvectors with SVD kernel cleaning;
  a more robust choice for non-normal problems.

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

summaries = []
for mode in ("hermitian", "right", "biorth"):
    result = feast(R, C, Y0.copy(), mode=mode, seed=1)
    lam = np.sort(result.eigenvalues.real)
    summaries.append((mode, lam, result.iterations))

    assert result.converged
    assert np.allclose(lam, exact[:3], rtol=1e-10)
    assert (result.left is None) == (mode != "biorth")

print("Mode comparison")
for mode, lam, iterations in summaries:
    print(f"  {mode:>9}: {lam}  iterations={iterations}")
