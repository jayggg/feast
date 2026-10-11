"""Quadratic eigenproblem P(lambda) x = 0 with NumPy/SciPy matrices.

A damped vibration problem: find lambda and x with

    (K + lambda C + lambda**2 M) x = 0,

where K is the finite difference Laplacian on (0, 1) (unscaled, so its
eigenvalues are mu_k = 2 - 2 cos(k pi h)), M is the identity, and C = c M.
Each mu_k gives the pair of roots

    lambda = -c/2 ± sqrt(c**2/4 - mu_k),

real for the three lowest (overdamped) modes and a complex conjugate
pair for every other mode, all with real part -c/2 = -0.1.

MatrixPolyResolvent takes the coefficient list [K, C, M] in increasing
powers of lambda; any number of coefficients (degree >= 2), dense or
sparse, is allowed. FEAST runs on the companion linearization, whose
vectors have d * n entries, but only n x n matrices P(z) are factorized.
R.first extracts the right eigenvectors of P from the companion vectors,
and R.last the left eigenvectors. The method is that of

    J. Gopalakrishnan, B. Q. Parker and P. VandenBerge, "Computing
    leaky modes of optical fibers using a FEAST algorithm for
    polynomial eigenproblems," Wave Motion 108 (2022) 102826,
    https://doi.org/10.1016/j.wavemoti.2021.102826

The contour is a small circle around -0.1 + 0.48j. It encloses exactly
three eigenvalues, the + roots of modes k = 15, 16, 17. The FEAST log
(INFO level) shows the iteration on the companion pencil of size
2n = 200; Ritz values outside the circle are dropped at the end.

The final table has one row per computed eigenvalue:

* k: the mode number, i.e. the eigenvalue of K it comes from;

* lambda: the computed eigenvalue;

* error: |lambda - exact|, with exact = -c/2 + sqrt(c**2/4 - mu_k);

* right res.: |P(lambda) x| / |x| for the right eigenvector x;

* left res.: |P(lambda)^H y| / |y| for the left eigenvector y,
  i.e. y^H P(lambda) = 0.

The residuals check the eigenvectors without knowing them exactly;
values near machine precision (1e-16) mean x and y are accurate.
Here K, C and M are real symmetric, so P(lambda)^T = P(lambda) and y
is, up to scaling, conj(x); for a general P the two differ.

Requires: numpy, scipy.
Run:      python demos/core/09_polynomial.py
"""

import logging
import sys

import numpy as np
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixPolyResolvent

logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    stream=sys.stdout)

n = 100
c = 0.2
h = 1.0 / (n + 1)
K = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
Id = sp.identity(n, format="csc")
mu = 2 - 2 * np.cos(np.arange(1, n + 1) * np.pi * h)
# One root per mode (the + sign); the other roots are all outside C.
exact = -c / 2 + np.sqrt((c**2 / 4 - mu).astype(complex))

# Encloses the three eigenvalues in the upper half plane with imaginary
# part near 0.48; all others are outside.
C = Contour.circle(center=-0.1 + 0.48j, radius=0.045, npts=16)
R = MatrixPolyResolvent([K, c * Id, Id])
Y = R.random_span(6, seed=3)

result = feast(R, C, Y, mode="biorth", seed=3)

order = np.argsort(result.eigenvalues.imag)
lam = result.eigenvalues[order]
want = exact[C.within(exact)]
x = R.first(result.right).data  # right eigenvectors of P, n x 3
y = R.last(result.left).data  # left eigenvectors of P, n x 3
right_res = [np.linalg.norm(R.pol(lk) @ x[:, j]) / np.linalg.norm(x[:, j])
             for j, lk in enumerate(result.eigenvalues)]
left_res = [np.linalg.norm(R.pol(lk).conj().T @ y[:, j])
            / np.linalg.norm(y[:, j])
            for j, lk in enumerate(result.eigenvalues)]

assert result.converged
assert len(want) == 3
assert np.allclose(lam, np.sort_complex(want), atol=1e-12)
assert max(right_res) < 1e-10 and max(left_res) < 1e-10

# Mode number k of each computed eigenvalue: the nearest exact root.
modes = [int(np.argmin(np.abs(exact - lk))) + 1 for lk in lam]
right_res = np.array(right_res)[order]
left_res = np.array(left_res)[order]

print(f"Quadratic eigenproblem (K + lambda C + lambda^2 M) x = 0, n = {n}")
print(f"Eigenvalues inside the circle |z - ({C.center:.2f})| < {C.radius}:")
print(f"  {'k':>2}  {'lambda':19}  {'error':>7}  {'right res.':>10}"
      f"  {'left res.':>9}")
for k, lk, rr, lr in zip(modes, lam, right_res, left_res):
    err = abs(lk - exact[k - 1])
    print(f"  {k:2d}  {lk.real:+.6f}{lk.imag:+.6f}j  {err:7.1e}"
          f"  {rr:10.1e}  {lr:9.1e}")
