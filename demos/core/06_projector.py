"""Applying the approximate spectral projector directly.

SpectralProjectorApprox(R, C) applies the quadrature sum at the heart
of FEAST without running the iteration:

    P y = sum_k w_k R(z_k) y,    R(z) = (z B - A)^(-1) B.

For a diagonalizable A = V diag(lambda) V^(-1) with B = I this is
exactly

    P = V diag(f(lambda)) V^(-1),

where f = C.filter is the rational filter of the contour. So P is not
the exact spectral projector; it approaches it as npts grows.

P.apply filters right vectors: it damps each right eigenvector (a
column of V) by its filter value f(lambda).

P.apply_adj is the left counterpart, used by mode="biorth" to filter
left eigenvectors (y^H A = lambda y^H B). It applies

    sum_k conj(w_k) (z_k B - A)^(-H) B^H,

which for B = I is exactly the conjugate transpose
P^H = V^(-H) diag(conj f(lambda)) V^H: it damps each left
eigenvector (a column of V^(-H)) by the same filter values.

The demo builds a non-normal matrix with known eigenvectors (V is not
orthogonal), so left and right eigenvectors differ and the two calls,
P.apply and P.apply_adj, give different results. Both are applied to
the same Y: a fixed block of three random vectors (a NumpySpan, like a
starting span for feast), which has components along every
eigenvector, inside and outside the contour. The demo checks both
identities, then prints a table with one row per npts:

* distance: ||P - E||_2, with E = V[:, :3] V^(-1)[:3, :] the exact
  (oblique) spectral projector onto the eigenvalues 1, 1.5, 2;

* P Y off right: the relative part of P Y outside the span of the
  three right eigenvectors V[:, :3];

* P^H Y off left: the relative part of P^H Y outside the span of the
  three left eigenvectors, the first three columns of V^(-H).

All three columns should shrink together as npts grows: P pushes
vectors into the right eigenspace, and P^H, its adjoint, into the left
one. The leftover is the filter's leakage from the eigenvalues outside
the contour, f(lambda) for lambda = 4, ..., 20, which more nodes
suppress. A last line shows the contrast: P^H Y lies far from the
right eigenspace, and P Y far from the left one, because the two
eigenspaces differ for a non-normal A.

Note: For a symmetric A, V would be orthogonal, P^H would equal P,
the two calls would coincide, and then typically the distance to
the exact projector decays more rapidly.

Requires: numpy, scipy.
Run:      python demos/core/06_projector.py

"""

import numpy as np

from feast import Contour, SpectralProjectorApprox
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

rng = np.random.default_rng(6)
n = 40
lam = np.concatenate([[1.0, 1.5, 2.0], np.linspace(4.0, 20.0, n - 3)])
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
V = Q @ (np.eye(n) + 0.3 * np.triu(rng.standard_normal((n, n)), 1))
Vinv = np.linalg.inv(V)  # rows of Vinv: left eigenvectors^H
A = V @ np.diag(lam) @ Vinv  # non-normal: V is not orthogonal
R = MatrixResolvent(A)
Y = NumpySpan.random(n, 3, seed=6)  # three random test vectors

# Oblique spectral projector onto the eigenvalues 1, 1.5, 2, and
# orthonormal bases of the corresponding right and left eigenspaces.
exact_projector = V[:, :3] @ Vinv[:3, :]
right_basis, _ = np.linalg.qr(V[:, :3])
left_basis, _ = np.linalg.qr(Vinv[:3, :].conj().T)


def off(x, basis):
    """Relative norm of the part of x outside span(basis)."""
    return np.linalg.norm(x - basis @ (basis.conj().T @ x)) / np.linalg.norm(x)


errors = []
for npts in (4, 8, 16, 32):
    C = Contour.circle(center=1.5, radius=1.5, npts=npts)
    P = SpectralProjectorApprox(R, C)
    PY = P.apply(Y)
    PhY = P.apply_adj(Y)

    filtered = V @ np.diag(C.filter(lam)) @ Vinv  # V f(Lambda) V^(-1)
    scale = np.linalg.norm(filtered @ Y.data)
    assert np.allclose(PY.data, filtered @ Y.data, atol=1e-10 * scale)
    # apply_adj is P^H, which differs from P because A is non-normal
    assert np.allclose(PhY.data,
                       filtered.conj().T @ Y.data,
                       atol=1e-10 * scale)
    assert not np.allclose(PY.data, PhY.data, atol=1e-3 * scale)

    error = np.linalg.norm(filtered - exact_projector, 2)
    errors.append((npts, error, off(PY.data,
                                    right_basis), off(PhY.data, left_basis)))

for column in (1, 2, 3):  # every column decreases with npts
    values = [row[column] for row in errors]
    assert all(a > b for a, b in zip(values, values[1:]))
assert max(errors[-1][1:]) < 1e-6
# Each call lands in its own eigenspace, not the other one.
assert off(PhY.data, right_basis) > 0.1
assert off(PY.data, left_basis) > 0.1

print(f"Non-normal A, cond(V) = {np.linalg.cond(V):.0f}. All three columns"
      " should shrink as npts grows:")
print("P approaches the exact projector; for Y = 3 random vectors, P Y")
print("approaches the right eigenspace and P^H Y the left eigenspace.")
print("  npts  distance  P Y off right  P^H Y off left")
for npts, error, off_right, off_left in errors:
    print(f"  {npts:4d}  {error:8.1e}  {off_right:13.1e}  {off_left:14.1e}")
print("For contrast at npts=32: P^H Y off right "
      f"{off(PhY.data, right_basis):.2f}, P Y off left "
      f"{off(PY.data, left_basis):.2f}")
