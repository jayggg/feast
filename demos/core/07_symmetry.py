"""Symmetry reduction: half the factorizations for real pencils.

When A and B are real and the quadrature nodes are symmetric about the
real axis, R(conj z) y = conj(R(z) conj(y)). The solve at a node's
mirror image then costs only a conjugation, so each conjugate pair of
nodes needs one factorization instead of two. feast(..., reduce_sym=True)
uses this. It requires mode="hermitian", a real pencil, and a contour
centered on the real axis (the circle and ellipse factories place
their nodes symmetrically).

The demo runs FEAST twice, without and with reduce_sym, and checks
that the eigenvalues agree. The saving is in the nodes: without the
reduction every one of the 16 nodes needs a sparse factorization; with
it only one node of each conjugate pair does, here the 8 nodes with
Im z > 0.

The pencil is the finite element pencil of 03_generalized_pencil.py.

Requires: numpy, scipy.
Run:      python demos/core/07_symmetry.py
"""

import numpy as np
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

n = 300
h = 1.0 / (n + 1)
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h
B = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n)) * h / 6
theta = np.arange(1, n + 1) * np.pi * h
exact = 6 / h**2 * (1 - np.cos(theta)) / (2 + np.cos(theta))

C = Contour.circle(center=100.0, radius=80.0, npts=16)

R = MatrixResolvent(A, B)
plain = feast(R, C, NumpySpan.random(n, 6, seed=7), mode="hermitian",
              seed=7)
reduced = feast(R, C, NumpySpan.random(n, 6, seed=7), mode="hermitian",
                seed=7, reduce_sym=True)

lam_plain = np.sort(plain.eigenvalues.real)
lam_reduced = np.sort(reduced.eigenvalues.real)
upper = C.z[C.z.imag > 0]      # one node of each conjugate pair

assert plain.converged and reduced.converged
assert np.allclose(lam_plain, exact[1:4], rtol=1e-10)
assert np.allclose(lam_reduced, lam_plain, rtol=1e-12)
assert len(upper) == C.npts // 2

print("Symmetry reduction")
print(f"  eigenvalues:     {lam_plain}")
print(f"  max difference:  {np.abs(lam_plain - lam_reduced).max():.1e}"
      "  (reduce_sym=True vs False)")
print(f"  factorizations:  {C.npts} without reduce_sym, {len(upper)} with it")
