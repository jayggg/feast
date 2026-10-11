"""Contours: circle, ellipse and a custom rectangle on the same problem.

A Contour bundles quadrature nodes z, weights w and a membership test.
FEAST only sees the rational filter they induce,

    f(lambda) = sum_k w_k / (z_k - lambda),

which is close to 1 inside the contour and small outside. The filter
values at the eigenvalues show how well each contour separates the
wanted eigenvalues from the rest.

This demo compares three contour shapes targeting the same three eigenvalues,
teh first two of which are built in, and the last is custom.

* Contour.circle(center, radius, npts)

* Contour.ellipse(center, radius, rhoinv, npts): flatter for smaller
  rhoinv, useful when the wanted eigenvalues lie along the real axis.

* Contour.custom(z, w, within): any nodes and weights approximating
  (2 pi i)^(-1) times the contour integral. Here we make a rectangle with
  Gauss-Legendre points on each side.

The problem is the finite difference Laplacian of 02_modes.py.

Requires: numpy, scipy.
Run:      python demos/core/04_contours.py
"""

import logging
import sys

import numpy as np
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    stream=sys.stdout)

n = 200
h = 1.0 / (n + 1)
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h**2
k = np.arange(1, n + 1)
exact = 4 / h**2 * np.sin(k * np.pi * h / 2)**2  # ~ 9.9, 39.5, 88.8, 158


def rectangle(x0, x1, y0, y1, q=4):
    """Counterclockwise rectangle, q Gauss-Legendre points per side."""
    t, g = np.polynomial.legendre.leggauss(q)
    corners = [x0 + 1j * y0, x1 + 1j * y0, x1 + 1j * y1, x0 + 1j * y1]
    z, w = [], []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        z.append(a + (b - a) * (t + 1) / 2)
        w.append(g * (b - a) / 2 / (2j * np.pi))
    z, w = np.concatenate(z), np.concatenate(w)

    def within(lam):
        lam = np.asarray(lam)
        return ((x0 < lam.real) & (lam.real < x1)
                & (y0 < lam.imag) & (lam.imag < y1))

    return Contour.custom(z, w, within, name="rectangle")


contours = {
    "circle": Contour.circle(center=50.0, radius=45.0, npts=8),
    "ellipse": Contour.ellipse(center=50.0, radius=45.0, rhoinv=0.5, npts=8),
    "rectangle": rectangle(5.0, 100.0, -20.0, 20.0),
}

R = MatrixResolvent(A)
summaries = []
for name, C in contours.items():
    result = feast(R,
                   C,
                   NumpySpan.random(n, 5, seed=2),
                   mode="hermitian",
                   seed=2)
    lam = np.sort(result.eigenvalues.real)
    f = np.abs(C.filter(exact[:5]))
    summaries.append((name, C.npts, result.iterations, f))

    assert result.converged
    assert np.allclose(lam, exact[:3], rtol=1e-10)
    assert f[:3].min() > 10 * f[3:].max()  # inside clearly above outside

print("Contour comparison: |f(lambda_k)| for k = 1..5 (first 3 inside)")
for name, npts, iterations, f in summaries:
    values = np.array2string(f, precision=2)
    print(f"  {name}: npts={npts:2d} iterations={iterations}")
    print(f"            {values}")
