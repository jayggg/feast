"""Splitting a contour: from one large search to one contour per group.

A first FEAST run with a large contour finds all eigenvalues in a real
interval. simple_multiple_zoom then proposes smaller circles: one per
isolated eigenvalue and one per cluster of nearby eigenvalues. Running
FEAST on each small circle computes every group separately, for
example to refine them independently or in parallel.

Clustering is by distance only (values closer than delta are grouped;
by default delta is a tenth of the average spacing). Each proposed
circle's radius is half the gap to the nearest value outside its
group, so check that every circle encloses what it should before
using it, as done below.

The matrix is Hermitian with known eigenvalues, two of them in tight
pairs: 2 and 2.0005, and 4.2 and 4.2002.

Requires: numpy, scipy.
Run:      python demos/core/08_zoom.py
"""

import logging
import sys

import numpy as np

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan
from feast.zoom import simple_multiple_zoom

logging.basicConfig(level=logging.INFO,
                    format="%(message)s",
                    stream=sys.stdout)
logging.getLogger("feast").setLevel(logging.WARNING)  # many runs: quiet

rng = np.random.default_rng(8)
wanted = np.array([1.0, 2.0, 2.0005, 3.0, 4.2, 4.2002, 5.5])
others = np.linspace(8.0, 30.0, 53)
n = len(wanted) + len(others)
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
A = V @ np.diag(np.concatenate([wanted, others])) @ V.T
R = MatrixResolvent(A)

# 1. One large contour over the interval (c1, c2).
c1, c2 = 0.5, 6.0
big = Contour.circle(center=(c1 + c2) / 2, radius=(c2 - c1) / 2, npts=16)
first = feast(R,
              big,
              NumpySpan.random(n, 10, seed=8),
              mode="hermitian",
              seed=8)
found = first.eigenvalues.real
assert first.converged
assert np.allclose(np.sort(found), wanted, rtol=1e-10)

# 2. Proposed contours: lists of indices into found, centers and radii.
simple, multiple = simple_multiple_zoom(c1, c2, found)
groups = [
    ([i], c, r)
    for i, c, r in zip(simple["index"], simple["center"], simple["radius"])
]
groups += [(list(idx), c, r) for idx, c, r in zip(
    multiple["index"], multiple["center"], multiple["radius"])]

# 3. Check each circle, then run FEAST on it.
print("Groups and their contours")
for idx, center, radius in groups:
    C = Contour.circle(center=center, radius=radius, npts=8)
    inside = np.flatnonzero(C.within(found))
    assert sorted(inside) == sorted(idx), "contour misses its group"

    m = len(idx) + 2  # a little extra room

    result = feast(R,
                   C,
                   NumpySpan.random(n, m, seed=8),
                   mode="hermitian",
                   seed=8)

    assert result.converged
    assert np.allclose(np.sort(result.eigenvalues.real),
                       np.sort(found[idx]),
                       rtol=1e-10)
    print(f"  center {center:8.5f}  radius {radius:.2e}: "
          f"{np.sort(result.eigenvalues.real)}")

assert len(groups) == 5  # three simple values and two pairs
