"""Quick start: eigenvalues of a small matrix inside a circle.

The diagonal matrix has eigenvalues -1, 1, -i, i, 3, 4. A circle of
radius 0.8 about 3.5 encloses exactly 3 and 4, which FEAST finds
together with their left and right eigenvectors.

Requires: numpy, scipy.
Run:      python demos/core/01_quickstart.py
"""

import logging

import numpy as np

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

# use logging.WARN to suppress FEAST output, logging.INFO to see it

logging.basicConfig(level=logging.INFO, format="%(message)s")

A = np.diag([-1.0, 1.0, -1j, 1j, 3.0, 4.0])
R = MatrixResolvent(A)  # B omitted: standard problem A x = lam x
C = Contour.circle(center=3.5, radius=0.8, npts=8)
Y = NumpySpan.random(A.shape[0], 4, seed=10)  # 4 >= expected count

result = feast(R, C, Y, mode="biorth", seed=10, report_residuals=True)

assert result.converged
assert np.allclose(np.sort_complex(result.eigenvalues), [3.0, 4.0])
