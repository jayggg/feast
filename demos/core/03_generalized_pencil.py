"""Generalized eigenproblem A x = lambda B x with a mass matrix.

Linear finite elements for -u'' = lambda u on (0, 1), u(0) = u(1) = 0,
on a uniform mesh give a stiffness matrix A and a mass matrix B, both
sparse, symmetric and tridiagonal. The discrete eigenvalues are known
exactly:

    lambda_k = (6 / h**2) (1 - cos(k pi h)) / (2 + cos(k pi h)).

Pass B as the second argument of MatrixResolvent. In mode="hermitian"
the returned eigenvectors are B-orthonormal, R.gram computes the
B-inner products that confirm it, and  R.residual returns the Euclidean
ℓ² norm |.| of the residual |A x - lambda B x| of the computed pairs.

Requires: numpy, scipy.
Run:      python demos/core/03_generalized_pencil.py
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

n = 300
h = 1.0 / (n + 1)
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h
B = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n)) * h / 6
theta = np.arange(1, n + 1) * np.pi * h
exact = 6 / h**2 * (1 - np.cos(theta)) / (2 + np.cos(theta))

# Encloses lambda_2, lambda_3, lambda_4 (~39.5, 88.8, 157.9); excludes
# lambda_1 (~9.9) and lambda_5 (~246.8).
C = Contour.circle(center=100.0, radius=80.0, npts=8)
R = MatrixResolvent(A, B)
Y = NumpySpan.random(n, 6, seed=3)

result = feast(R, C, Y, mode="hermitian", seed=3)

order = np.argsort(result.eigenvalues.real)
lam = result.eigenvalues.real[order]
gram = R.gram(result.right, result.right)  # X^H B X
residuals = R.residual(result.eigenvalues, result.right)

assert result.converged
assert np.allclose(lam, exact[1:4], rtol=1e-10)
assert np.allclose(gram, np.eye(3), atol=1e-10)
assert np.all(residuals < 1e-8 * np.abs(result.eigenvalues))

print("Generalized pencil")
print("  computed:", lam)
print("  exact:   ", exact[1:4])
print("  max |X^H B X - I|:", np.abs(gram - np.eye(3)).max())
print("  residuals:", residuals[order])
