Getting started
=====================

Installation
------------------

Use Python 3.10 or later. From the repository root::

    python -m pip install -e .

For finite element computations, use an environment in which NGSolve
already imports successfully, or install the optional dependency::

    python -m pip install -e '.[ngsolve]'

The PyPI distribution named ``feast`` is an unrelated project. Do not
install it as a substitute for this source checkout.

A matrix eigenproblem
---------------------------

This example finds the two eigenvalues near 3.5. Matrices may be dense
NumPy arrays or SciPy sparse matrices. Omitting B uses the identity.

.. code-block:: python

   import numpy as np
   from feast import Contour, feast
   from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

   A = np.diag([-1.0, 1.0, -1j, 1j, 3.0, 4.0])
   R = MatrixResolvent(A)
   C = Contour.circle(center=3.5, radius=0.8, npts=8)
   Y = NumpySpan.random(A.shape[0], 4, seed=10)

   result = feast(R, C, Y, mode="biorth", seed=10)
   print(np.sort_complex(result.eigenvalues))
   print(result.converged)

``result.right`` contains the right vectors; ``result.left`` contains
the left vectors in biorthogonal mode. NumPy spans store vectors as
columns of their ``data`` array. Eigenvalues are not guaranteed to be
sorted. If you sort them, apply the same permutation to both spans.

Choose an iteration mode
------------------------------

* ``mode="hermitian"`` uses a Hermitian reduced eigensolve. The reduced
  B matrix must be positive definite.
* ``mode="right"`` computes right vectors only for a general pencil.
* ``mode="biorth"`` computes left and right vectors and removes nearly
  singular directions using the B-Gram matrix. An independent random
  left span is generated unless ``yl`` is supplied.

Hermitian and right-only modes remove dependent filtered directions using
QR and a small SVD. ``rank_tol=1e-12`` is the default relative threshold
on basis singular values; it differs from ``eta_tol``, which acts on the
biorthogonal B-Gram matrix. ``expected_dim`` requests replenishment if the
basis becomes too small. If that size cannot be recovered within the
refresh and restart limits, ``result.converged`` is false and the result
contains the last computed eigenvalue/vector pairs.

For reproducibility, seed both the initial span and the driver. To retain
an initial span for another run, pass ``Y.copy()``.

Diagnostics
-----------------

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
   result = feast(
       R, C, Y.copy(), mode="biorth", seed=10,
       report_residuals=True,
   )
   residuals = R.residual(result.eigenvalues, result.right, result.left)
   inside = C.within(result.eigenvalues)

The convergence flag measures stability of the Ritz values, not residual
accuracy or completeness. Inspect residuals, contour membership, and span
dimensions. See :doc:`limitations` for currently known failure cases.
History entries are recorded before contour trimming, so they need not
have the same length as the final eigenvalue array.

Contours and direct filtering
-----------------------------------

.. code-block:: python

   from feast import SpectralProjectorApprox

   ellipse = Contour.ellipse(
       center=3.5, radius=0.8, rhoinv=0.5, npts=16,
   )
   P = SpectralProjectorApprox(R, ellipse)
   filtered = P.apply(Y)
   filter_values = ellipse.filter(np.array([3.0, 4.0, 8.0]))

Circle rules are ``"trapezoid_shifted"``, ``"trapezoid"``, and
``"snug"``. Ellipse rules are ``"trapezoid_shifted"`` and ``"snug"``.
Use ``Contour.custom(z, w, within)`` for supplied nodes, weights, and a
membership function. Keep eigenvalues away from the contour itself.
