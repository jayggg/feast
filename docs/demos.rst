Demos
===========

Small, complete scripts showing typical use. Each one runs in seconds
and ends with assertions against known answers; the test suite runs all
of them, so the code below is checked on every push. The scripts live
in ``demos/`` in the repository and can be run directly, for example::

    python demos/core/01_quickstart.py

Demos under ``demos/core`` need only NumPy and SciPy. Demos under
``demos/ngsolve`` need NGSolve. Backend demos differ from the core ones
only in how the resolvent and starting span are built; the
:func:`~feast.driver.feast` call and the result are the same.

Core
----------

Quick start
^^^^^^^^^^^^^^^

Eigenvalues of a small diagonal matrix inside a circle, with left and
right eigenvectors.

.. literalinclude:: ../demos/core/01_quickstart.py
   :language: python

Iteration modes
^^^^^^^^^^^^^^^^^^^

The same sparse pencil solved with ``mode="hermitian"``, ``"right"`` and
``"biorth"``, checked against exact finite difference eigenvalues. One
resolvent is reused, so its factorizations are computed only once.

.. literalinclude:: ../demos/core/02_modes.py
   :language: python

Generalized pencil
^^^^^^^^^^^^^^^^^^^^^^

A finite element pencil ``A x = λ B x`` with a mass matrix B. The
eigenvectors are B-orthonormal; ``R.gram`` and ``R.residual`` check the
result.

.. literalinclude:: ../demos/core/03_generalized_pencil.py
   :language: python

Contours
^^^^^^^^^^^^

A circle, an ellipse, and a custom rectangle with Gauss-Legendre
points, compared through the values of their rational filters at the
eigenvalues.

.. literalinclude:: ../demos/core/04_contours.py
   :language: python

Diagnostics
^^^^^^^^^^^^^^^

Controlling FEAST's logging, reading ``result.history``, checking
residuals and contour membership, and what a nonconverged result
contains.

.. literalinclude:: ../demos/core/05_diagnostics.py
   :language: python

Applying the projector
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~feast.SpectralProjectorApprox` applied without the FEAST
iteration, and its relation to the filter: ``P = V f(Λ) V⁻¹``. On a
non-normal matrix, ``apply`` (right vectors) and ``apply_adj`` (left
vectors, ``Pᴴ``) give different results.

.. literalinclude:: ../demos/core/06_projector.py
   :language: python

Symmetry reduction
^^^^^^^^^^^^^^^^^^^^^^

``reduce_sym=True`` for a real pencil: the same eigenvalues with half
the factorizations.

.. literalinclude:: ../demos/core/07_symmetry.py
   :language: python

Splitting a contour
^^^^^^^^^^^^^^^^^^^^^^^

One large search followed by :func:`~feast.zoom.simple_multiple_zoom`
and a separate run for each isolated eigenvalue and each cluster.

.. literalinclude:: ../demos/core/08_zoom.py
   :language: python

Polynomial eigenproblem
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A damped quadratic eigenproblem ``(K + λ C + λ² M) x = 0`` with sparse
SciPy coefficients, solved with
:class:`~feast.backends.numpy_backend.MatrixPolyResolvent` by the
method of [GPV2022]_. Right and
left eigenvectors of the polynomial are extracted from the companion
vectors.

.. literalinclude:: ../demos/core/09_polynomial.py
   :language: python

NGSolve
-------------

Dirichlet Laplacian
^^^^^^^^^^^^^^^^^^^^^^^

A finite element pencil on the unit square, including a degenerate
eigenvalue pair.

.. literalinclude:: ../demos/ngsolve/01_dirichlet_square.py
   :language: python
