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
:func:`~feast.feast` call and the result are the same.

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

NGSolve
-------------

Dirichlet Laplacian
^^^^^^^^^^^^^^^^^^^^^^^

A finite element pencil on the unit square, including a degenerate
eigenvalue pair.

.. literalinclude:: ../demos/ngsolve/01_dirichlet_square.py
   :language: python
