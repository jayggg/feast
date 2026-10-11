feast: contour-integral eigensolvers
==========================================

``feast`` computes selected eigenvalues and eigenspaces of a matrix pencil
``A x = λ B x`` using contour integration and subspace iteration. The
distribution name is ``feast-eig``; Python imports use ``feast``.

The package separates the search contour, discretization backend, and
iteration. It ships with two backends: a NumPy/SciPy backend, which
works with a default install, and an NGSolve backend for finite element
matrices, which requires NGSolve to be installed separately. Both handle
linear pencils and polynomial eigenproblems
``(A0 + λ A1 + ⋯ + λᵈ Ad) x = 0``.

.. toctree::
   :maxdepth: 2

   getting_started
   demos
   architecture
   algorithm
   limitations
   api
   ngsolve
   building
   references

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
