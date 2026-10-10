feast: contour-integral eigensolvers
==========================================

``feast`` computes selected eigenvalues and eigenspaces of a matrix pencil
``A x = λ B x`` using contour integration and subspace iteration. The
distribution name is ``feast-eig``; Python imports use ``feast``.

The package separates the search contour, discretization backend, and
iteration. NumPy/SciPy support is included; NGSolve support is optional.

.. toctree::
   :maxdepth: 2

   getting_started
   demos
   architecture
   limitations
   api
   ngsolve
   building

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
