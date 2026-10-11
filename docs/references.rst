References
================

.. [Pol2009] E. Polizzi, "Density-matrix-based algorithm for solving
   eigenvalue problems," *Physical Review B* 79:115112, 2009.
   `doi:10.1103/PhysRevB.79.115112
   <https://doi.org/10.1103/PhysRevB.79.115112>`_

   The FEAST algorithm.

.. [GPV2022] J. Gopalakrishnan, B. Q. Parker and P. VandenBerge,
   "Computing leaky modes of optical fibers using a FEAST algorithm
   for polynomial eigenproblems," *Wave Motion* 108:102826, 2022.
   `doi:10.1016/j.wavemoti.2021.102826
   <https://doi.org/10.1016/j.wavemoti.2021.102826>`_

   The companion linearization and block recurrences used by
   :class:`~feast.backends.numpy_backend.MatrixPolyResolvent` and
   ``NGPolyResolvent`` (Theorem 4 and Algorithm 1 there), and the
   polynomial section of :doc:`algorithm`.

.. [GGO2020] J. Gopalakrishnan, L. Grubišić and J. Ovall, "Spectral
   discretization errors in filtered subspace iteration," *Mathematics
   of Computation* 89(321):203–228, 2020.
   `doi:10.1090/mcom/3483 <https://doi.org/10.1090/mcom/3483>`_

   FEAST as filtered subspace iteration applied to a discretized
   resolvent of a selfadjoint operator: convergence of the iteration
   (Theorem 3.4), eigenspace discretization errors (Theorem 4.1), and
   bounds on the Hausdorff distance between computed and exact
   eigenvalue clusters (Theorem 5.7, Corollary 5.8), the measure used
   in the convergence test of :doc:`algorithm`.

.. [GP2026] J. Gopalakrishnan and G. Pinochet-Soto, "Reliable
   eigenspace error estimation using source error estimators,"
   *Computational Methods in Applied Mathematics*, 2026 (online ahead
   of print).
   `doi:10.1515/cmam-2025-0161
   <https://doi.org/10.1515/cmam-2025-0161>`_

   The eigenspace gap estimator of :mod:`feast.estimator`
   (Theorem 3.10 there), built from error estimators for the source
   problems solved at the contour nodes.
