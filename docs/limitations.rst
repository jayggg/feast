Current limitations
=========================

These limitations remain open in the current development version.

* Custom contours use ``abs(mean(z))`` as their convergence scale. A
  zero or nearly zero mean can prevent convergence. Built-in circle and
  ellipse factories use a scale that accounts for the contour radius.
* ``simple_multiple_zoom`` mixes sorted indices with unsorted centers.
  Supply sorted real eigenvalues and keep the permutation if original
  indices are needed. Check that each proposed contour encloses its
  intended cluster before using it.

There is no DPG discretization backend, MPI scheduler, automatic
mesh-transfer method, or built-in adaptive refinement loop. Residual
norms are backend-specific; convergence uses eigenvalue-set distance
alone and does not certify that all enclosed eigenvalues were found.

Hermitian and right-only modes reduce filtered bases by QR and SVD using
``rank_tol``. Biorthogonal mode uses the B-Gram threshold ``eta_tol``.
Results contain matching in-contour eigenvalues and vectors from the last
successful reduced solve. Empty in-contour sets and runs with fewer than
``expected_dim`` in-contour values are nonconverged. Rank tolerances still
require judgment: directions below the
threshold are discarded even when they contain desired spectral content.
