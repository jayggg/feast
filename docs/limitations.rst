Limitations
===========

These limitations remain open in the current development version.

* **Convergence is not a completeness certificate.** It is judged by the
  distance between successive in-contour eigenvalue sets alone. It does
  not certify that every enclosed eigenvalue was found, nor residual
  accuracy. Inspect residuals (their norms are backend-specific) and
  choose a starting span larger than the expected eigenvalue count.
  Passing ``count_probes`` (for example 16) to :func:`~feast.driver.feast`
  adds a stochastic completeness check. It estimates the *unexplained
  weight*: the sum of the contour's filter values f(λ) over every
  eigenvalue not returned, where f is about 1 well inside the contour
  and near 0 far outside. Near 0 means nothing was missed; about 1 per
  eigenvalue means missed eigenvalues. FEAST warns when the value
  exceeds both 1/4 and three standard errors. The check is statistical,
  so a missed eigenvalue can go undetected (more probes make that less
  likely), and an eigenvalue just outside the contour also triggers it.
  See :mod:`feast.count` for definitions and references.
* **Rank tolerances require judgment.** Hermitian and right-only modes
  discard filtered directions with singular values at or below
  ``rank_tol`` times the largest; biorthogonal mode applies ``eta_tol``
  to the B-Gram matrix. Directions below the threshold are discarded
  even when they contain desired spectral content. To judge a cut, look
  at the singular values the decision was based on, recorded per
  iteration in ``result.history.basis_singular_values`` or
  ``gram_singular_values``, together with ``retained_ranks``. A clear
  gap at the cut is reassuring; a smooth decay is not.
* **Contour splitting is distance based.** ``simple_multiple_zoom``
  groups eigenvalues closer than ``delta`` and gives each group a radius
  of half the gap to its nearest outside neighbor. A long chain of
  closely spaced values can extend beyond its proposed circle. Check
  that each contour encloses its intended cluster before using it.
