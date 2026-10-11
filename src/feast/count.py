"""Completeness check: is any filter weight left unexplained?

A FEAST run can converge and still miss eigenvalues inside its contour,
for instance when the starting span is too small. This module estimates,
from random probe vectors, how much of the contour's filter is not
accounted for by the eigenvalues a run returned.
:func:`~feast.driver.feast` runs the check when ``count_probes > 0`` and
returns the result as ``result.count_estimate``; it can also be called
directly with :func:`estimate_eigenvalue_count`.

.. rubric:: Filter weight

The quadrature of a :class:`~feast.contour.Contour` defines a rational
filter f(z) = Σₖ wₖ / (zₖ − z), available as
:meth:`~feast.contour.Contour.filter`. The *filter weight* of an
eigenvalue λ is Re f(λ): about 1 well inside the contour, about 1/2 on
it, and near 0 far outside. (For the circle of the diagnostics demo
with 4 nodes, the eigenvalues 9.9, 39.5, 88.8 inside have weights 0.61,
1.00, 0.64; the next one, 157.9, just outside, has 0.03.)

For a diagonalizable pencil the approximate spectral projector
P = Σₖ wₖ R(zₖ) equals V f(Λ) V⁻¹, so

    trace(P) = Σᵢ f(λᵢ),

the *total filter weight*, a filter-weighted eigenvalue count. It is
not an integer: eigenvalues near the contour count fractionally. (A
Jordan block contributes f(λ) per multiplicity, so the identity also
holds for defective pencils.)

.. rubric:: The two reported quantities

``estimate`` ± ``stderr``
    The total filter weight trace(P). Useful for sizing a starting span
    before a run, but noisy.

``unexplained`` ± ``unexplained_stderr``
    The *unexplained weight*: trace(P) minus the weight of the supplied
    eigenvalues, that is, Σ f(λ) over every eigenvalue *not* returned.
    This is the quantity to read after a FEAST run, and it is much less
    noisy (see Method).

In both, "±" is one standard error of the mean over the probes: the
sample standard deviation of the per-probe values divided by
√nprobes. It shrinks like 1/√nprobes. It is a noise level, not a
guaranteed bound, and with few probes (such as 16) it tends to be
slightly optimistic.

.. rubric:: Reading the unexplained weight

========================  ============================================
unexplained weight        likely meaning
========================  ============================================
near 0 (within noise)     every eigenvalue that the filter sees was
                          returned
about 1 per eigenvalue    eigenvalues missed well inside the contour;
                          use a larger starting span
between 1/2 and 1         an eigenvalue missed near the contour
up to about 1/2           an eigenvalue just outside the contour (not
                          an error, but the contour is tight)
========================  ============================================

:meth:`EigenvalueCountEstimate.suggests_missing`, and with it the
warning logged by :func:`~feast.driver.feast`, fires when the unexplained
weight exceeds both 1/4 and three standard errors. Because the check is
statistical, a missed eigenvalue with small filter weight can go
undetected; more probes make that less likely.

.. rubric:: Method

The trace is estimated without forming P, by Hutchinson's method [1]_:
trace(P) = E[dᴴ P d] / c for random probe vectors d with E[d dᴴ] = c I;
see [2]_ for an analysis of its accuracy and [3]_ for its use in
counting eigenvalues with FEAST's filters. Spans only generate random
vectors with entries uniform on [0, 1), whose mean is not zero, so each
probe is the difference of two independent random vectors: its entries
have mean zero and E[d dᴴ] = I/3. For finite element spans the probes
vanish on constrained dofs, so the trace is taken over the free dofs,
which is what the resolvent acts on.

The unexplained weight uses the same probes with a control variate.
With X the returned right vectors, Yl any left vectors and
G = Ylᴴ B X invertible, the operator P_X = X f(Λ) G⁻¹ Ylᴴ B satisfies
trace(P_X) = Σⱼ f(λⱼ) over the returned eigenvalues exactly, because
the trace is cyclic. Averaging dᴴ (P − P_X) d / c is therefore unbiased
for the unexplained weight, whatever left vectors are used; accurate
ones make the samples nearly zero when nothing is missing, so a missed
eigenvalue stands out against very little noise. Subtracting a known
part before applying Hutchinson's estimator is the same idea as the
deflation step of Hutch++ [4]_.

The estimate costs one projector application to ``nprobes`` vectors,
reusing any factorizations the resolvent has cached. It requires spans
with a ``dot`` method returning Euclidean coefficient inner products
(:class:`~feast.backends.numpy_backend.NumpySpan` and
:class:`~feast.backends.ngsolve.span.NGSpan` provide it).

.. rubric:: References

.. [1] M. F. Hutchinson, "A stochastic estimator of the trace of the
   influence matrix for Laplacian smoothing splines", *Communications
   in Statistics - Simulation and Computation* 18(3):1059-1076, 1989.
   doi:10.1080/03610918908812806
.. [2] H. Avron and S. Toledo, "Randomized algorithms for estimating
   the trace of an implicit symmetric positive semi-definite matrix",
   *Journal of the ACM* 58(2), Article 8, 2011.
   doi:10.1145/1944345.1944349
.. [3] E. Di Napoli, E. Polizzi and Y. Saad, "Efficient estimation of
   eigenvalue counts in an interval", *Numerical Linear Algebra with
   Applications* 23(4):674-692, 2016. doi:10.1002/nla.2048
.. [4] R. A. Meyer, C. Musco, C. Musco and D. P. Woodruff, "Hutch++:
   optimal stochastic trace estimation", Symposium on Simplicity in
   Algorithms (SOSA), SIAM, 2021. arXiv:2010.09649
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .contour import Contour
from .projector import Resolvent, SpectralProjectorApprox
from .span import Span

log = logging.getLogger(__name__)

__all__ = ["EigenvalueCountEstimate", "estimate_eigenvalue_count"]

_PROBE_COVARIANCE = 1.0 / 3.0   # E[d d^H] = I/3 for differences of U[0,1)


def _mean_and_stderr(samples: np.ndarray) -> tuple[float, float]:
    re = np.asarray(samples).real
    stderr = float(np.std(re, ddof=1) / np.sqrt(len(re))) if len(re) > 1 \
        else 0.0
    return float(np.mean(re)), stderr


@dataclass(frozen=True)
class EigenvalueCountEstimate:
    """Result of the completeness check (see :mod:`feast.count`).

    Attributes
    ----------
    estimate : float
        Total filter weight trace(P) = Σᵢ Re f(λᵢ) over all eigenvalues,
        estimated as the mean of the real parts of ``samples``. A
        filter-weighted eigenvalue count; not an integer.
    stderr : float
        One standard error of ``estimate``: the sample standard deviation
        of the per-probe values divided by √nprobes. Zero for one probe.
    samples : numpy.ndarray
        Complex per-probe values dᴴ P d / c, with c = 1/3 the probe
        covariance.
    found_weight : float or None
        Σ Re f(λ) over the supplied eigenvalues, computed exactly from
        :meth:`~feast.contour.Contour.filter`; ``None`` if no eigenpairs
        were supplied.
    unexplained : float or None
        Unexplained weight: Σ Re f(λ) over all eigenvalues not supplied,
        estimated with the control variate described in
        :mod:`feast.count`. Near 0 when nothing is missing, about 1 per
        eigenvalue missed well inside the contour, up to about 1/2 per
        eigenvalue just outside it. ``None`` if no eigenpairs were
        supplied.
    unexplained_stderr : float or None
        One standard error of ``unexplained``, computed like ``stderr``
        from the per-probe differences.
    """

    estimate: float
    stderr: float
    samples: np.ndarray
    found_weight: float | None = None
    unexplained: float | None = None
    unexplained_stderr: float | None = None

    def suggests_missing(self) -> bool:
        """Whether the unexplained weight is clearly above noise.

        True when ``unexplained`` exceeds both 1/4 and three times
        ``unexplained_stderr``: some eigenvalue inside or just outside
        the contour is probably not among those supplied. See the table
        in :mod:`feast.count` for how to read the value.
        """
        if self.unexplained is None:
            return False
        return (self.unexplained > 0.25
                and self.unexplained > 3 * self.unexplained_stderr)

    def __str__(self) -> str:
        total = (f"total weight {self.estimate:.2f} ± {self.stderr:.2f} "
                 f"({len(self.samples)} probes)")
        if self.unexplained is None:
            return total
        return (f"unexplained weight {self.unexplained:.2g} ± "
                f"{self.unexplained_stderr:.2g}; {total}")


def _probe_span(like: Span, nprobes: int, rng: np.random.Generator,
                resolvent: Resolvent) -> Span:
    """Mean-zero probes built only from Span and Resolvent operations."""
    d = like.copy()
    d.remove(list(range(d.dim)))
    d.append_random(nprobes, rng)
    other = d.copy()
    other.remove(list(range(other.dim)))
    other.append_random(nprobes, rng)
    resolvent.axpy(d, -1.0, other)          # d = u1 - u2
    return d


def estimate_eigenvalue_count(
    resolvent: Resolvent,
    contour: Contour,
    like: Span,
    nprobes: int = 16,
    seed=None,
    projector: SpectralProjectorApprox | None = None,
    eigenvalues=None,
    right: Span | None = None,
    left: Span | None = None,
) -> EigenvalueCountEstimate:
    """Estimate the total filter weight Σᵢ f(λᵢ) and, given computed
    eigenpairs, the weight they leave unexplained.

    Definitions, the meaning of the standard errors, how to read the
    result, and references are in :mod:`feast.count`.
    :func:`~feast.driver.feast` calls this function when ``count_probes > 0``.

    Parameters
    ----------
    resolvent : Resolvent
        Backend for the pencil (see :class:`~feast.projector.Resolvent`);
        cached factorizations are reused.
    contour : Contour
        Contour whose filter f (:meth:`~feast.contour.Contour.filter`)
        weights the eigenvalues.
    like : Span
        Any span of the backend's vector type, for example the initial
        FEAST span. Only its type and storage are used; it is not
        modified. Its class must provide ``dot``.
    nprobes : int, optional
        Number of random probe vectors; defaults to 16. Standard errors
        decrease like ``1/sqrt(nprobes)``.
    seed : int or numpy.random.Generator or None, optional
        Randomness for the probes.
    projector : SpectralProjectorApprox or None, optional
        Projector to apply, for example a
        :class:`~feast.projector.SymmetryReducedProjector`. Defaults to
        ``SpectralProjectorApprox(resolvent, contour)``.
    eigenvalues : array_like or None, optional
        Computed eigenvalues, for example ``result.eigenvalues``. When
        given, the weight they leave unexplained is estimated too.
    right : Span or None, optional
        Right eigenvectors, one per eigenvalue, for example
        ``result.right``. Required with ``eigenvalues``.
    left : Span or None, optional
        Left eigenvectors, for example ``result.left`` in biorthogonal
        mode. If omitted, the right vectors are used in their place. Any
        left vectors give an unbiased estimate; accurate ones give the
        smallest variance.

    Returns
    -------
    EigenvalueCountEstimate
        ``estimate`` and ``stderr`` always; ``found_weight``,
        ``unexplained`` and ``unexplained_stderr`` when eigenpairs are
        supplied.

    Notes
    -----
    The estimate is a filter-weighted count, not an integer. With a
    coarse quadrature, eigenvalues near the contour contribute
    fractions, so do not round it. To check a FEAST result, use
    ``unexplained`` (or :meth:`EigenvalueCountEstimate.suggests_missing`)
    rather than comparing a rounded estimate with the number found.
    """
    if nprobes < 1:
        raise ValueError("nprobes must be at least 1")
    if not hasattr(like, "dot"):
        raise TypeError(
            f"{type(like).__name__} has no dot method; the count estimate "
            "needs Euclidean inner products between spans"
        )
    rng = np.random.default_rng(seed)
    d = _probe_span(like, nprobes, rng, resolvent)
    if projector is None:
        projector = SpectralProjectorApprox(resolvent, contour)
    pd = projector.apply(d)
    samples = np.diagonal(d.dot(pd)) / _PROBE_COVARIANCE
    estimate, stderr = _mean_and_stderr(samples)

    found = unexplained = unexplained_stderr = None
    if eigenvalues is not None:
        lam = np.asarray(eigenvalues, dtype=complex)
        if right is None or right.dim != len(lam):
            raise ValueError("right must have one vector per eigenvalue")
        found = float(np.sum(contour.filter(lam).real)) if len(lam) else 0.0
        diff = samples
        if len(lam):
            yl = right if left is None else left
            g = resolvent.gram(yl, right)               # Yl^H B X
            coef = np.linalg.solve(g, resolvent.gram(yl, d))
            px_d = right.copy()                         # P_X d =
            px_d.rotate(contour.filter(lam)[:, None] * coef)  # X F G^-1 ..
            diff = samples - np.diagonal(d.dot(px_d)) / _PROBE_COVARIANCE
        unexplained, unexplained_stderr = _mean_and_stderr(diff)

    result = EigenvalueCountEstimate(
        estimate, stderr, np.asarray(samples),
        found, unexplained, unexplained_stderr,
    )
    log.debug("Eigenvalue count estimate: %s", result)
    return result
