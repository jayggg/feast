"""The FEAST driver: one iteration loop for all modes.

The ``mode`` parameter selects the reduced eigenproblem and subspace
treatment:

- ``"hermitian"``: A self-adjoint; Rayleigh–Ritz via eigh.
- ``"right"``: non-self-adjoint, right eigenvectors only.
- ``"biorth"``: non-self-adjoint with biorthogonal left/right spans and
  SVD kernel cleaning between projector applications.

Subspace decisions (SVD cuts, convergence, refresh) are made from small
m×m matrices supplied by the backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgError, eig, eigh

from .contour import Contour
from .projector import (
    Resolvent,
    SpectralProjectorApprox,
    SymmetryReducedProjector,
)
from .span import Span
from .subspace import analyze_gram, apply_cleaning, orthonormalize

log = logging.getLogger(__name__)

__all__ = ["feast", "FeastResult", "FeastHistory"]

_MODES = ("hermitian", "right", "biorth")


@dataclass
class FeastHistory:
    """Diagnostics collected during FEAST iterations.

    Attributes
    ----------
    ritz_values : list of numpy.ndarray
        Ritz values after each successful reduced solve, before trimming.
    hausdorff : list of float
        Eigenvalue-set distances divided by ``contour.scale``.
    residuals : list of numpy.ndarray
        Backend residuals when ``report_residuals=True``.
    gram_singular_values : list of numpy.ndarray
        Singular values from successful biorthogonal Gram analyses.

    Notes
    -----
    Lists need not have the same length: a failed solve increments the
    iteration count without recording Ritz values, and Gram analysis is
    performed only in biorthogonal mode.
    """

    ritz_values: list = field(default_factory=list)
    hausdorff: list = field(default_factory=list)
    residuals: list = field(default_factory=list)
    gram_singular_values: list = field(default_factory=list)


@dataclass
class FeastResult:
    """Outcome of a FEAST run.

    Attributes
    ----------
    eigenvalues : numpy.ndarray
        Most recently computed Ritz values, with any contour trimming.
    right : Span
        Last computed right Ritz vectors, one column per eigenvalue.
    left : Span or None
        Corresponding left Ritz vectors in biorthogonal mode; otherwise
        ``None``. Search vectors added for refresh are not returned.
    converged : bool
        Whether the relative eigenvalue-set distance met ``stop_tol``.
        The requested ``expected_dim``, if any, must also be recovered.
        This is not a residual-based accuracy or completeness guarantee.
    iterations : int
        Total attempted iterations across all restarts.
    history : FeastHistory
        Recorded iteration diagnostics.
    """

    eigenvalues: np.ndarray
    right: Span
    left: Span | None
    converged: bool
    iterations: int
    history: FeastHistory

    def __repr__(self) -> str:
        return (
            f"FeastResult(converged={self.converged}, "
            f"iterations={self.iterations}, "
            f"eigenvalues={np.array_str(self.eigenvalues, precision=6)})"
        )


def _hausdorff(lam: np.ndarray, lam0: np.ndarray, scale: float) -> float:
    """Relative Hausdorff distance between eigenvalue sets."""
    if len(lam0) == 0:
        return np.inf
    h = np.abs(lam[:, None] - lam0[None, :])
    return max(h.min(axis=0).max(), h.min(axis=1).max()) / scale


def _small_eig(h_a, h_b, mode):
    """Solve the reduced Rayleigh–Ritz problem; return (lam, wl, wr)."""
    if mode == "hermitian":
        lam, wr = eigh(h_a, h_b)
        return lam, None, wr
    if mode == "biorth":
        lam, wl, wr = eig(h_a, h_b, left=True)
        return lam, wl, wr
    lam, wr = eig(h_a, h_b)
    return lam, None, wr


def _augment(y, yl, count, rng):
    y = y.copy()
    y.append_random(count, rng)
    if yl is not None:
        yl = yl.copy()
        yl.append_random(count, rng)
    return y, yl


def feast(
    resolvent: Resolvent,
    contour: Contour,
    y: Span,
    *,
    mode: str = "hermitian",
    yl: Span | None = None,
    stop_tol: float = 1e-11,
    eta_tol: float = 1e-8,
    rank_tol: float = 1e-12,
    niterations: int = 50,
    nrestarts: int = 1,
    expected_dim: int | None = None,
    nrefresh: int = 3,
    check_contour: int = 3,
    report_residuals: bool = False,
    reduce_sym: bool = False,
    seed=None,
) -> FeastResult:
    """Run FEAST for eigenvalues of the resolvent's pencil enclosed by
    the contour.

    Parameters
    ----------
    resolvent : Resolvent
        Backend implementing the pencil's shifted solves and reduced
        matrices.
    contour : Contour
        Quadrature rule and membership test for the search region.
    y : Span
        Initial right span. Its dimension limits the number of Ritz
        values in the first iteration; refreshes and restarts may enlarge
        it. Pass a copy if the initial span must be preserved.
    mode : {"hermitian", "right", "biorth"}, optional
        Hermitian Rayleigh–Ritz, right-only iteration, or a two-sided
        iteration with SVD kernel cleaning. Hermitian mode requires a
        positive definite reduced B matrix.
    yl : Span or None, optional
        Initial left span for ``mode="biorth"``, with the same dimension
        as ``y``. If omitted, independent random vectors are generated
        using ``seed``. Ignored in the other modes.
    stop_tol : float, optional
        Stopping tolerance on the relative Hausdorff distance between
        successive eigenvalue sets; defaults to 1e-11.
    eta_tol : float, optional
        Relative singular-value threshold for kernel cleaning in
        biorthogonal mode; defaults to 1e-8. See :mod:`feast.subspace`.
    rank_tol : float, optional
        Relative singular-value threshold for filtered basis reduction
        in Hermitian and right-only modes; defaults to 1e-12. Directions
        with singular values at or below this fraction of the largest
        are removed using QR and a small SVD. Must lie in (0, 1).
    niterations : int, optional
        Maximum iterations per attempt; defaults to 50.
    nrestarts : int, optional
        Total number of attempts, including the first; defaults to 1.
        Values at or below zero also request one attempt. Later attempts
        append random vectors to enlarge the current spans.
    expected_dim : int or None, optional
        Minimum requested span size. Smaller spans are replenished while
        refreshes remain, and contour trimming will not reduce the span
        below this size. A run that cannot recover this dimension is
        reported as nonconverged. This does not certify an eigenvalue
        count. Newly appended columns must pass through another solve.
    nrefresh : int, optional
        Maximum replenishments per attempt; defaults to 3.
    check_contour : int, optional
        Check for out-of-contour Ritz values every this many iterations;
        defaults to 3. Zero disables the check. A check does not remove
        all vectors or reduce the span below ``expected_dim``.
    report_residuals : bool, optional
        Record backend residuals in the history; defaults to ``False``.
        The backend must implement ``residual``. Residuals do not control
        convergence, and their norm and normalization depend on the
        backend.
    reduce_sym : bool, optional
        Reuse factorizations for conjugate node pairs. Requires
        ``mode="hermitian"``, a real pencil, a node set symmetric about
        the real axis, and spans with ``conjugate()``. Defaults to False.
    seed : int or numpy.random.Generator or None, optional
        Randomness for refreshes, restarts, and the default left span.
        Seed the initial right span separately for reproducibility.

    Returns
    -------
    FeastResult
        Last successfully computed Ritz pairs and diagnostics. Returned
        spans have one column per eigenvalue, including on nonconvergence.
        If no reduced solve succeeds, the eigenvalues and spans are empty.

    Notes
    -----
    A refresh or restart resets the convergence comparison. At least two
    successful solves with equal eigenvalue counts are needed afterward.
    See the documentation's current limitations for contour membership
    and eigenvalue-set convergence caveats.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    if reduce_sym and mode != "hermitian":
        raise ValueError('reduce_sym requires mode="hermitian"')
    if not np.isfinite(rank_tol) or not 0 < rank_tol < 1:
        raise ValueError("rank_tol must be finite and in (0, 1)")
    if nrestarts <= 0:
        nrestarts = 1
    rng = np.random.default_rng(seed)
    if reduce_sym:
        projector = SymmetryReducedProjector(resolvent, contour)
    else:
        projector = SpectralProjectorApprox(resolvent, contour)
    biorth = mode == "biorth"
    if biorth and yl is not None and yl.dim != y.dim:
        raise ValueError("left and right spans must have the same dimension")
    if biorth and yl is None:
        # independent random left span, built via Span protocol only
        m = y.dim
        yl = y.copy()
        yl.remove(list(range(m)))
        yl.append_random(m, rng)
    if not biorth:
        yl = None

    history = FeastHistory()
    saved_lam = np.array([], dtype=complex)
    saved_y = None
    saved_yl = None
    converged = False
    total_iterations = 0

    for restart in range(nrestarts):
        if restart > 0:
            log.info("FEAST restart %d: augmenting span", restart)
            y, yl = _augment(y, yl, max(y.dim, 1), rng)
        refreshes_left = nrefresh
        lam0 = np.array([], dtype=complex)

        for it in range(1, niterations + 1):
            total_iterations += 1
            log.info("FEAST iteration %d with %d vectors", it, y.dim)

            qr = projector.apply(y)
            ql = projector.apply_adj(yl) if biorth else None

            if biorth and qr.dim:
                analysis = analyze_gram(
                    resolvent.gram(ql, qr), eta_tol=eta_tol
                )
                if analysis.ok:
                    history.gram_singular_values.append(analysis.s)
                    if analysis.cut:
                        log.info(
                            "kernel cleaning: cutting %s of %d "
                            "(singular values %s)",
                            analysis.cut, qr.dim,
                            np.array_str(analysis.s, precision=1),
                        )
                    apply_cleaning(analysis, ql, qr)
                else:
                    log.warning("invalid B-Gram matrix; restarting")
                    break
            elif not biorth:
                try:
                    before = qr.dim
                    orthonormalize(qr, rank_tol)
                    log.info(
                        "rank reduction: retained %d of %d", qr.dim, before,
                    )
                except (LinAlgError, ValueError) as exc:
                    log.warning("rank reduction failed (%s); restarting", exc)
                    break

            if qr.dim == 0:
                if refreshes_left > 0 and it < niterations:
                    count = max(expected_dim or y.dim, 1)
                    log.info("no usable directions; refreshing %d", count)
                    y, yl = _augment(qr, ql, count, rng)
                    refreshes_left -= 1
                    lam0 = np.array([], dtype=complex)
                    continue
                log.warning("no usable directions; restarting")
                break

            h_a, h_b = resolvent.rayleigh(ql if biorth else qr, qr)
            try:
                lam, wl, wr = _small_eig(h_a, h_b, mode)
                if not np.all(np.isfinite(lam)):
                    raise LinAlgError("non-finite Ritz values")
            except (LinAlgError, ValueError) as exc:
                log.warning("small eigensolve failed (%s); restarting", exc)
                break

            qr.rotate(wr)
            if biorth:
                ql.rotate(wl)
            y = qr
            if biorth:
                yl = ql

            history.ritz_values.append(lam.copy())
            if report_residuals:
                res = resolvent.residual(lam, y, yl)
                history.residuals.append(res)
                log.info("residuals: %s", np.array_str(res, precision=2))
            dist = _hausdorff(lam, lam0, contour.scale)
            same_count = len(lam) == len(lam0)
            history.hausdorff.append(dist)
            log.info(
                "Ritz values %s; relative Hausdorff %.3e",
                np.array_str(np.sort_complex(lam), precision=6), dist,
            )

            if check_contour and it % check_contour == 0:
                inside = contour.within(lam)
                cut = [k for k in range(len(lam)) if not inside[k]]
                keep_enough = (
                    expected_dim is None
                    or len(lam) - len(cut) >= expected_dim
                )
                if cut and len(cut) < len(lam) and keep_enough:
                    log.info("removing out-of-contour Ritz values %s", cut)
                    lam = lam[inside]
                    y.remove(cut)
                    if yl is not None:
                        yl.remove(cut)

            saved_lam, saved_y, saved_yl = lam.copy(), y, yl

            if expected_dim is not None and y.dim < expected_dim:
                if refreshes_left > 0 and it < niterations:
                    d = expected_dim - y.dim
                    log.info("refreshing span with %d random vector(s)", d)
                    y, yl = _augment(y, yl, d, rng)
                    refreshes_left -= 1
                    lam0 = np.array([], dtype=complex)
                    continue
                log.warning("requested span dimension was not recovered")
                break

            lam0 = lam
            if y.dim == 0:
                log.warning("no candidates left; restarting")
                break
            if same_count and dist < stop_tol:
                log.info("FEAST converged: Hausdorff %.3e", dist)
                converged = True
                break

        if converged:
            break

    if not converged:
        log.warning("FEAST did not converge in %d restart(s)", nrestarts)

    if saved_y is None:
        saved_y = y.copy()
        saved_y.remove(list(range(saved_y.dim)))
        if yl is not None:
            saved_yl = yl.copy()
            saved_yl.remove(list(range(saved_yl.dim)))

    return FeastResult(
        eigenvalues=saved_lam,
        right=saved_y,
        left=saved_yl,
        converged=converged,
        iterations=total_iterations,
        history=history,
    )
