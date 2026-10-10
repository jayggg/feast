"""Eigenspace error estimation from source-problem error estimators.

Implements Theorem 3.10 of Gopalakrishnan & Pinochet-Soto, "Estimators
for Eigenspace Error": if a backend can estimate the error of the
discrete source problems u_h ≈ (z_j − A)⁻¹ v at the quadrature nodes
z_j, those estimators combine into a reliable estimator for the gap
between the computed eigenspace E_h and the exact eigenspace E.

Dictionary between the paper and this package:

- S_r = ω₀ + Σ ωⱼ R(zⱼ) with ω₀ = 0, ωⱼ = ``contour.w``, zⱼ =
  ``contour.z``. This is the quadrature used by
  :class:`~feast.projector.SpectralProjectorApprox` for the FEAST run
  (the paper's Example 3.6).
- Assumption 3.9 requires a LINEAR map E : V_h → Y with
  ‖S_r v − S_{r,h} v‖_V ≤ C₀ ‖E v‖_Y. A backend supplies it per node
  through the :class:`SourceErrorEstimator` protocol below; linearity
  means ``estimate`` must return an error *representation* object in Y
  (e.g. a DPG error representation function), never a norm.
- With a basis e¹..e^L of E_h (the FEAST output span), the matrices
  (3.18) are G_ij = (E eʲ, E eⁱ)_Y and M_ij = (eʲ, eⁱ)_V. The maximal
  eigenpair (λ̂, x̂) of G x = λ M x gives the estimator
  η = ‖E ê_h‖_Y = √λ̂ with ê_h = Σ x̂ᵢ eⁱ, ‖ê_h‖_V = 1 (3.19), and
  Theorem 3.10 gives gap_V(E_h, E) ≤ C η.
- ê_h and E ê_h are returned so a backend can compute local
  (elementwise) indicators of E ê_h to drive adaptive refinement; the
  estimator depends only on the span of E_h, not the basis (cluster
  robustness).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.linalg import eigh

from .contour import Contour
from .projector import Resolvent
from .span import Span

log = logging.getLogger(__name__)

__all__ = [
    "SourceErrorEstimator",
    "EigenspaceErrorEstimate",
    "estimate_eigenspace_error",
]


@runtime_checkable
class SourceErrorEstimator(Protocol):
    """Backend contract for Assumption 3.9 (source error estimators).

    Y-objects returned by ``estimate``/``combine`` are spans of error
    representations (one column per input column). They must support
    ``copy()`` and in-place ``rotate(matrix)`` so the worst error
    direction can be extracted. The three methods below must use the
    same representation and inner product.
    """

    def estimate(self, z: complex, v: Span, u: Span):
        """Error representation in Y of the source problem
        (z − A) w = v whose computed solution is u (= R_h(z) v).
        Must be LINEAR in v (return a representation, not a norm)."""
        ...

    def combine(self, weights: Sequence[complex], ys: list):
        """Return Σₖ weights[k] · ys[k] (linear combination in Y)."""
        ...

    def y_gram(self, ya, yb) -> np.ndarray:
        """Gram matrix of Y-inner products: out[i, j] = (yb[j], ya[i])_Y,
        Hermitian positive semidefinite when ya is yb."""
        ...


@dataclass
class EigenspaceErrorEstimate:
    """Result of Theorem 3.10's construction.

    eta:    the global estimator ‖E ê_h‖_Y = √λ̂;
            gap_V(E_h, E) ≤ C·eta for the theorem's constant C.
    worst:  ê_h — the V-normalized worst-case element of the computed
            eigenspace (a Span with one column).
    eworst: E ê_h in Y — apply backend-local norms to it for
            elementwise adaptivity indicators.
    rayleigh_values: all eigenvalues of G x = λ M x (ascending);
            their square roots estimate errors of successively
            better-approximated directions in E_h.
    """

    eta: float
    worst: Span
    eworst: object
    rayleigh_values: np.ndarray
    G: np.ndarray
    M: np.ndarray


def estimate_eigenspace_error(
    source_estimator: SourceErrorEstimator,
    resolvent: Resolvent,
    contour: Contour,
    span: Span,
    v_gram=None,
) -> EigenspaceErrorEstimate:
    """Estimate the eigenspace gap for a FEAST-computed span.

    Parameters
    ----------
    source_estimator : SourceErrorEstimator
        Backend implementation of the linear source error representation
        described in Assumption 3.9.
    resolvent : Resolvent
        The same resolvent used in the FEAST run, allowing cached
        factorizations to be reused.
    contour : Contour
        The same contour as the FEAST run; its rational filter must match.
    span : Span
        Computed eigenspace basis, for example ``FeastResult.right``.
    v_gram : callable or None, optional
        ``v_gram(a, b)`` returns the V-inner-product matrix defining
        ‖·‖_V. Defaults to ``resolvent.gram``, the B-Gram matrix.
        The resulting matrix must be positive definite on ``span``;
        supply a suitable inner product if B is not Hermitian positive
        definite, or for a stronger norm such as a graph norm.

    Returns
    -------
    EigenspaceErrorEstimate
        Global estimator and its worst direction.
    """
    if span.dim == 0:
        raise ValueError("empty span: nothing to estimate")

    # E e^i for all basis vectors at once: Σ_j w_j · estimate(z_j, ...)
    ys = []
    for z, w in zip(contour.z, contour.w):
        u = resolvent.apply(z, span)
        ys.append(source_estimator.estimate(z, span, u))
    ey = source_estimator.combine(contour.w, ys)

    # The small Hermitian eigenproblem (3.18): G x = λ M x.
    G = source_estimator.y_gram(ey, ey)
    M = v_gram(span, span) if v_gram is not None else \
        resolvent.gram(span, span)
    G = 0.5 * (G + G.conj().T)
    M = 0.5 * (M + M.conj().T)
    lam, X = eigh(G, M)

    lam_hat = float(lam[-1])
    xhat = X[:, -1]                     # eigh gives x^H M x = 1
    eta = float(np.sqrt(max(lam_hat, 0.0)))
    log.info(
        "eigenspace estimator: eta = %.3e (rayleigh values %s)",
        eta, np.array_str(np.sqrt(np.maximum(lam, 0)), precision=2),
    )

    worst = span.copy()
    worst.rotate(xhat.reshape(-1, 1))
    eworst = _rotate_y(ey, xhat)

    return EigenspaceErrorEstimate(
        eta=eta,
        worst=worst,
        eworst=eworst,
        rayleigh_values=lam,
        G=G,
        M=M,
    )


def _rotate_y(ey, xhat):
    """E ê_h = Σ x̂ᵢ (E eⁱ), using the Y-object's span interface."""
    if hasattr(ey, "copy") and hasattr(ey, "rotate"):
        out = ey.copy()
        out.rotate(np.asarray(xhat).reshape(-1, 1))
        return out
    raise TypeError(
        "Y-objects returned by this SourceErrorEstimator must support "
        "copy/rotate (span-like); wrap them in a span type."
    )
