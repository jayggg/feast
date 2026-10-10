"""The Resolvent protocol and the SpectralProjectorApprox composition.

A backend supplies a :class:`Resolvent`; a :class:`~feast.contour.Contour`
supplies the quadrature. :class:`SpectralProjectorApprox` applies the
right and left filters::

    Qr = Σₖ wₖ (zₖ B − A)⁻¹ B Yr
    Ql = Σₖ conj(wₖ) (zₖ B − A)⁻ᴴ Bᴴ Yl

The left filter targets generalized left eigenvectors satisfying
Aᴴ y = conj(λ) Bᴴ y. For general B, it is not the Euclidean adjoint
of the right filter: the order of Bᴴ and the inverse matters.

Quadrature nodes are processed serially. Backend methods control vector
storage and shifted solves. The driver constructs its projector internally;
it does not currently accept a custom projector or an MPI scheduler.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .contour import Contour
from .span import Span

__all__ = [
    "Resolvent",
    "SpectralProjectorApprox",
    "SPA",
    "SymmetryReducedProjector",
]


@runtime_checkable
class Resolvent(Protocol):
    """What a discretization must provide to the FEAST machinery."""

    def apply(self, z: complex, x: Span) -> Span:
        """Return (z B − A)⁻¹ B x as a new span."""
        ...

    def apply_adj(self, z: complex, x: Span) -> Span:
        """Return (z B − A)⁻ᴴ Bᴴ x as a new span."""
        ...

    def new_like(self, x: Span) -> Span:
        """Return a zero span with the same shape/storage as x."""
        ...

    def axpy(self, y: Span, a: complex, x: Span) -> None:
        """y += a * x (in-place accumulation over quadrature nodes)."""
        ...

    def rayleigh(self, yl: Span, yr: Span):
        """Return the pair of small matrices
        (ylᴴ A yr, ylᴴ B yr), each of shape (yl.dim, yr.dim),
        in the backend's inner product."""
        ...

    def gram(self, yl: Span, yr: Span) -> np.ndarray:
        """Return only ylᴴ B yr (used for kernel cleaning)."""
        ...

    def residual(self, lam: np.ndarray, yr: Span, yl: Span | None = None):
        """Optional: norms of A y − λ B y (and left residuals if yl)."""
        ...


class SpectralProjectorApprox:
    """Quadrature sum of backend resolvent applications."""

    def __init__(self, resolvent: Resolvent, contour: Contour):
        self.resolvent = resolvent
        self.contour = contour

    def apply(self, y: Span) -> Span:
        """Return P·y (right application)."""
        out = self.resolvent.new_like(y)
        for z, w in zip(self.contour.z, self.contour.w):
            self.resolvent.axpy(out, w, self.resolvent.apply(z, y))
        return out

    def apply_adj(self, yl: Span) -> Span:
        """Return the filtered generalized left span.

        Sum ``conj(w) * resolvent.apply_adj(z, yl)`` over the nodes.
        This equals the Euclidean adjoint projector action when B is
        identity; the general pencil uses the left filter defined above.
        """
        out = self.resolvent.new_like(yl)
        for z, w in zip(self.contour.z, self.contour.w):
            self.resolvent.axpy(
                out, np.conjugate(w), self.resolvent.apply_adj(z, yl)
            )
        return out


class SymmetryReducedProjector(SpectralProjectorApprox):
    """Projector application with quadrature nodes halved by symmetry.

    Assumes R(conj z)·f = conj(R(z)·conj(f)), which holds when the
    operator pencil is real (e.g. self-adjoint problems with real
    matrices), and that the node set is symmetric about the real axis.
    Only one node from each conjugate pair needs a factorization;
    the reflected node's contribution is recovered by conjugation.

    Requires spans with an in-place ``conjugate()`` method (NumpySpan
    and NGSpan both provide it). Left application is not available, so
    this is for mode="hermitian" only.
    """

    def __init__(self, resolvent: Resolvent, contour: Contour):
        super().__init__(resolvent, contour)
        z, w = contour.z, contour.w
        zmax = np.max(np.abs(z))
        used: set[int] = set()
        # (node index, weight, reflected weight or None)
        self.reduced: list[tuple[int, complex, complex | None]] = []
        for i in range(len(z)):
            if i in used:
                continue
            if abs(z[i].imag) < 1e-13 * zmax:
                self.reduced.append((i, w[i], None))
                used.add(i)
                continue
            match = np.where(np.abs(z[i].conjugate() - z) / zmax < 1e-14)[0]
            if len(match) != 1:
                raise ValueError(
                    "Quadrature nodes are not symmetric about the real "
                    "axis; cannot use symmetry reduction."
                )
            j = int(match[0])
            self.reduced.append((i, w[i], w[j]))
            used.update((i, j))

    def apply(self, y: Span) -> Span:
        out = self.resolvent.new_like(y)
        for i, w, w_refl in self.reduced:
            z = self.contour.z[i]
            self.resolvent.axpy(out, w, self.resolvent.apply(z, y))
            if w_refl is not None:
                # conj(R(z) conj(y)) = R(conj z) y
                yc = y.copy()
                yc.conjugate()
                ry = self.resolvent.apply(z, yc)
                ry.conjugate()
                self.resolvent.axpy(out, w_refl, ry)
        return out

    def apply_adj(self, yl: Span) -> Span:
        raise NotImplementedError(
            "Adjoint application is not implemented with symmetry "
            "reduction; use SpectralProjectorApprox."
        )


#: Short alias for :class:`SpectralProjectorApprox`.
SPA = SpectralProjectorApprox
