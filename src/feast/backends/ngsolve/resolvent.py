"""NGSolve resolvents implementing the feast Resolvent protocol.

Two flavors:

- :class:`NGPencilResolvent` — for pencils (A, B) given as assembled
  NGSolve matrices: R(z) = (z B − A)⁻¹ B.
  B need not be Hermitian (e.g. a mass matrix assembled with complex
  PML is only complex symmetric).

- :class:`NGCallbackResolvent` — wraps a user factory
  ``make(z) -> r`` where r supplies in-place ``act`` and ``adj``
  operations, reduced matrices through ``rayleigh_nsa``, and optionally
  ``residual``. The required interface is documented on the class.

Reduced matrices use the following inner-product orientation:
h_a[i, j] = (A yr[j], yl[i])_H via ``ng.InnerProduct(A·yr, yl)``.
"""

from __future__ import annotations

import logging

import ngsolve as ng
import numpy as np

from .span import NGSpan

log = logging.getLogger(__name__)

__all__ = ["NGPencilResolvent", "NGCallbackResolvent"]


class NGPencilResolvent:
    """R(z) = (z B − A)⁻¹ B for assembled NGSolve matrices A, B.

    Parameters
    ----------
    fes : ngsolve.FESpace
        Complex finite element space supplying FreeDofs and storage.
    A, B : ngsolve.BaseMatrix
        Matrices of assembled bilinear forms, for example ``a.mat`` and
        ``b.mat``. B need not be Hermitian: apply_adj uses Bᴴ, so
        left eigenvectors (Aᴴ y = λ̄ Bᴴ y) are filtered correctly either
        way. When B is Hermitian it generates the H-inner product (a
        semi-inner product is allowed). No inverse of B is needed,
        but the reduced eigenproblem must remain well posed.
    inverse : str or None, optional
        NGSolve sparse inverse type, such as "umfpack" or "pardiso".
        The inverse must implement the conjugate-transpose operation
        ``.H`` correctly when left filtering is requested.

    Notes
    -----
    Factorizations are computed lazily per quadrature node and cached.
    """

    def __init__(self, fes, A, B, inverse: str | None = None):
        self.fes = fes
        self.A = A
        self.B = B
        self.inverse = inverse
        self._inv: dict[complex, object] = {}

    def _factorization(self, z: complex):
        z = complex(z)
        if z not in self._inv:
            log.info("NGPencilResolvent: factorizing at z = %s", z)
            with ng.TaskManager():
                r = self.A.CreateMatrix()
                r.AsVector().data = (
                    z * self.B.AsVector() - self.A.AsVector()
                )
                self._inv[z] = r.Inverse(
                    self.fes.FreeDofs(), inverse=self.inverse
                )
        return self._inv[z]

    # Resolvent protocol ###############################################

    def apply(self, z: complex, x: NGSpan) -> NGSpan:
        rinv = self._factorization(z)
        out = x.new()
        with ng.TaskManager():
            bx = ng.MultiVector(x._gf.vec, x.dim)
            bx[:] = self.B * x._mv
            out._mv[:] = rinv * bx
        out.zero_constrained()
        return out

    def apply_adj(self, z: complex, x: NGSpan) -> NGSpan:
        """(z B − A)⁻ᴴ Bᴴ x.

        For a left eigenvector y (Aᴴ y = λ̄ Bᴴ y) this gives
        y / (z̄ − λ̄), the filter FEAST needs. Using B instead of Bᴴ is
        only correct when B is Hermitian; with PML, B is only complex
        symmetric.
        """
        rinv = self._factorization(z)
        out = x.new()
        with ng.TaskManager():
            bx = ng.MultiVector(x._gf.vec, x.dim)
            bx[:] = self.B.H * x._mv
            out._mv[:] = rinv.H * bx
        out.zero_constrained()
        return out

    def new_like(self, x: NGSpan) -> NGSpan:
        return x.new()

    def axpy(self, y: NGSpan, a: complex, x: NGSpan) -> None:
        y._mv[:] += complex(a) * x._mv

    def rayleigh(self, yl: NGSpan, yr: NGSpan):
        with ng.TaskManager():
            work = ng.MultiVector(yr._gf.vec, yr.dim)
            work[:] = self.A * yr._mv
            h_a = ng.InnerProduct(work, yl._mv).NumPy()
            work[:] = self.B * yr._mv
            h_b = ng.InnerProduct(work, yl._mv).NumPy()
        return h_a, h_b

    def gram(self, yl: NGSpan, yr: NGSpan) -> np.ndarray:
        with ng.TaskManager():
            work = ng.MultiVector(yr._gf.vec, yr.dim)
            work[:] = self.B * yr._mv
            h_b = ng.InnerProduct(work, yl._mv).NumPy()
        return h_b

    def residual(self, lam, yr: NGSpan, yl: NGSpan | None = None):
        """Coefficient-vector 2-norms of A y − λ B y on free dofs.

        These are unnormalized coefficient norms. If ``yl`` is given,
        return the larger right/left norm for each eigenvalue.
        """
        free = np.array(self.fes.FreeDofs())
        res = np.zeros(yr.dim)
        r = yr._gf.vec.CreateVector()
        for j in range(yr.dim):
            r.data = self.A * yr._mv[j]
            r.data += complex(-lam[j]) * (self.B * yr._mv[j])
            res[j] = np.linalg.norm(r.FV().NumPy() * free)
        if yl is not None:
            for j in range(yl.dim):
                r.data = self.A.H * yl._mv[j]
                r.data += complex(-np.conj(lam[j])) * (self.B.H * yl._mv[j])
                res[j] = max(res[j], np.linalg.norm(r.FV().NumPy() * free))
        return res


class NGCallbackResolvent:
    """Wrap a factory of in-place NGSolve resolvent operations.

    Parameters
    ----------
    make : callable
        ``make(z)`` returns an object for the complex shift ``z``.
        One object is cached per shift. It must provide the methods below.

    Notes
    -----
    ``act(v, out)`` must overwrite ``out`` with ``(z B - A)⁻¹ B v``.
    ``adj(v, out)`` must overwrite it with ``(z B - A)⁻ᴴ Bᴴ v`` for
    biorthogonal iteration. Inputs and outputs are :class:`NGSpan`
    instances exposing ``dim``, ``fes``, and the ``_mv`` MultiVector.
    Callbacks must preserve the inputs and zero constrained output dofs.

    ``rayleigh_nsa(ql, qr)`` returns ``(qlᴴ A qr, qlᴴ B qr)`` as NumPy
    arrays with shape ``(ql.dim, qr.dim)``. It must also accept
    ``qAq=None`` and still return a pair whose second entry is the Gram
    matrix. Reduced matrices must describe the same pencil for every
    shift: the wrapper uses the first cached object's method.

    ``residual(lam, yr, yl)`` is required only when residual reporting is
    enabled; it returns one diagnostic value per eigenvalue. The wrapper
    does not pass workspace arguments, so callbacks must allocate their
    own scratch storage or provide defaults. ``rayleigh`` and ``gram``
    become available after the first resolvent application.
    """

    def __init__(self, make):
        self._make = make
        self._cache: dict[complex, object] = {}

    def _resolvent(self, z: complex):
        z = complex(z)
        if z not in self._cache:
            log.info("NGCallbackResolvent: making resolvent at z = %s", z)
            self._cache[z] = self._make(z)
        return self._cache[z]

    def _any_resolvent(self):
        if not self._cache:
            raise RuntimeError(
                "No resolvent constructed yet; rayleigh/gram are only "
                "available after the projector has been applied once."
            )
        return next(iter(self._cache.values()))

    # Resolvent protocol ###############################################

    def apply(self, z: complex, x: NGSpan) -> NGSpan:
        out = x.new()
        self._resolvent(z).act(x, out)
        return out

    def apply_adj(self, z: complex, x: NGSpan) -> NGSpan:
        out = x.new()
        self._resolvent(z).adj(x, out)
        return out

    def new_like(self, x: NGSpan) -> NGSpan:
        return x.new()

    def axpy(self, y: NGSpan, a: complex, x: NGSpan) -> None:
        y._mv[:] += complex(a) * x._mv

    def rayleigh(self, yl: NGSpan, yr: NGSpan):
        return self._any_resolvent().rayleigh_nsa(yl, yr)

    def gram(self, yl: NGSpan, yr: NGSpan) -> np.ndarray:
        _, h_b = self._any_resolvent().rayleigh_nsa(yl, yr, qAq=None)
        return h_b

    def residual(self, lam, yr: NGSpan, yl: NGSpan | None = None):
        r = self._any_resolvent()
        if hasattr(r, "residual"):
            return r.residual(lam, yr, yl)
        raise NotImplementedError(
            "underlying resolvent has no residual method"
        )
