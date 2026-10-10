"""Polynomial eigenproblems via companion linearization.

Solves A(λ) x = 0 where A(λ) = A[0] + λ A[1] + ⋯ + λᵈ A[d] is a matrix
polynomial, by the method of https://doi.org/10.1016/j.wavemoti.2021.102826:
the problem is linearized to the first companion pencil::

    AA = [  0   I   0  ...   0  ]    BB = [ I           0    ]
         [  0   0   I        :  ]         [    I        :    ]
         [  :        .       I  ]         [       .     0    ]
         [ A[0] A[1] ... A[d-1] ]         [ 0  ...   -A[d]   ]

of size d·ndof, but only the SHORT ndof×ndof matrix A(z) is ever
factorized: the companion resolvent is applied through block
recurrences.

Spans live in the product space [fes]^d — plain :class:`NGSpan` over
``ngsolve.FESpace([fes]*d)``; no special span class is needed. Use
:func:`first_component` to extract the physical eigenvector block.

The sign convention is ``z BB - AA``. Thus ``apply`` returns
(z BB − AA)⁻¹ BB, and ``apply_adj`` returns (z BB − AA)⁻ᴴ BBᴴ.
The latter filters generalized left eigenvectors; when BB is Hermitian
positive definite, it is also the BB-inner-product adjoint of ``apply``.

-A[d] must be Hermitian and generate a (semi-)inner product.
"""

from __future__ import annotations

import logging

import ngsolve as ng
import numpy as np

from .span import NGSpan

log = logging.getLogger(__name__)

__all__ = ["NGPolyResolvent", "first_component", "component"]


def component(span: NGSpan, fes, j: int) -> NGSpan:
    """Extract the j-th block (length ndof) of a product-space span."""
    n = fes.ndof
    out = span.new(fes=fes)
    for k in range(span.dim):
        out._mv[k].FV().NumPy()[:] = (
            span._mv[k].FV().NumPy()[j * n:(j + 1) * n]
        )
    return out


def first_component(span: NGSpan, fes) -> NGSpan:
    """The physical eigenvector block of a companion-space span."""
    return component(span, fes, 0)


class NGPolyResolvent:
    """Resolvent of the companion pencil of A(λ) = Σ λⁱ A[i].

    Parameters
    ----------
    coefs : list of NGSolve matrices
        Assembled coefficients [A0, A1, ..., Ad], for example
        ``[a0.mat, a1.mat, a2.mat]``, all on the same fes.
        Requires d = len(coefs) - 1 >= 2 and -A[d] Hermitian.
    fes : ngsolve.FESpace
        The short finite element space the coefficients act on.
    inverse : str or None, optional
        NGSolve sparse inverse type for factorizing A(z).

    Notes
    -----
    ``product_fes`` is the companion space [fes]^d; make initial spans
    with :meth:`random_span` or ``NGSpan.random(R.product_fes, m)``.
    """

    def __init__(self, coefs, fes, inverse: str | None = None):
        self.coefs = list(coefs)
        self.degree = len(self.coefs) - 1
        if self.degree < 2:
            raise ValueError(
                "Not a polynomial eigenproblem: need degree >= 2, "
                f"got {self.degree}"
            )
        self.fes = fes
        self.inverse = inverse
        self.product_fes = ng.FESpace([fes] * self.degree)
        self._inv: dict[complex, object] = {}
        self._short = ng.GridFunction(fes).vec
        self.n = fes.ndof

    def random_span(self, m: int, seed=None) -> NGSpan:
        return NGSpan.random(self.product_fes, m, seed=seed)

    def first(self, span: NGSpan) -> NGSpan:
        """First (physical eigenvector) block of a companion span."""
        return component(span, self.fes, 0)

    def last(self, span: NGSpan) -> NGSpan:
        """Extract the last block of a companion-space span.

        For a left companion eigenvector, this block is a left null
        vector of the evaluated matrix polynomial A(λ).
        """
        return component(span, self.fes, self.degree - 1)

    # A(z) and its factorization #######################################

    def pol(self, z: complex):
        """The evaluated short matrix polynomial A(z) (sparse)."""
        with ng.TaskManager():
            expr = complex(z**0) * self.coefs[0]
            for i in range(1, self.degree + 1):
                expr = expr + complex(z**i) * self.coefs[i]
            return expr.CreateSparseMatrix()

    def _factorization(self, z: complex):
        z = complex(z)
        if z not in self._inv:
            log.info("NGPolyResolvent: factorizing A(z) at z = %s", z)
            with ng.TaskManager():
                self._inv[z] = self.pol(z).Inverse(
                    self.fes.FreeDofs(), inverse=self.inverse
                )
        return self._inv[z]

    # Block-vector helpers (long = d*ndof BaseVector) ##################

    def _blocks(self, vv):
        n = self.n
        return [vv[k * n:(k + 1) * n] for k in range(self.degree)]

    def _apply_bb(self, xx, out, adjoint: bool = False) -> None:
        """out = BB xx (or BB^H xx), blockwise."""
        xs = self._blocks(xx)
        os = self._blocks(out)
        d = self.degree
        for k in range(d - 1):
            os[k].data = xs[k]
        Ad = self.coefs[d]
        tmp = self._short.CreateVector()
        tmp.data = (Ad.H if adjoint else Ad) * xs[d - 1]
        os[d - 1].data = -1.0 * tmp

    def _shiftedinv(self, z, yy, Ryy) -> None:
        """Solve (z BB - AA) Ryy = yy using block elimination."""
        suminner = self._short.CreateVector()
        sumouter = self._short.CreateVector()
        d = self.degree
        ys = self._blocks(yy)
        Rys = self._blocks(Ryy)
        sumouter.FV().NumPy()[:] = 0.0

        for k in range(1, d):
            suminner.data = complex(z ** (k - 1)) * ys[0]
            for j in range(1, k):
                suminner.data += complex(z ** (k - 1 - j)) * ys[j]
            sumouter.data += self.coefs[k] * suminner

        suminner.data = complex(z ** (d - 1)) * ys[0]
        for j in range(1, d - 1):
            suminner.data += complex(z ** (d - 1 - j)) * ys[j]
        sumouter.data += self.coefs[d] * suminner - ys[d - 1]

        Rys[0].data = self._factorization(z) * sumouter
        for k in range(1, d):
            Rys[k].data = complex(z) * Rys[k - 1] - ys[k - 1]

    def _shiftedinvadj(self, z, yy, xx) -> None:
        """Solve (z BB - AA).H xx = yy using block elimination."""
        ys = self._blocks(yy)
        xs = self._blocks(xx)
        zbar = complex(np.conjugate(z))
        d = self.degree
        s = self._short.CreateVector()

        # x[d-1] = -inv(A(z).H) * sum_j conj(z)^j y[j]
        s.data = -1.0 * ys[0]
        zbarj = zbar
        for j in range(1, d):
            s.data -= zbarj * ys[j]
            zbarj = complex(zbar * zbarj)
        xs[d - 1].data = self._factorization(z).H * s

        # x[d-2] = -y[d-1] - conj(z) A[d].H x[d-1] - A[d-1].H x[d-1]
        s.data = self.coefs[d].H * xs[d - 1]
        s.data *= zbar
        s.data += self.coefs[d - 1].H * xs[d - 1]
        xs[d - 2].data = -1.0 * ys[d - 1] - s

        # x[d-3] ... x[0]
        for j in range(d - 3, -1, -1):
            s.data = self.coefs[j + 1].H * xs[d - 1]
            xs[j].data = zbar * xs[j + 1] - ys[j + 1] - s

    # Resolvent protocol ###############################################

    def apply(self, z: complex, x: NGSpan) -> NGSpan:
        """(z BB − AA)⁻¹ BB x on the companion space."""
        out = x.new()
        bbx = x._gf.vec.CreateVector()
        with ng.TaskManager():
            for k in range(x.dim):
                self._apply_bb(x._mv[k], bbx)
                self._shiftedinv(z, bbx, out._mv[k])
        return out

    def apply_adj(self, z: complex, x: NGSpan) -> NGSpan:
        """Return (z BB − AA)⁻ᴴ BBᴴ x for generalized left filtering."""
        out = x.new()
        bbx = x._gf.vec.CreateVector()
        with ng.TaskManager():
            for k in range(x.dim):
                self._apply_bb(x._mv[k], bbx, adjoint=True)
                self._shiftedinvadj(z, bbx, out._mv[k])
        return out

    def new_like(self, x: NGSpan) -> NGSpan:
        return x.new()

    def axpy(self, y: NGSpan, a: complex, x: NGSpan) -> None:
        y._mv[:] += complex(a) * x._mv

    def rayleigh(self, yl: NGSpan, yr: NGSpan):
        d = self.degree
        with ng.TaskManager():
            work = ng.MultiVector(yr._gf.vec, yr.dim)
            for k in range(yr.dim):
                work[k].FV().NumPy()[:] = 0.0
                xs = self._blocks(yr._mv[k])
                ws = self._blocks(work[k])
                for i in range(d - 1):  # (AA x)[i] = x[i+1]
                    ws[i].data = xs[i + 1]
                for i in range(d):  # (AA x)[d-1] = sum A[i] x[i]
                    ws[d - 1].data += self.coefs[i] * xs[i]
            h_a = ng.InnerProduct(work, yl._mv).NumPy()
            for k in range(yr.dim):
                self._apply_bb(yr._mv[k], work[k])
            h_b = ng.InnerProduct(work, yl._mv).NumPy()
        return h_a, h_b

    def gram(self, yl: NGSpan, yr: NGSpan) -> np.ndarray:
        with ng.TaskManager():
            work = ng.MultiVector(yr._gf.vec, yr.dim)
            for k in range(yr.dim):
                self._apply_bb(yr._mv[k], work[k])
            return ng.InnerProduct(work, yl._mv).NumPy()

    def residual(self, lam, yr: NGSpan, yl: NGSpan | None = None):
        """Return physical residual norms on free dofs of the first block.

        Compute the coefficient 2-norm of A(λ) x for each right vector.
        ``yl`` is accepted for the common interface but is not used.
        """
        res = np.zeros(len(lam))
        r = self._short.CreateVector()
        free = np.array(self.fes.FreeDofs())
        for k in range(len(lam)):
            xs = self._blocks(yr._mv[k])
            r.data = self.pol(lam[k]) * xs[0]
            res[k] = np.linalg.norm(r.FV().NumPy() * free)
        return res
