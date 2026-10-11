"""Dense/sparse numpy-scipy backend: reference implementation of the
Span and Resolvent protocols.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

log = logging.getLogger(__name__)

__all__ = [
    "NumpySpan",
    "MatrixResolvent",
    "MatrixPolyResolvent",
    "ExactSourceError",
]


class NumpySpan:
    """m complex vectors of length n stored as columns of an n×m array."""

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=complex)
        if data.ndim != 2:
            raise ValueError("NumpySpan expects an n×m array")
        self.data = data

    # Constructors #####################################################

    @classmethod
    def random(cls, n: int, m: int, seed=None) -> "NumpySpan":
        """Span of m random complex vectors (reproducible given seed)."""
        rng = np.random.default_rng(seed)
        return cls(rng.random((n, m)) + 1j * rng.random((n, m)))

    @classmethod
    def zeros(cls, n: int, m: int) -> "NumpySpan":
        return cls(np.zeros((n, m), dtype=complex))

    # Span protocol ####################################################

    @property
    def dim(self) -> int:
        return self.data.shape[1]

    @property
    def n(self) -> int:
        return self.data.shape[0]

    def rotate(self, v: np.ndarray) -> None:
        self.data = self.data @ v

    def scale(self, i: int, a: complex) -> None:
        self.data[:, i] *= a

    def remove(self, indices: Sequence[int]) -> None:
        keep = [j for j in range(self.dim) if j not in set(indices)]
        self.data = self.data[:, keep]

    def append_random(self, d: int, rng: np.random.Generator) -> None:
        new = rng.random((self.n, d)) + 1j * rng.random((self.n, d))
        self.data = np.hstack([self.data, new])

    def conjugate(self) -> None:
        """Complex-conjugate all vectors in place (used by reduce_sym)."""
        self.data = self.data.conj()

    def copy(self) -> "NumpySpan":
        return NumpySpan(self.data.copy())

    def dot(self, other: "NumpySpan") -> np.ndarray:
        """Euclidean coefficient inner products ``self^H other``.

        Optional span method used by the eigenvalue count estimate.
        Returns an array of shape ``(self.dim, other.dim)``.
        """
        return self.data.conj().T @ other.data

    def qr(self) -> np.ndarray:
        """Replace the columns by Q and return the reduced QR factor R."""
        q, r = np.linalg.qr(self.data, mode="reduced")
        self.data = q
        return r

    def __repr__(self) -> str:
        return f"NumpySpan(n={self.n}, m={self.dim})"


class MatrixResolvent:
    """Resolvent of the matrix pencil (A, B): (z B − A)⁻¹ B.

    A, B are dense or sparse square matrices; B defaults to identity.
    LU factorizations are computed lazily per quadrature node and cached,
    so a resolvent can be reused across contours sharing nodes.
    """

    def __init__(self, A, B=None):
        self.A = csc_matrix(A)
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if B is None:
            from scipy.sparse import identity
            B = identity(n, dtype=complex, format="csc")
        self.B = csc_matrix(B)
        if self.B.shape != (n, n):
            raise ValueError("B must have the same shape as A")
        self.n = n
        self._lu: dict[complex, object] = {}

    def _factorization(self, z: complex):
        z = complex(z)
        if z not in self._lu:
            log.debug("  MatrixResolvent: factorizing at z = %s", z)
            self._lu[z] = splu((z * self.B - self.A).astype(complex))
        return self._lu[z]

    # Resolvent protocol ###############################################

    def apply(self, z: complex, x: NumpySpan) -> NumpySpan:
        lu = self._factorization(z)
        return NumpySpan(lu.solve(self.B @ x.data))

    def apply_adj(self, z: complex, x: NumpySpan) -> NumpySpan:
        lu = self._factorization(z)
        return NumpySpan(lu.solve(self.B.conj().T @ x.data, trans="H"))

    def new_like(self, x: NumpySpan) -> NumpySpan:
        return NumpySpan.zeros(x.n, x.dim)

    def axpy(self, y: NumpySpan, a: complex, x: NumpySpan) -> None:
        y.data += a * x.data

    def rayleigh(self, yl: NumpySpan, yr: NumpySpan):
        h_a = yl.data.conj().T @ (self.A @ yr.data)
        h_b = yl.data.conj().T @ (self.B @ yr.data)
        return h_a, h_b

    def gram(self, yl: NumpySpan, yr: NumpySpan) -> np.ndarray:
        return yl.data.conj().T @ (self.B @ yr.data)

    def residual(self, lam, yr: NumpySpan, yl: NumpySpan | None = None):
        """Return unnormalized Euclidean 2-norms of algebraic residuals.

        For each right eigenpair, compute ``||A x - lambda B x||_2``
        in coefficient coordinates. No mass weighting or division by
        the eigenvalue, vector norm, or matrix norms is applied.

        If ``yl`` is supplied, also compute the left residual
        ``||A^H y - conjugate(lambda) B^H y||_2`` and return the larger
        right/left norm for each pair.

        Returns
        -------
        numpy.ndarray
            One absolute residual norm per eigenvalue, in input order.

        Notes
        -----
        These values depend on eigenvector scaling and, for finite
        element matrices, the mesh and basis. They are not the discrete
        dual L2 norms ``sqrt(r^H B^{-1} r)``. B-orthonormal eigenvectors
        do not imply that the residual is measured in a B-weighted norm.
        """
        r = self.A @ yr.data - (self.B @ yr.data) * np.asarray(lam)
        res = np.linalg.norm(r, axis=0)
        if yl is not None:
            rl = self.A.conj().T @ yl.data - (
                self.B.conj().T @ yl.data
            ) * np.conjugate(np.asarray(lam))
            res = np.maximum(res, np.linalg.norm(rl, axis=0))
        return res

    def random_span(self, m: int, seed=None) -> NumpySpan:
        return NumpySpan.random(self.n, m, seed=seed)


class MatrixPolyResolvent:
    """Resolvent of the companion pencil of P(λ) = Σᵢ λⁱ A[i].

    Solves the polynomial eigenproblem P(λ) x = 0 for dense or sparse
    coefficients A[0], ..., A[d] by the method of Gopalakrishnan,
    Parker and VandenBerge, Wave Motion 108 (2022) 102826,
    https://doi.org/10.1016/j.wavemoti.2021.102826. The problem is
    linearized to the first companion pencil::

        AA = [  0   I   0  ...   0  ]    BB = [ I           0    ]
             [  0   0   I        :  ]         [    I        :    ]
             [  :        .       I  ]         [       .     0    ]
             [ A[0] A[1] ... A[d-1] ]         [ 0  ...   -A[d]   ]

    of size d·n, but only the n×n matrix P(z) is ever factorized: the
    companion resolvent is applied through block recurrences. A[d] need
    not be invertible; its null space is the eigenspace of λ = ∞, which
    the iteration filters out.

    Spans are :class:`NumpySpan` of length d·n, the blocks stacked
    vertically. Use :meth:`first` to extract the right eigenvectors of
    P and :meth:`last` the left eigenvectors.

    The sign convention is ``z BB − AA``: ``apply`` returns
    (z BB − AA)⁻¹ BB and ``apply_adj`` returns (z BB − AA)⁻ᴴ BBᴴ.
    The companion pencil is generally not Hermitian even when every
    A[i] is, so use ``mode="biorth"`` or ``mode="right"``. See
    :doc:`/algorithm` for how a singular A[d] is handled.

    Parameters
    ----------
    coefs : sequence of matrices
        Dense or sparse n×n coefficients [A0, A1, ..., Ad], d >= 2.
        For a linear pencil use :class:`MatrixResolvent`.
    """

    def __init__(self, coefs):
        self.coefs = [csc_matrix(a).astype(complex) for a in coefs]
        self.degree = len(self.coefs) - 1
        if self.degree < 2:
            raise ValueError(
                "Not a polynomial eigenproblem: need degree >= 2, "
                f"got {self.degree}. Use MatrixResolvent for a pencil."
            )
        n = self.coefs[0].shape[0]
        for a in self.coefs:
            if a.shape != (n, n):
                raise ValueError("coefficients must be n×n of equal size")
        self.n = n
        self._lu: dict[complex, object] = {}

    # Block helpers ####################################################

    def _blocks(self, data: np.ndarray) -> list[np.ndarray]:
        n = self.n
        return [data[k * n:(k + 1) * n] for k in range(self.degree)]

    def component(self, span: NumpySpan, j: int) -> NumpySpan:
        """The j-th length-n block of a companion span."""
        return NumpySpan(self._blocks(span.data)[j].copy())

    def first(self, span: NumpySpan) -> NumpySpan:
        """First block: right eigenvectors of P for a right span."""
        return self.component(span, 0)

    def last(self, span: NumpySpan) -> NumpySpan:
        """Last block: left eigenvectors of P for a left span."""
        return self.component(span, self.degree - 1)

    def random_span(self, m: int, seed=None) -> NumpySpan:
        return NumpySpan.random(self.degree * self.n, m, seed=seed)

    # P(z) and its factorization #######################################

    def pol(self, z: complex):
        """The evaluated n×n matrix polynomial P(z) (sparse)."""
        z = complex(z)
        out = self.coefs[self.degree].copy()
        for a in reversed(self.coefs[:-1]):  # Horner
            out = z * out + a
        return csc_matrix(out)

    def _factorization(self, z: complex):
        z = complex(z)
        if z not in self._lu:
            log.debug("  MatrixPolyResolvent: factorizing at z = %s", z)
            self._lu[z] = splu(self.pol(z))
        return self._lu[z]

    def _bb(self, data: np.ndarray, adjoint: bool = False) -> np.ndarray:
        out = data.copy()
        ad = self.coefs[self.degree]
        if adjoint:
            ad = ad.conj().T
        self._blocks(out)[-1][:] = -(ad @ self._blocks(data)[-1])
        return out

    def _aa(self, data: np.ndarray) -> np.ndarray:
        xs = self._blocks(data)
        out = np.empty_like(data)
        os = self._blocks(out)
        for i in range(self.degree - 1):
            os[i][:] = xs[i + 1]
        os[-1][:] = sum(a @ x for a, x in zip(self.coefs, xs))
        return out

    def _shiftedinv(self, z: complex, yy: np.ndarray) -> np.ndarray:
        """Solve (z BB − AA) X = Y by Theorem 4, eq. (15)."""
        d, z = self.degree, complex(z)
        ys = self._blocks(yy)
        rhs = -ys[d - 1]
        inner = np.zeros_like(ys[0])
        for i in range(1, d + 1):
            # inner = Σ_{j<i} z^{i-1-j} Y_j, but for i = d the j = d-1
            # term cancels against −A_d Y_{d-1} and is omitted.
            inner = z * inner + ys[i - 1] if i < d else z * inner
            rhs = rhs + self.coefs[i] @ inner
        out = np.empty_like(yy)
        xs = self._blocks(out)
        xs[0][:] = self._factorization(z).solve(rhs)
        for i in range(1, d):
            xs[i][:] = z * xs[i - 1] - ys[i - 1]
        return out

    def _shiftedinvadj(self, z: complex, yy: np.ndarray) -> np.ndarray:
        """Solve (z BB − AA)ᴴ X = Y by Theorem 4, eq. (16)."""
        d, zb = self.degree, complex(np.conj(z))
        ys = self._blocks(yy)
        out = np.empty_like(yy)
        xs = self._blocks(out)
        s = np.zeros_like(ys[0])
        for j in reversed(range(d)):  # Horner: Σ_j zb^j Y_j
            s = zb * s + ys[j]
        xs[d - 1][:] = self._factorization(z).solve(-s, trans="H")
        xl = xs[d - 1]
        h = [a.conj().T for a in self.coefs]
        xs[d - 2][:] = -ys[d - 1] - zb * (h[d] @ xl) - h[d - 1] @ xl
        for j in range(d - 3, -1, -1):
            xs[j][:] = zb * xs[j + 1] - ys[j + 1] - h[j + 1] @ xl
        return out

    # Resolvent protocol ###############################################

    def apply(self, z: complex, x: NumpySpan) -> NumpySpan:
        """(z BB − AA)⁻¹ BB x on the companion space."""
        return NumpySpan(self._shiftedinv(z, self._bb(x.data)))

    def apply_adj(self, z: complex, x: NumpySpan) -> NumpySpan:
        """(z BB − AA)⁻ᴴ BBᴴ x, for generalized left filtering."""
        bbx = self._bb(x.data, adjoint=True)
        return NumpySpan(self._shiftedinvadj(z, bbx))

    def new_like(self, x: NumpySpan) -> NumpySpan:
        return NumpySpan.zeros(x.n, x.dim)

    def axpy(self, y: NumpySpan, a: complex, x: NumpySpan) -> None:
        y.data += a * x.data

    def rayleigh(self, yl: NumpySpan, yr: NumpySpan):
        h_a = yl.data.conj().T @ self._aa(yr.data)
        h_b = yl.data.conj().T @ self._bb(yr.data)
        return h_a, h_b

    def gram(self, yl: NumpySpan, yr: NumpySpan) -> np.ndarray:
        return yl.data.conj().T @ self._bb(yr.data)

    def residual(self, lam, yr: NumpySpan, yl: NumpySpan | None = None):
        """Return unnormalized 2-norms of polynomial residuals.

        For each right companion vector, compute ``||P(lambda) x||_2``
        where x is its first block. If ``yl`` is supplied, also compute
        ``||P(lambda)^H y||_2`` with y the last block of the left vector,
        and return the larger of the two. As for
        :meth:`MatrixResolvent.residual`, no normalization is applied.
        """
        xs = self._blocks(yr.data)[0]
        ys = None if yl is None else self._blocks(yl.data)[-1]
        res = np.zeros(len(lam))
        for k, lk in enumerate(np.asarray(lam)):
            p = self.pol(lk)
            res[k] = np.linalg.norm(p @ xs[:, k])
            if ys is not None:
                res[k] = max(
                    res[k], np.linalg.norm(p.conj().T @ ys[:, k])
                )
        return res


class ExactSourceError:
    """Reference SourceErrorEstimator: the exact source error.

    E_z v = R_true(z) v − u_h, where R_true is a trusted ("fine")
    resolvent and u_h = R_h(z) v is the computed solution. Linear in v
    and satisfies Assumption 3.9 with C₀ = 1 and Y = V by construction.

    This is the contract documentation for real backends (a DPG
    backend returns its error representation function instead): useful
    for validating the Theorem 3.10 machinery, not for production —
    if you can afford R_true you don't need an estimator.
    """

    def __init__(self, true_resolvent: MatrixResolvent):
        self.true = true_resolvent

    def estimate(self, z: complex, v: NumpySpan, u: NumpySpan) -> NumpySpan:
        exact = self.true.apply(z, v)
        return NumpySpan(exact.data - u.data)

    def combine(self, weights, ys) -> NumpySpan:
        out = np.zeros_like(ys[0].data)
        for w, y in zip(weights, ys):
            out += complex(w) * y.data
        return NumpySpan(out)

    def y_gram(self, ya: NumpySpan, yb: NumpySpan) -> np.ndarray:
        return ya.data.conj().T @ yb.data
