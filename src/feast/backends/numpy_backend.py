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

__all__ = ["NumpySpan", "MatrixResolvent", "ExactSourceError"]


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
