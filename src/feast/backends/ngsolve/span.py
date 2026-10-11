"""NGSpan: the Span protocol over NGSolve MultiVectors.

Algebra only — postprocessing helpers (grid functions, drawing) live in
:mod:`feast.backends.ngsolve.post`, keeping the FEAST-facing surface
minimal.
"""

from __future__ import annotations

from typing import Sequence

import ngsolve as ng
import numpy as np

__all__ = ["NGSpan"]


class NGSpan:
    """m vectors from a (complex) NGSolve finite element space.

    Vectors are stored in an ``ngsolve.MultiVector``. Values at
    Dirichlet-constrained dofs are kept at zero.
    """

    def __init__(self, fes, m: int = 0):
        self.fes = fes
        self._gf = ng.GridFunction(fes, autoupdate=fes.autoupdate)
        self._mv = ng.MultiVector(self._gf.vec, m) if m > 0 else None
        if m > 0:
            self.zero()

    # Constructors #####################################################

    @classmethod
    def random(cls, fes, m: int, seed=None) -> "NGSpan":
        """Span of m random vectors, zero at constrained dofs."""
        s = cls(fes, 0)
        s.append_random(m, np.random.default_rng(seed))
        return s

    @classmethod
    def from_numpy(cls, fes, y: np.ndarray) -> "NGSpan":
        """Import columns of the ndof×m array y (constrained dofs zeroed)."""
        n, m = y.shape
        if n != fes.ndof:
            raise ValueError(f"expected {fes.ndof} rows, got {n}")
        s = cls(fes, m)
        free = np.array(fes.FreeDofs())
        for j in range(m):
            s._mv[j].FV().NumPy()[:] = y[:, j] * free
        return s

    # Utilities ########################################################

    @property
    def n(self) -> int:
        return self.fes.ndof

    def zero(self) -> "NGSpan":
        for k in range(self.dim):
            self._mv[k].FV().NumPy()[:] = 0
        return self

    def zero_constrained(self) -> None:
        """Zero out values at Dirichlet-constrained dofs."""
        free = np.array(self.fes.FreeDofs())
        for k in range(self.dim):
            self._mv[k].FV().NumPy()[~free] = 0

    def to_numpy(self) -> np.ndarray:
        y = np.zeros((self.n, self.dim), dtype=complex)
        for j in range(self.dim):
            y[:, j] = self._mv[j].FV().NumPy()
        return y

    # Span protocol ####################################################

    @property
    def dim(self) -> int:
        return 0 if self._mv is None else len(self._mv)

    def rotate(self, v: np.ndarray) -> None:
        """Replace span S by S @ v (v is m×k)."""
        m, k = v.shape
        if m != self.dim:
            raise ValueError(f"rotation rows {m} != span dim {self.dim}")
        w = ng.MultiVector(self._gf.vec, k)
        for col in range(k):
            w[col].FV().NumPy()[:] = 0
            for j in range(m):
                w[col].data += complex(v[j, col]) * self._mv[j]
        self._mv = w

    def scale(self, i: int, a: complex) -> None:
        self._mv[i] *= complex(a)

    def remove(self, indices: Sequence[int]) -> None:
        keep = [j for j in range(self.dim) if j not in set(indices)]
        self._mv = self._mv[keep] if keep else None

    def append_random(self, d: int, rng: np.random.Generator) -> None:
        if d == 0:
            return
        old = self.dim
        if self._mv is None:
            self._mv = ng.MultiVector(self._gf.vec, d)
        else:
            self._mv.Extend(d)
        free = np.array(self.fes.FreeDofs())
        for j in range(old, old + d):
            vals = (rng.random(self.n) + 1j * rng.random(self.n)) * free
            self._mv[j].FV().NumPy()[:] = vals

    def conjugate(self) -> None:
        """Complex-conjugate all vectors in place (used by reduce_sym)."""
        for k in range(self.dim):
            self._mv[k].FV().NumPy().imag *= -1

    def new(self, fes=None, m: int | None = None) -> "NGSpan":
        """A zero span of the same (sub)class — subclasses are preserved
        through FEAST iterations, so conveniences added by downstream
        packages survive on the returned spans. Subclasses must remain
        constructible as ``type(self)(fes, m)``."""
        return type(self)(
            fes if fes is not None else self.fes,
            self.dim if m is None else m,
        )

    def copy(self) -> "NGSpan":
        c = self.new()
        for k in range(self.dim):
            c._mv[k].data = self._mv[k]
        return c

    def dot(self, other: "NGSpan") -> np.ndarray:
        """Euclidean coefficient inner products ``self^H other``.

        Optional span method used by the eigenvalue count estimate.
        Returns an array of shape ``(self.dim, other.dim)``.
        """
        if self.dim == 0 or other.dim == 0:
            return np.zeros((self.dim, other.dim), dtype=complex)
        # ng.InnerProduct(a, b)[i, j] = conj(b_i) . a_j, i.e. (b^H a)
        return np.array(ng.InnerProduct(other._mv, self._mv))

    def qr(self, ipmat=None) -> np.ndarray:
        """Replace Y by an orthonormal basis Q and return R with Y ≈ Q R.

        Parameters
        ----------
        ipmat : ngsolve.BaseMatrix or None, optional
            Hermitian positive definite inner product on the free dofs.
            None uses the coefficient Euclidean inner product. The FEAST
            driver uses this default, independently of the pencil B.

        Returns
        -------
        numpy.ndarray
            Factor with shape (retained_dimension, original_dimension).
            Pivoting is absorbed into its columns; it need not be
            triangular. Directions at roundoff level are discarded.

        Notes
        -----
        Pivoted, reorthogonalized Gram–Schmidt keeps vectors in NGSolve.
        Native normalization is called only for nonzero residuals.
        """
        m = self.dim
        if m == 0:
            return np.empty((0, 0), dtype=complex)

        def weighted(mv):
            if ipmat is None:
                return mv
            out = ng.MultiVector(self._gf.vec, len(mv))
            out[:] = ipmat * mv
            return out

        def norms(mv):
            if ipmat is None:
                return np.array([ng.Norm(v) for v in mv])
            wm = weighted(mv)
            return np.array([
                np.sqrt(abs(ng.InnerProduct(wm[j], mv[j])))
                for j in range(len(mv))
            ])

        original = self._mv
        work = self.copy()._mv
        q = ng.MultiVector(self._gf.vec, 0)
        remaining = list(range(m))
        initial_norms = norms(work)
        if not np.all(np.isfinite(initial_norms)):
            raise ValueError("cannot orthogonalize non-finite vectors")
        floor = np.finfo(float).eps * m * initial_norms.max()
        while remaining:
            sizes = norms(work[remaining])
            pivot = int(np.argmax(sizes))
            if sizes[pivot] <= floor:
                break
            j = remaining.pop(pivot)
            candidate = ng.MultiVector(self._gf.vec, 1)
            candidate[0].data = work[j]
            if len(q):
                for _ in range(2):
                    coeff = ng.InnerProduct(weighted(candidate), q).NumPy()
                    candidate[:] -= q * ng.Matrix(coeff)
            if norms(candidate)[0] <= floor:
                continue
            if ipmat is None:
                q.AppendOrthogonalize(candidate[0], iterations=2)
            else:
                # The complex custom-mass AppendOrthogonalize path does
                # not preserve orthogonality in NGSolve 6.2.2608.
                candidate.Orthogonalize(ipmat=ipmat)
                q.Append(candidate[0])
            if remaining:
                rest = work[remaining]
                coeff = ng.InnerProduct(weighted(rest), q[-1:]).NumPy()
                rest[:] -= q[-1:] * ng.Matrix(coeff)

        r = (
            ng.InnerProduct(weighted(original), q).NumPy().copy()
            if len(q) else np.empty((0, m), dtype=complex)
        )
        self._mv = q if len(q) else None
        return r

    def __repr__(self) -> str:
        return f"NGSpan(ndof={self.n}, m={self.dim})"
