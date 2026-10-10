"""The Span protocol: what the FEAST driver requires of a set of vectors.

The driver never touches vector entries. It manipulates spans only
through the methods below, so that a backend may store vectors as numpy
columns, NGSolve MultiVectors, or MPI-distributed vectors. All branch
decisions in the driver are made from small matrices (QR factors supplied
by the Span and Gram/Rayleigh matrices supplied by the Resolvent), which are
reduction results that a distributed backend must make consistent
on every rank.

``NumpySpan`` (in ``feast.backends.numpy_backend``) is the reference
implementation.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["Span"]


@runtime_checkable
class Span(Protocol):
    """m vectors spanning a subspace; the FEAST-facing surface."""

    @property
    def dim(self) -> int:
        """Number of vectors m currently in the span."""
        ...

    def rotate(self, v: np.ndarray) -> None:
        """Replace span S by S @ v, where v is m×k (in-place)."""
        ...

    def scale(self, i: int, a: complex) -> None:
        """Scale the i-th vector by a (in-place)."""
        ...

    def remove(self, indices: Sequence[int]) -> None:
        """Delete the vectors at the given positions (in-place)."""
        ...

    def append_random(self, d: int, rng: np.random.Generator) -> None:
        """Append d random vectors drawn from rng (in-place).

        Must be deterministic given (rng state, storage partition).
        """
        ...

    def copy(self) -> "Span":
        """Deep copy."""
        ...

    def qr(self) -> np.ndarray:
        """Replace the span Y by an orthonormal basis Q and return R.

        Y = Q R up to roundoff. R has shape (Q.dim, original_dim);
        it need not be triangular. Backends may discard directions at
        roundoff level, including zero columns. An empty basis returns
        an array with zero rows. The default inner product is the
        coefficient Euclidean inner product, independent of the pencil B.
        """
        ...
