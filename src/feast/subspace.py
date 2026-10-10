"""Rank reduction of filtered bases and biorthogonal B-Gram analysis.

Hermitian and right-only iteration use QR followed by an SVD of its
small factor. This reveals the basis singular values without squaring
the condition number through a Gram matrix.

Biorthogonal analysis uses only the small m×m Gram matrix. Its rotations,
scalings, and cuts are then applied through the Span protocol, keeping
rank decisions separate from vector storage and mutation.

Given the B-Gram matrix G = Yl^H B Yr = U S W^H, rotating Yl by U and Yr
by W makes the Gram matrix exactly diag(S). Since U, W are unitary, the
rotation is perfectly conditioned regardless of how singular G is.
Singular values with s[i] <= eta_tol * s[0] (RELATIVE threshold) mark
kernel directions to cut; kept vectors are scaled by 1/sqrt(s[i]) so the
retained Gram matrix is close to the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, svd

__all__ = [
    "GramAnalysis", "analyze_gram", "apply_cleaning", "orthonormalize",
]


def orthonormalize(span, rank_tol: float = 1e-12) -> np.ndarray:
    """Remove dependent directions using QR followed by a small SVD.

    The span is replaced in place by orthonormal columns corresponding
    to singular values greater than ``rank_tol * s[0]``. Return all
    singular values of the QR factor, before truncation. The tolerance
    is relative to the basis singular values, not their squares and not
    the biorthogonal B-Gram singular values used by ``eta_tol``.
    """
    if not np.isfinite(rank_tol) or not 0 < rank_tol < 1:
        raise ValueError("rank_tol must be finite and in (0, 1)")
    r = span.qr()
    u, s, _ = svd(r, full_matrices=False)
    keep = s > rank_tol * s[0] if len(s) else np.zeros(0, dtype=bool)
    if np.any(keep):
        span.rotate(u[:, keep])
    else:
        span.remove(list(range(span.dim)))
    return s


@dataclass(frozen=True)
class GramAnalysis:
    """Result of analyzing a B-Gram matrix.

    ok=False means the analysis failed (non-finite Gram or zero matrix)
    and no cleaning should be applied.
    """

    ok: bool
    u: np.ndarray | None = None       # left rotation (unitary)
    w: np.ndarray | None = None       # right rotation (unitary)
    s: np.ndarray | None = None       # singular values, descending
    keep: list[int] | None = None     # indices to keep (prefix of range)
    cut: list[int] | None = None      # indices to cut

    @property
    def rank(self) -> int:
        return len(self.keep) if self.keep is not None else 0


def analyze_gram(gram: np.ndarray, eta_tol: float = 1e-8) -> GramAnalysis:
    """SVD analysis of the m×m B-Gram matrix with relative threshold.

    eta_tol is relative: directions with s[i] <= eta_tol * s[0] are
    marked for removal.
    """
    try:
        u, s, wh = svd(gram)
    except (ValueError, LinAlgError):
        return GramAnalysis(ok=False)
    if s[0] == 0 or not np.isfinite(s[0]):
        return GramAnalysis(ok=False)
    m = gram.shape[0]
    keep = [i for i in range(m) if s[i] > eta_tol * s[0]]
    cut = [i for i in range(m) if s[i] <= eta_tol * s[0]]
    return GramAnalysis(True, u, wh.conj().T, s, keep, cut)


def apply_cleaning(analysis: GramAnalysis, yl, yr) -> None:
    """Rotate spans to the singular bases, rescale kept directions to a
    near-identity Gram matrix, and cut kernel directions (in-place).

    yl, yr: Span objects (yr rotated by W, yl by U). After this call,
    Yl^H B Yr ≈ identity of size analysis.rank.
    """
    if not analysis.ok:
        return
    yl.rotate(analysis.u)
    yr.rotate(analysis.w)
    for i in analysis.keep:
        f = 1.0 / np.sqrt(analysis.s[i])
        yl.scale(i, f)
        yr.scale(i, f)
    if analysis.cut:
        yl.remove(analysis.cut)
        yr.remove(analysis.cut)
