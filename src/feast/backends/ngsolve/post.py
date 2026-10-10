"""Postprocessing helpers for NGSpan (kept out of the Span protocol)."""

from __future__ import annotations

import ngsolve as ng

from .span import NGSpan

__all__ = ["gridfun", "draw"]


def gridfun(span: NGSpan, name: str = "eigenmode", i: int | None = None):
    """Return a (multidim) GridFunction holding the span's vectors,
    or only the i-th one if i is given."""
    if i is None:
        gf = ng.GridFunction(
            span.fes, multidim=span.dim, name=name,
            autoupdate=span.fes.autoupdate,
        )
        for k in range(span.dim):
            gf.vecs[k].data = span._mv[k]
    else:
        gf = ng.GridFunction(
            span.fes, name=name, autoupdate=span.fes.autoupdate
        )
        gf.vec.data = span._mv[i]
    return gf


def draw(span: NGSpan, name: str = "eigenmode"):
    """Draw the span in an NGSolve GUI/webgui context."""
    ng.Draw(gridfun(span, name))
