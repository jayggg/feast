"""NGSolve backend: finite element spans and resolvents.

Requires NGSolve (`pip install feast-eig[ngsolve]`). This module import
fails with a clear message when NGSolve is absent; the feast core never
imports it unconditionally.
"""

try:
    import ngsolve  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The feast NGSolve backend requires NGSolve. "
        "Install it with: pip install feast-eig[ngsolve]"
    ) from exc

from .poly import NGPolyResolvent, component, first_component
from .resolvent import NGCallbackResolvent, NGPencilResolvent
from .span import NGSpan

__all__ = [
    "NGSpan",
    "NGPencilResolvent",
    "NGCallbackResolvent",
    "NGPolyResolvent",
    "first_component",
    "component",
]
