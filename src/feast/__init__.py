"""FEAST contour-integral eigensolver.

Core (this package) is pure numpy/scipy. Discretization backends live in
``feast.backends``; the NGSolve backend is imported only on demand so the
core works without NGSolve installed.
"""

__version__ = "2.0.0.dev0"

from .contour import Contour
from .driver import FeastHistory, FeastResult, feast
from .estimator import (
    EigenspaceErrorEstimate,
    SourceErrorEstimator,
    estimate_eigenspace_error,
)
from .projector import (
    SPA,
    Resolvent,
    SpectralProjectorApprox,
    SymmetryReducedProjector,
)
from .span import Span

__all__ = [
    "Contour",
    "Span",
    "Resolvent",
    "SpectralProjectorApprox",
    "SPA",
    "SymmetryReducedProjector",
    "feast",
    "FeastResult",
    "FeastHistory",
    "SourceErrorEstimator",
    "EigenspaceErrorEstimate",
    "estimate_eigenspace_error",
    "__version__",
]
