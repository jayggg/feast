"""Contours and quadrature rules for the FEAST contour integral.

A :class:`Contour` bundles quadrature nodes ``z`` and weights ``w``
approximating (2πi)⁻¹ ∮ f(z) dz ≈ Σₖ wₖ f(zₖ), a membership test
:meth:`within`, and the rational filter :meth:`filter` they induce.
Everything here is pure numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Contour"]


@dataclass(frozen=True)
class Contour:
    """Quadrature rule on a contour in the complex plane.

    Do not call the constructor directly; use the factory classmethods
    :meth:`circle`, :meth:`ellipse`, or :meth:`custom`.

    Attributes
    ----------
    z, w : complex ndarray
        Quadrature nodes and weights.
    within : callable
        ``within(lam)`` -> boolean array, True where ``lam`` is enclosed.
    name : str
        Human-readable rule name.
    scale : float
        Characteristic magnitude of enclosed eigenvalues; used for
        relative eigenvalue-distance measures.
    """

    z: np.ndarray
    w: np.ndarray
    within: Callable[[np.ndarray], np.ndarray]
    name: str
    scale: float
    center: complex | None = field(default=None)
    radius: float | None = field(default=None)

    @property
    def npts(self) -> int:
        return len(self.z)

    # Factories ########################################################

    @classmethod
    def circle(
        cls,
        center: complex,
        radius: float,
        npts: int = 8,
        rule: str = "trapezoid_shifted",
        boundary_tol: float = 1e-12,
    ) -> "Contour":
        """Circular contour ``abs(z - center) = radius``.

        rule is one of ``trapezoid_shifted`` (default; nodes shifted off
        the real axis when center is real), ``trapezoid`` or ``snug``
        (unshifted nodes on a smaller circle with radius
        ``radius / 2**(1/npts)``).

        boundary_tol: relative margin of the membership test. A point
        counts as enclosed when
        ``abs(z - center) < radius*(1 + boundary_tol)``,
        so an eigenvalue lying on the contour up to rounding error is
        kept rather than trimmed platform-dependently. (Eigenvalues ON
        the contour remain a badly-posed input; the margin only makes
        the failure mode deterministic.)
        """
        if npts > 1 and npts % 2:
            npts += 1
        theta = np.linspace(0, 2 * np.pi, num=npts, endpoint=False)
        if rule == "trapezoid_shifted":
            theta += np.pi / npts
            rho = radius
        elif rule == "trapezoid":
            rho = radius
        elif rule == "snug":
            rho = radius / 2 ** (1.0 / npts)
        else:
            raise ValueError(f"Unknown circle rule {rule!r}")
        z = center + rho * np.exp(1j * theta)
        w = rho * np.exp(1j * theta) / npts

        def within(lam: np.ndarray) -> np.ndarray:
            return np.abs(lam - center) < radius * (1 + boundary_tol)

        c = complex(center)
        scale = (
            radius
            if abs(c.imag) == 0 and (c.real - radius) * (c.real + radius) < 0
            else abs(c) + radius
        )
        log.info(
            "Contour: circle rule=%s center=%s radius=%g npts=%d",
            rule, center, radius, npts,
        )
        return cls(z, w, within, f"circle_{rule}", scale,
                   center=c, radius=radius)

    @classmethod
    def ellipse(
        cls,
        center: complex,
        radius: float,
        rhoinv: float,
        npts: int = 8,
        rule: str = "trapezoid_shifted",
        boundary_tol: float = 1e-12,
    ) -> "Contour":
        """Bernstein-ellipse contour with parameter rho = 1/rhoinv,
        intersecting the real axis at center ± radius. rule is
        ``trapezoid_shifted`` or ``snug``.

        boundary_tol: relative margin of the membership test. With
        q(z) = (Re(z-c)/r)² + (Im(z-c)/(r·e))² — the level function
        whose q = 1 level set is the ellipse — a point counts as
        enclosed when q < 1 + boundary_tol. Since q - 1 vanishes
        linearly at the boundary, this is a relative margin on the
        boundary distance, the ellipse analogue of the circle's
        ``abs(z-c) < r(1 + tol)``."""
        if not 0 < rhoinv < 1:
            raise ValueError("rhoinv must be in (0, 1)")
        if npts > 1 and npts % 2:
            npts += 1
        rho = 1.0 / rhoinv
        theta = np.linspace(0, 2 * np.pi, num=npts, endpoint=False)
        if rule == "trapezoid_shifted":
            theta += np.pi / npts
            eta = 2 * radius / (rho + rhoinv)
        elif rule == "snug":
            trho = 1 + rho**npts + rhoinv**npts
            a = (trho + np.sqrt(trho * trho - 1)) ** (1.0 / npts)
            eta = 2 * radius / (a + 1.0 / a)
        else:
            raise ValueError(f"Unknown ellipse rule {rule!r}")
        z = center + 0.5 * eta * (
            rho * np.exp(1j * theta) + rhoinv * np.exp(-1j * theta)
        )
        w = (0.5 * eta / npts) * (
            rho * np.exp(1j * theta) - rhoinv * np.exp(-1j * theta)
        )
        e = (rho - rhoinv) / (rho + rhoinv)

        def within(lam: np.ndarray) -> np.ndarray:
            z0 = np.asarray(lam) - center
            q = (z0.real / radius) ** 2 + (z0.imag / (radius * e)) ** 2
            return q < 1 + boundary_tol

        c = complex(center)
        scale = (
            radius
            if abs(c.imag) == 0 and (c.real - radius) * (c.real + radius) < 0
            else abs(c) + radius
        )
        log.info(
            "Contour: ellipse rule=%s center=%s radius=%g rho=%g npts=%d",
            rule, center, radius, rho, npts,
        )
        return cls(z, w, within, f"ellipse_{rule}", scale,
                   center=c, radius=radius)

    @classmethod
    def custom(
        cls,
        z: np.ndarray,
        w: np.ndarray,
        within: Callable[[np.ndarray], np.ndarray],
        name: str = "custom",
    ) -> "Contour":
        """Arbitrary user-supplied nodes, weights, and membership test.

        No boundary tolerance is applied here: for a custom contour the
        supplied ``within`` is authoritative, and any margin policy is
        the caller's to build into it."""
        z = np.asarray(z, dtype=complex)
        w = np.asarray(w, dtype=complex)
        if z.shape != w.shape:
            raise ValueError("z and w must have the same shape")
        return cls(z, w, within, name, scale=float(abs(z.mean())))

    # Derived quantities ###############################################

    def filter(self, zeta: np.ndarray) -> np.ndarray:
        """Rational filter f(ζ) = Σₖ wₖ / (zₖ - ζ) induced by the rule.

        Approximately 1 well inside the quadrature curve and decaying
        far outside. Boundary behavior depends on the rule; the filter
        has poles at the quadrature nodes.
        """
        zeta = np.asarray(zeta, dtype=complex)
        return np.sum(
            self.w[:, None] / (self.z[:, None] - zeta[None, :]), axis=0
        )

    def __repr__(self) -> str:
        return (
            f"Contour({self.name}, npts={self.npts}, center={self.center}, "
            f"radius={self.radius})"
        )
