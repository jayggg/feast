"""NGPolyResolvent tests. Run with:  pytest -m ngsolve

Quadratic eigenproblem A(lam) = K + lam*0 - lam^2 M on the unit square
(K = Dirichlet stiffness, M = mass), i.e. K x = lam^2 M x, so
lam = ±sqrt((m^2 + n^2) pi^2). A search contour in the right half plane
picks out the positive square roots.
"""

import numpy as np
import pytest

ng = pytest.importorskip("ngsolve")
netgen = pytest.importorskip("netgen")

from feast import Contour, feast  # noqa: E402
from feast.backends.ngsolve import (  # noqa: E402
    NGPolyResolvent,
    first_component,
)
from feast.projector import Resolvent  # noqa: E402

pytestmark = pytest.mark.ngsolve

PI = np.pi
# lam = sqrt((m²+n²)) * pi: sqrt2, sqrt5 (double), then sqrt8 = 2.828...
EXACT = PI * np.array([np.sqrt(2), np.sqrt(5), np.sqrt(5)])
FE_RTOL = 2e-3


@pytest.fixture(scope="module")
def poly_resolvent():
    from netgen.geom2d import unit_square

    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.15))
    fes = ng.H1(
        mesh, order=3, dirichlet="left|right|top|bottom", complex=True
    )
    u, v = fes.TnT()
    k = ng.BilinearForm(ng.grad(u) * ng.grad(v) * ng.dx).Assemble()
    negm = ng.BilinearForm(-1 * u * v * ng.dx).Assemble()
    # zero matrix with the mass sparsity ("0*u*v*dx" would fold away
    # the trial/test functions and fail to assemble)
    zero = negm.mat.CreateMatrix()
    zero.AsVector().data = 0.0 * negm.mat.AsVector()
    # A(lam) = K + lam*0 + lam^2 (-M);  -A[2] = M is s.p.d.
    return fes, NGPolyResolvent([k.mat, zero, negm.mat], fes)


def test_poly_resolvent_protocol(poly_resolvent):
    _, R = poly_resolvent
    assert isinstance(R, Resolvent)
    assert R.degree == 2


def test_degree_one_rejected(poly_resolvent):
    fes, R = poly_resolvent
    with pytest.raises(ValueError):
        NGPolyResolvent(R.coefs[:2], fes)


def test_quadratic_eigenvalues(poly_resolvent):
    fes, R = poly_resolvent
    # circle around {sqrt2, sqrt5, sqrt5}*pi ~ {4.44, 7.02, 7.02},
    # excluding sqrt8*pi = 8.89 and all negative roots
    C = Contour.circle(center=5.7, radius=2.9, npts=8)
    res = feast(
        R, C, R.random_span(6, seed=10),
        mode="biorth", stop_tol=1e-11, niterations=50, seed=1,
    )
    assert res.converged
    lam = np.sort(res.eigenvalues.real)[:3]
    assert np.allclose(lam, EXACT, rtol=FE_RTOL)
    assert np.max(np.abs(res.eigenvalues.imag)) < 1e-6


def test_first_component_is_physical_eigenvector(poly_resolvent):
    fes, R = poly_resolvent
    C = Contour.circle(center=4.4, radius=1.5, npts=8)
    res = feast(
        R, C, R.random_span(3, seed=10),
        mode="biorth", stop_tol=1e-11, niterations=50, seed=1,
    )
    assert res.converged and len(res.eigenvalues) == 1
    x = first_component(res.right, fes)
    assert x.dim == 1 and x.n == fes.ndof
    # physical residual ||A(lam) x|| small relative to ||x||
    r = R.residual(res.eigenvalues, res.right)
    xnorm = np.linalg.norm(x.to_numpy())
    assert r[0] / xnorm < 1e-6
