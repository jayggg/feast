"""NGSolve backend tests. Run with:  pytest -m ngsolve

Skipped automatically when NGSolve is not installed.
"""

import numpy as np
import pytest

ng = pytest.importorskip("ngsolve")
netgen = pytest.importorskip("netgen")

from feast import Contour, feast  # noqa: E402
from feast.backends.ngsolve import (  # noqa: E402
    NGCallbackResolvent,
    NGPencilResolvent,
    NGSpan,
)
from feast.projector import Resolvent  # noqa: E402
from feast.span import Span  # noqa: E402

pytestmark = pytest.mark.ngsolve

PI2 = np.pi**2
# -Δu = lam u on unit square, Dirichlet: lam = (m² + n²) π²
EXACT = np.array([2, 5, 5]) * PI2  # 5π² is a degenerate pair
FE_RTOL = 2e-3  # discretization error bound for order 3, maxh 0.15


@pytest.fixture(scope="module")
def square():
    """H1 space and assembled (A, B) for -Δ on the unit square."""
    from netgen.geom2d import unit_square

    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.15))
    fes = ng.H1(
        mesh, order=3, dirichlet="left|right|top|bottom", complex=True
    )
    u, v = fes.TnT()
    a = ng.BilinearForm(ng.grad(u) * ng.grad(v) * ng.dx).Assemble()
    b = ng.BilinearForm(u * v * ng.dx).Assemble()
    return fes, a.mat, b.mat


def test_ngspan_protocol_and_algebra(square):
    fes, _, _ = square
    s = NGSpan.random(fes, 3, seed=1)
    assert isinstance(s, Span)
    ref = s.to_numpy()

    v = np.array([[1, 0], [1, 1], [0, 2j]], dtype=complex)
    s.rotate(v)
    assert s.dim == 2
    assert np.allclose(s.to_numpy(), ref @ v, atol=1e-12)

    s.scale(0, 2.0)
    s.remove([1])
    assert s.dim == 1
    assert np.allclose(s.to_numpy()[:, 0], 2 * (ref @ v)[:, 0], atol=1e-12)

    s2 = NGSpan.random(fes, 2, seed=5)
    s3 = NGSpan.random(fes, 2, seed=5)
    assert np.array_equal(s2.to_numpy(), s3.to_numpy())

    # remove-all then append (driver's default-yl pattern)
    s2.remove([0, 1])
    assert s2.dim == 0
    s2.append_random(2, np.random.default_rng(0))
    assert s2.dim == 2


def test_pencil_resolvent_protocol(square):
    fes, A, B = square
    assert isinstance(NGPencilResolvent(fes, A, B), Resolvent)


def test_hermitian_square_eigenvalues_with_degenerate_pair(square):
    fes, A, B = square
    R = NGPencilResolvent(fes, A, B)
    # circle around {2π², 5π², 5π²} but excluding 8π² = 78.96
    C = Contour.circle(center=40.0, radius=35.0, npts=8)
    res = feast(
        R, C, NGSpan.random(fes, 6, seed=10),
        mode="hermitian", stop_tol=1e-11, niterations=30,
    )
    assert res.converged
    lam = np.sort(res.eigenvalues.real)[:3]
    assert np.allclose(lam, EXACT, rtol=FE_RTOL)


def test_hermitian_reduce_sym(square):
    """Same eigenvalues with half the factorizations."""
    fes, A, B = square
    R = NGPencilResolvent(fes, A, B)
    C = Contour.circle(center=40.0, radius=35.0, npts=8)
    res = feast(
        R, C, NGSpan.random(fes, 6, seed=10),
        mode="hermitian", reduce_sym=True, stop_tol=1e-11, niterations=30,
    )
    assert res.converged
    assert len(R._inv) == 4  # 8 nodes, 4 conjugate pairs
    assert np.allclose(np.sort(res.eigenvalues.real)[:3], EXACT,
                       rtol=FE_RTOL)


def test_biorth_finds_ground_state(square):
    fes, A, B = square
    R = NGPencilResolvent(fes, A, B)
    # circle around 2π² = 19.74 only
    C = Contour.circle(center=20.0, radius=10.0, npts=8)
    res = feast(
        R, C, NGSpan.random(fes, 3, seed=10),
        mode="biorth", stop_tol=1e-11, niterations=50, seed=1,
    )
    assert res.converged
    assert len(res.eigenvalues) == 1
    assert np.allclose(res.eigenvalues.real, 2 * PI2, rtol=FE_RTOL)
    assert abs(res.eigenvalues.imag[0]) < 1e-8


def test_callback_resolvent_parity(square):
    """Factory callbacks reproduce the assembled pencil backend."""
    fes, A, B = square

    class V1StyleResolvent:
        """Supply in-place solves and reduced matrices for a fixed shift."""

        def __init__(self, z):
            self._p = NGPencilResolvent(fes, A, B)
            self._z = z

        def act(self, v, Rv, workspace=None):
            Rv._mv[:] = self._p.apply(self._z, v)._mv

        def adj(self, v, Rv, workspace=None):
            Rv._mv[:] = self._p.apply_adj(self._z, v)._mv

        def rayleigh_nsa(self, ql, qr, qAq=not None, qBq=not None,
                         workspace=None):
            h_a, h_b = self._p.rayleigh(ql, qr)
            return (h_a if qAq is not None else None,
                    h_b if qBq is not None else None)

    R = NGCallbackResolvent(lambda z: V1StyleResolvent(z))
    C = Contour.circle(center=20.0, radius=10.0, npts=8)
    res = feast(
        R, C, NGSpan.random(fes, 3, seed=10),
        mode="biorth", stop_tol=1e-11, niterations=50, seed=1,
    )
    assert res.converged
    assert np.allclose(res.eigenvalues.real, 2 * PI2, rtol=FE_RTOL)


def test_native_orthogonalize_custom_mass_contract(square):
    fes, _, mass = square
    span = NGSpan.random(fes, 4, seed=12)
    original = span.to_numpy()
    r = np.asarray(span._mv.Orthogonalize(ipmat=mass))
    q = span.to_numpy()
    mq = span.new()
    mq._mv[:] = mass * span._mv
    assert np.allclose(q @ r, original, atol=1e-12)
    assert np.allclose(q.conj().T @ mq.to_numpy(), np.eye(4), atol=1e-12)
    assert np.allclose(np.tril(r, -1), 0)


def test_native_append_orthogonalize_complex_contract(square):
    fes, _, _ = square
    span = NGSpan.random(fes, 4, seed=12)
    original = span.to_numpy()
    q = ng.MultiVector(span._gf.vec, 0)
    r = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        r[:j + 1, j] = np.asarray(
            q.AppendOrthogonalize(span._mv[j], iterations=2)
        )
    data = np.column_stack([v.FV().NumPy() for v in q])
    assert np.allclose(data @ r, original, atol=1e-12)
    assert np.allclose(data.conj().T @ data, np.eye(4), atol=1e-12)
    assert np.array_equal(span.to_numpy(), original)


def test_ngspan_qr_custom_mass_with_dependent_columns(square):
    fes, _, mass = square
    original = NGSpan.random(fes, 4, seed=12).to_numpy()
    original[:, 0] = 0
    original[:, 2] = (2 + 3j) * original[:, 1]
    span = NGSpan.from_numpy(fes, original)
    r = span.qr(ipmat=mass)
    q = span.to_numpy()
    mq = span.new()
    mq._mv[:] = mass * span._mv
    assert span.dim == 2
    assert np.allclose(q @ r, original, atol=1e-11)
    assert np.allclose(q.conj().T @ mq.to_numpy(), np.eye(2), atol=1e-12)
    assert np.all(q[~np.array(fes.FreeDofs())] == 0)


@pytest.mark.parametrize("mode", ["hermitian", "right"])
def test_ng_rank_reduction_for_dependent_start(square, mode):
    fes, a, b = square
    data = NGSpan.random(fes, 2, seed=1).to_numpy()
    data = np.column_stack([np.zeros(fes.ndof), data, (2 + 3j) * data[:, 0]])
    r = NGPencilResolvent(fes, a, b)
    result = feast(
        r, Contour.circle(20.0, 3.0, npts=16),
        NGSpan.from_numpy(fes, data), mode=mode,
    )
    assert result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [2 * PI2], rtol=FE_RTOL)
    residual = r.residual(result.eigenvalues, result.right)
    norms = np.linalg.norm(result.right.to_numpy(), axis=0)
    assert np.max(residual / norms) < 1e-8


def test_ng_unrecoverable_refresh_returns_matching_pairs(square):
    fes, a, b = square
    r = NGPencilResolvent(fes, a, b)
    result = feast(
        r, Contour.circle(20.0, 3.0, npts=16),
        NGSpan.random(fes, 3, seed=10), mode="biorth", seed=1,
        expected_dim=2, nrefresh=1, niterations=5,
    )
    assert not result.converged
    assert result.right.dim == result.left.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [2 * PI2], rtol=FE_RTOL)
