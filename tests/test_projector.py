import numpy as np

from feast import Contour
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan
from feast.projector import SPA, Resolvent, SpectralProjectorApprox


def test_matrixresolvent_satisfies_protocol():
    assert isinstance(MatrixResolvent(np.eye(3)), Resolvent)


def test_spa_alias():
    assert SPA is SpectralProjectorApprox


def test_projector_projects_diagonal():
    # A = diag(1, 2, 10): contour around {1, 2} -> P ~ diag(1, 1, 0)
    A = np.diag([1.0, 2.0, 10.0])
    P = SpectralProjectorApprox(
        MatrixResolvent(A), Contour.circle(1.5, 1.0, npts=32)
    )
    y = NumpySpan(np.eye(3))
    py = P.apply(y)
    assert np.allclose(py.data, np.diag([1.0, 1.0, 0.0]), atol=1e-6)


def test_projector_adjoint_matches_right_apply_for_normal_A():
    # For A normal (diagonal), left application = right application conj
    A = np.diag([1.0, 2.0, 10.0])
    P = SpectralProjectorApprox(
        MatrixResolvent(A), Contour.circle(1.5, 1.0, npts=32)
    )
    yl = NumpySpan.random(3, 2, seed=1)
    ql = P.apply_adj(yl)
    # ql^H = yl^H P  =>  ql = P^H yl; P^H here = P (real projector matrix)
    expect = P.apply(NumpySpan(np.eye(3))).data.conj().T @ yl.data
    assert np.allclose(ql.data, expect, atol=1e-8)


def test_generalized_pencil():
    # A x = lam B x with B = 2 I: eigenvalues halved
    A = np.diag([2.0, 4.0, 20.0])
    B = 2.0 * np.eye(3)
    P = SpectralProjectorApprox(
        MatrixResolvent(A, B), Contour.circle(1.5, 1.0, npts=32)
    )
    py = P.apply(NumpySpan(np.eye(3)))
    assert np.allclose(py.data, np.diag([1.0, 1.0, 0.0]), atol=1e-6)


def test_factorization_cache_reused():
    R = MatrixResolvent(np.diag([1.0, 3.0]))
    C = Contour.circle(1.0, 0.5, npts=4)
    P = SpectralProjectorApprox(R, C)
    y = NumpySpan.random(2, 1, seed=0)
    P.apply(y)
    assert len(R._lu) == 4
    P.apply(y)  # second apply: no new factorizations
    assert len(R._lu) == 4
