import numpy as np
import pytest

from feast import (
    Contour,
    SpectralProjectorApprox,
    SymmetryReducedProjector,
    feast,
)
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan


def real_sym_matrix(n=30, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    return (A + A.T) / 2


def test_reduced_apply_matches_full():
    A = real_sym_matrix()
    exact = np.sort(np.linalg.eigvalsh(A))
    C = Contour.circle(center=exact[0], radius=0.5 * (exact[1] - exact[0]),
                       npts=8)
    y = NumpySpan.random(A.shape[0], 3, seed=1)
    full = SpectralProjectorApprox(MatrixResolvent(A), C).apply(y)
    red = SymmetryReducedProjector(MatrixResolvent(A), C).apply(y)
    assert np.allclose(full.data, red.data, atol=1e-12)


def test_reduced_halves_factorizations():
    A = real_sym_matrix()
    C = Contour.circle(center=0.0, radius=1.0, npts=8)
    R = MatrixResolvent(A)
    SymmetryReducedProjector(R, C).apply(NumpySpan.random(A.shape[0], 2))
    assert len(R._lu) == 4  # 8 nodes, 4 conjugate pairs


def test_feast_reduce_sym_converges():
    A = real_sym_matrix()
    exact = np.sort(np.linalg.eigvalsh(A))
    C = Contour.circle(
        center=0.5 * (exact[0] + exact[2]),
        radius=0.55 * (exact[2] - exact[0]),
        npts=16,
    )
    res = feast(
        MatrixResolvent(A), C, NumpySpan.random(A.shape[0], 6, seed=0),
        mode="hermitian", reduce_sym=True, stop_tol=1e-12,
    )
    assert res.converged
    assert np.allclose(np.sort(res.eigenvalues.real)[:3], exact[:3],
                       atol=1e-8)


def test_reduce_sym_rejected_for_nonhermitian():
    with pytest.raises(ValueError):
        feast(
            MatrixResolvent(np.eye(2)),
            Contour.circle(0, 1, npts=4),
            NumpySpan.random(2, 1),
            mode="biorth",
            reduce_sym=True,
        )


def test_asymmetric_nodes_rejected():
    z = np.exp(1j * np.linspace(0.1, 2, 5))
    C = Contour.custom(z, np.ones(5), lambda lam: np.abs(lam) < 1)
    with pytest.raises(ValueError):
        SymmetryReducedProjector(MatrixResolvent(np.eye(3)), C)


def test_conjugate_inplace():
    s = NumpySpan.random(5, 2, seed=0)
    ref = s.data.copy()
    s.conjugate()
    assert np.array_equal(s.data, ref.conj())
