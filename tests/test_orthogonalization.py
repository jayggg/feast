import numpy as np
import pytest

from feast.backends.numpy_backend import NumpySpan
from feast.subspace import orthonormalize


@pytest.fixture(params=[
    "numpy", pytest.param("ngsolve", marks=pytest.mark.ngsolve),
])
def backend(request):
    if request.param == "numpy":
        return NumpySpan, lambda s: s.data, np.ones(40, dtype=bool)
    ng = pytest.importorskip("ngsolve")
    from netgen.geom2d import unit_square

    from feast.backends.ngsolve import NGSpan

    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.25))
    fes = ng.H1(
        mesh, order=2, complex=True, dirichlet="left|right|top|bottom",
    )
    return (
        lambda data: NGSpan.from_numpy(fes, data),
        lambda s: s.to_numpy(),
        np.array(fes.FreeDofs()),
    )


@pytest.mark.parametrize("kind", ["full", "dependent", "zero", "wide"])
def test_qr_reconstruction_and_orthogonality(backend, kind):
    make, array, free = backend
    rng = np.random.default_rng(15)
    n = len(free)
    m = n + 2 if kind == "wide" else 4
    data = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    data[~free] = 0
    if kind == "dependent":
        data[:, 0] = 0
        data[:, 2] = (2 + 3j) * data[:, 1]
    elif kind == "zero":
        data[:] = 0
    span = make(data.copy())
    r = span.qr()
    q = array(span)
    assert r.shape == (span.dim, m)
    assert np.allclose(q @ r, data, atol=1e-12)
    assert np.allclose(q.conj().T @ q, np.eye(span.dim), atol=1e-12)
    # A dense QR may complete Q outside the input span; truncation must
    # remove those directions before FEAST uses the basis.
    original = make(data.copy())
    orthonormalize(original)
    reduced = array(original)
    assert original.dim == np.linalg.matrix_rank(data)
    assert np.allclose(reduced[~free], 0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-40, 1.0, 1e40])
@pytest.mark.parametrize("rank_tol, rank", [(1e-12, 2), (1e-9, 1)])
def test_relative_rank_threshold_without_squaring(
    backend, scale, rank_tol, rank,
):
    make, array, free = backend
    rng = np.random.default_rng(22)
    vectors = rng.standard_normal((len(free), 3))
    vectors = vectors + 1j * rng.standard_normal(vectors.shape)
    vectors[~free] = 0
    u, _ = np.linalg.qr(vectors)
    v, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    data = scale * (u * np.array([1, 1e-10, 1e-15])) @ v.T
    span = make(data)
    orthonormalize(span, rank_tol)
    q = array(span)
    assert span.dim == rank
    assert np.allclose(q.conj().T @ q, np.eye(rank), atol=1e-12)
    assert np.linalg.norm(data - q @ (q.conj().T @ data)) / scale < rank_tol


def test_empty_span_reduction(backend):
    make, array, free = backend
    span = make(np.zeros((len(free), 0)))
    assert len(orthonormalize(span)) == 0
    assert array(span).shape == (len(free), 0)


@pytest.mark.parametrize("rank_tol", [0, -1, 1, np.inf, np.nan])
def test_invalid_rank_tolerance(rank_tol):
    with pytest.raises(ValueError, match="rank_tol"):
        orthonormalize(NumpySpan(np.eye(2)), rank_tol)
