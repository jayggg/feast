"""MatrixPolyResolvent: polynomial eigenproblems with NumPy/SciPy
coefficients, checked against a dense eigensolve of the companion
linearization."""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixPolyResolvent, NumpySpan
from feast.projector import Resolvent


def companion(coefs):
    """Dense first companion pencil (AA, BB) of sum z^i A_i."""
    coefs = [np.asarray(sp.csc_matrix(a).todense(), dtype=complex)
             for a in coefs]
    d, n = len(coefs) - 1, coefs[0].shape[0]
    aa = np.zeros((d * n, d * n), dtype=complex)
    bb = np.eye(d * n, dtype=complex)
    for i in range(d - 1):
        aa[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = np.eye(n)
    for i in range(d):
        aa[(d - 1) * n:, i * n:(i + 1) * n] = coefs[i]
    bb[(d - 1) * n:, (d - 1) * n:] = -coefs[d]
    return aa, bb


def random_coefs(n, d, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            for _ in range(d + 1)]


@pytest.mark.parametrize("d", [2, 3, 4])
def test_apply_matches_dense_companion(d):
    coefs = random_coefs(5, d, seed=d)
    R = MatrixPolyResolvent(coefs)
    aa, bb = companion(coefs)
    x = R.random_span(3, seed=0)
    z = 0.3 + 0.7j
    m = z * bb - aa
    assert np.allclose(R.apply(z, x).data,
                       np.linalg.solve(m, bb @ x.data))
    assert np.allclose(R.apply_adj(z, x).data,
                       np.linalg.solve(m.conj().T, bb.conj().T @ x.data))
    h_a, h_b = R.rayleigh(x, x)
    xd = x.data
    assert np.allclose(h_a, xd.conj().T @ aa @ xd)
    assert np.allclose(h_b, xd.conj().T @ bb @ xd)
    assert np.allclose(R.gram(x, x), h_b)


def test_protocol_and_degree_check():
    R = MatrixPolyResolvent(random_coefs(4, 2, seed=1))
    assert isinstance(R, Resolvent)
    assert R.degree == 2
    with pytest.raises(ValueError):
        MatrixPolyResolvent(random_coefs(4, 1, seed=1))


def _inside(lam, center, radius):
    return lam[np.abs(lam - center) < radius]


@pytest.mark.parametrize("d", [2, 3])
def test_biorth_eigenpairs(d):
    n = 12
    coefs = random_coefs(n, d, seed=10 + d)
    R = MatrixPolyResolvent(coefs)
    exact = sla.eig(*companion(coefs), right=False)
    # circle about 0 enclosing the k = 2..6 eigenvalues of smallest
    # modulus, k chosen at the widest modulus gap. Random coefficients
    # give poorly separated spectra, so 32 nodes sharpen the filter
    # enough for eigenvectors (not just eigenvalues) to converge.
    r = np.sort(np.abs(exact))
    k = 2 + int(np.argmax(r[2:7] / r[1:6]))
    center, radius = 0.0, np.sqrt(r[k - 1] * r[k])
    want = _inside(exact, center, radius)
    assert len(want) == k
    res = feast(R, Contour.circle(center=center, radius=radius, npts=32),
                R.random_span(len(want) + 3, seed=3), mode="biorth",
                stop_tol=1e-12, niterations=80, seed=1)
    assert res.converged
    assert np.allclose(np.sort_complex(res.eigenvalues),
                       np.sort_complex(want), atol=1e-9)
    # first block of right vectors, last block of left vectors, are
    # null vectors of P(lambda) and P(lambda)^H (relative residuals)
    x, y = R.first(res.right).data, R.last(res.left).data
    for j, lam in enumerate(res.eigenvalues):
        p = R.pol(lam).toarray()
        scale = np.linalg.norm(p, 2)
        assert np.linalg.norm(p @ x[:, j]) < (
            1e-9 * scale * np.linalg.norm(x[:, j]))
        assert np.linalg.norm(p.conj().T @ y[:, j]) < (
            1e-9 * scale * np.linalg.norm(y[:, j]))
    assert R.residual(res.eigenvalues, res.right, res.left).shape == (k,)


def test_hermitian_quadratic_sparse():
    # P(lam) = K - lam^2 M with K = 1D Dirichlet Laplacian, M = I:
    # lam = ±sqrt(mu) for the eigenvalues mu of K.
    n = 40
    K = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
                 [-1, 0, 1], format="csc")
    M = sp.identity(n, format="csc")
    R = MatrixPolyResolvent([K, sp.csc_matrix((n, n)), -M])
    mu = 2 - 2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))
    exact = np.sqrt(mu)
    want = exact[(exact > 0.5) & (exact < 1.0)]
    C = Contour.circle(center=0.75, radius=0.25, npts=16)
    res = feast(R, C, R.random_span(len(want) + 3, seed=4),
                mode="biorth", stop_tol=1e-12, niterations=80, seed=1)
    assert res.converged
    assert np.allclose(np.sort(res.eigenvalues.real), np.sort(want),
                       atol=1e-9)


def test_singular_leading_coefficient():
    # A_d singular: infinite eigenvalues must be filtered, finite ones
    # inside the contour still found.
    n = 10
    coefs = random_coefs(n, 2, seed=7)
    coefs[2][:, : n // 2] = 0.0
    R = MatrixPolyResolvent(coefs)
    aa, bb = companion(coefs)
    exact = sla.eig(aa, bb, right=False)
    exact = exact[np.isfinite(exact)]
    center, radius = 0.0, 1.0
    want = _inside(exact, center, radius)
    assert len(want) >= 1
    res = feast(R, Contour.circle(center=center, radius=radius, npts=16),
                R.random_span(len(want) + 3, seed=5), mode="biorth",
                stop_tol=1e-12, niterations=80, seed=1)
    assert res.converged
    assert np.allclose(np.sort_complex(res.eigenvalues),
                       np.sort_complex(want), atol=1e-8)


def test_component_shapes():
    R = MatrixPolyResolvent(random_coefs(6, 3, seed=2))
    y = R.random_span(4, seed=0)
    assert isinstance(y, NumpySpan) and y.n == 18
    assert R.first(y).data.shape == (6, 4)
    assert np.array_equal(R.component(y, 1).data, y.data[6:12])
    assert np.array_equal(R.last(y).data, y.data[12:])
