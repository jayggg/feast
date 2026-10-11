"""Stochastic eigenvalue count estimate and FEAST's completeness check."""

import logging

import numpy as np
import pytest
import scipy.sparse as sp

from feast import (
    Contour,
    EigenvalueCountEstimate,
    estimate_eigenvalue_count,
    feast,
)
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan


def laplacian(n=200):
    h = 1.0 / (n + 1)
    A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h**2
    lam = 4 / h**2 * np.sin(np.arange(1, n + 1) * np.pi * h / 2) ** 2
    return A, lam


A, LAM = laplacian()
C = Contour.circle(center=50.0, radius=45.0, npts=8)   # 3 inside


def test_numpy_span_dot():
    a = NumpySpan.random(6, 3, seed=1)
    b = NumpySpan.random(6, 2, seed=2)
    assert np.allclose(a.dot(b), a.data.conj().T @ b.data)


def test_estimate_is_unbiased_for_filter_trace():
    R = MatrixResolvent(A)
    est = estimate_eigenvalue_count(R, C, NumpySpan.random(200, 1),
                                    nprobes=400, seed=1)
    exact = C.filter(LAM).real.sum()          # ~2.48: a weighted count
    assert abs(est.estimate - exact) < 4 * est.stderr
    assert est.stderr < 0.1
    assert est.unexplained is None and not est.suggests_missing()


def test_estimate_for_nonnormal_generalized_pencil():
    rng = np.random.default_rng(0)
    m = 60
    T = (np.triu(rng.standard_normal((m, m)), 1) * 0.3
         + np.diag(np.linspace(1, 20, m)))
    B = np.eye(m) + 0.1 * np.diag(rng.random(m))
    R = MatrixResolvent(T, B)
    contour = Contour.circle(3.0, 2.5, npts=16)
    lam = np.linalg.eigvals(np.linalg.solve(B, T))
    est = estimate_eigenvalue_count(R, contour, NumpySpan.random(m, 1),
                                    nprobes=400, seed=3)
    assert abs(est.estimate - contour.filter(lam).real.sum()) \
        < 4 * est.stderr


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_complete_result_leaves_nothing_unexplained(mode):
    result = feast(MatrixResolvent(A), C, NumpySpan.random(200, 5, seed=1),
                   mode=mode, seed=1, count_probes=16)
    est = result.count_estimate
    assert isinstance(est, EigenvalueCountEstimate)
    assert result.converged and len(result.eigenvalues) == 3
    assert np.isclose(est.found_weight,
                      C.filter(result.eigenvalues).real.sum())
    assert abs(est.unexplained) < 0.01
    assert not est.suggests_missing()


def test_missed_eigenvalue_is_flagged(caplog):
    # Two vectors cannot capture three enclosed eigenvalues.
    with caplog.at_level(logging.WARNING, logger="feast"):
        result = feast(MatrixResolvent(A), C,
                       NumpySpan.random(200, 2, seed=1),
                       mode="hermitian", seed=1, count_probes=16)
    est = result.count_estimate
    assert len(result.eigenvalues) == 2
    assert est.suggests_missing()
    assert 0.4 < est.unexplained < 1.2      # one eigenvalue, weight ~0.7
    assert "completeness check" in caplog.text


def test_count_probes_do_not_change_the_iteration():
    for seed in (7, np.random.default_rng(7)):
        runs = []
        for probes in (0, 16):
            s = seed if isinstance(seed, int) else np.random.default_rng(7)
            runs.append(feast(MatrixResolvent(A), C,
                              NumpySpan.random(200, 5, seed=1),
                              mode="biorth", seed=s, count_probes=probes))
        assert runs[0].count_estimate is None
        assert np.array_equal(runs[0].eigenvalues, runs[1].eigenvalues)
        assert runs[0].iterations == runs[1].iterations


def test_count_with_symmetry_reduction():
    result = feast(MatrixResolvent(A), C, NumpySpan.random(200, 5, seed=1),
                   mode="hermitian", seed=1, count_probes=16, reduce_sym=True)
    assert abs(result.count_estimate.unexplained) < 0.01


def test_input_validation():
    R = MatrixResolvent(np.eye(3))
    contour = Contour.circle(0.0, 2.0, npts=4)
    with pytest.raises(TypeError):
        estimate_eigenvalue_count(R, contour, object())
    with pytest.raises(ValueError):
        estimate_eigenvalue_count(R, contour, NumpySpan.random(3, 1),
                                  nprobes=0)
    with pytest.raises(ValueError):
        feast(R, contour, NumpySpan.random(3, 2), count_probes=-1)


@pytest.mark.ngsolve
def test_ngsolve_span_dot_and_count():
    ng = pytest.importorskip("ngsolve")
    pytest.importorskip("netgen")
    from netgen.geom2d import unit_square

    from feast.backends.ngsolve import NGPencilResolvent, NGSpan

    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.2))
    fes = ng.H1(mesh, order=2, dirichlet=".*", complex=True)
    a, b = NGSpan.random(fes, 3, seed=1), NGSpan.random(fes, 2, seed=2)
    assert np.allclose(a.dot(b), a.to_numpy().conj().T @ b.to_numpy())

    u, v = fes.TnT()
    K = ng.BilinearForm(ng.grad(u) * ng.grad(v) * ng.dx).Assemble()
    M = ng.BilinearForm(u * v * ng.dx).Assemble()
    result = feast(NGPencilResolvent(fes, K.mat, M.mat),
                   Contour.circle(center=40.0, radius=35.0, npts=8),
                   NGSpan.random(fes, 5, seed=1), mode="hermitian",
                   seed=1, count_probes=16)
    assert len(result.eigenvalues) == 3
    assert not result.count_estimate.suggests_missing()
