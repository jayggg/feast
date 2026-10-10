"""Driver regression tests with known diagonal and triangular spectra."""

import numpy as np
import pytest

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan


def sort(lam):
    return np.sort_complex(np.asarray(lam))


def test_biorth_diag_parity():
    # Radius 1.0 puts the target eigenvalues on the boundary. A margin
    # keeps 1 and 1j inside and -1 and -1j outside despite rounding.
    A = np.diag([-1.0, 1.0, -1j, 1j])
    res = feast(
        MatrixResolvent(A),
        Contour.circle(center=1.0 + 1.0j, radius=1.3, npts=8),
        NumpySpan.random(4, 2, seed=2),
        mode="biorth",
        stop_tol=1e-12,
        niterations=100,
        seed=1,
    )
    assert res.converged
    assert np.allclose(sort(res.eigenvalues), sort([1.0, 1.0j]), atol=1e-10)


def test_biorth_triu_parity():
    N = 5
    A = np.triu(np.ones((N, N), dtype=complex))
    np.fill_diagonal(A, np.arange(1, N + 1))
    res = feast(
        MatrixResolvent(A),
        Contour.circle(center=3.0, radius=1.5, npts=8),
        NumpySpan.random(N, 3, seed=2),
        mode="biorth",
        stop_tol=1e-12,
        niterations=100,
        seed=1,
    )
    assert res.converged
    assert np.allclose(sort(res.eigenvalues).real, [2, 3, 4], atol=1e-8)


def test_hermitian_laplacian():
    # 1D Dirichlet Laplacian (n=40): eigenvalues 4 sin^2(k pi / (2(n+1)))
    n = 40
    A = (
        2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    )
    exact = 4 * np.sin(np.arange(1, n + 1) * np.pi / (2 * (n + 1))) ** 2
    target = exact[:3]
    C = Contour.circle(
        center=np.mean(target), radius=1.6 * np.ptp(target), npts=16
    )
    res = feast(
        MatrixResolvent(A), C, NumpySpan.random(n, 6, seed=0),
        mode="hermitian", stop_tol=1e-12, niterations=50,
    )
    assert res.converged
    found = np.sort(res.eigenvalues.real)
    assert np.allclose(found[:3], target, atol=1e-9)


def test_oversized_span_trimmed_to_true_count():
    A = np.diag([-1.0, 1.0, -1j, 1j, 3.0, 4.0])
    res = feast(
        MatrixResolvent(A),
        Contour.circle(center=3.5, radius=0.8, npts=8),
        NumpySpan.random(6, 4, seed=2),
        mode="biorth",
        stop_tol=1e-12,
        niterations=100,
        seed=1,
    )
    assert res.converged
    assert len(res.eigenvalues) == 2
    assert np.allclose(sort(res.eigenvalues), [3.0, 4.0], atol=1e-10)


def test_seeded_run_reproducible():
    A = np.diag([-1.0, 1.0, -1j, 1j, 3.0, 4.0])

    def run():
        return feast(
            MatrixResolvent(A),
            Contour.circle(center=1.0, radius=0.5, npts=8),
            NumpySpan.random(6, 4, seed=2),
            mode="biorth",
            expected_dim=2,
            niterations=40,
            seed=7,
        )
    r1, r2 = run(), run()
    assert np.array_equal(r1.eigenvalues, r2.eigenvalues)
    assert r1.iterations == r2.iterations


def test_history_populated():
    A = np.diag([1.0, 5.0])
    res = feast(
        MatrixResolvent(A),
        Contour.circle(center=1.0, radius=0.5, npts=8),
        NumpySpan.random(2, 1, seed=0),
        mode="biorth",
        seed=0,
    )
    h = res.history
    assert len(h.ritz_values) == res.iterations
    assert len(h.hausdorff) == res.iterations
    assert len(h.gram_singular_values) >= 1


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        feast(
            MatrixResolvent(np.eye(2)),
            Contour.circle(0, 1, npts=4),
            NumpySpan.random(2, 1),
            mode="leftish",
        )


@pytest.mark.parametrize("mode", ["hermitian", "right"])
def test_rank_reduction_of_oversized_filtered_span(mode):
    r = MatrixResolvent(np.diag([1.0, 10.0, 20.0, 30.0]))
    result = feast(
        r, Contour.circle(1.0, 0.5), NumpySpan.random(4, 3, seed=0),
        mode=mode,
    )
    assert result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [1.0], atol=1e-11)
    assert np.max(r.residual(result.eigenvalues, result.right)) < 1e-10


@pytest.mark.parametrize("mode", ["hermitian", "right"])
def test_rank_reduction_preserves_multiple_eigenvalue(mode):
    factor = np.eye(5) + 0.2 * np.eye(5, k=-1)
    b = factor @ factor.T
    a = factor @ np.diag([1.0, 1.0, 10.0, 20.0, 30.0]) @ factor.T
    r = MatrixResolvent(a, b)
    result = feast(
        r, Contour.circle(1.0, 0.5), NumpySpan.random(5, 4, seed=0),
        mode=mode, check_contour=1,
    )
    assert result.converged
    assert result.right.dim == len(result.eigenvalues) == 2
    assert np.allclose(result.eigenvalues, [1, 1], atol=1e-11)
    assert np.linalg.matrix_rank(result.right.data) == 2
    assert np.max(r.residual(result.eigenvalues, result.right)) < 1e-10


def test_right_mode_with_nonhermitian_mass():
    b = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    b[0, 1] = 0.5j
    transform = np.eye(4) + 0.3j * np.eye(4, k=1)
    a = b @ transform @ np.diag([1.0, 10.0, 20.0, 30.0])
    a = a @ np.linalg.inv(transform)
    r = MatrixResolvent(a, b)
    result = feast(
        r, Contour.circle(1.0, 0.5), NumpySpan.random(4, 3, seed=0),
        mode="right",
    )
    assert result.converged
    assert len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [1.0], atol=1e-11)
    assert np.max(r.residual(result.eigenvalues, result.right)) < 1e-10


@pytest.mark.parametrize("mode", ["hermitian", "right"])
def test_dependent_initial_columns(mode):
    column = NumpySpan.random(4, 1, seed=0).data
    y = NumpySpan(np.hstack([0 * column, column, 2j * column]))
    r = MatrixResolvent(np.diag([1.0, 10.0, 20.0, 30.0]))
    result = feast(r, Contour.circle(1.0, 0.5), y, mode=mode)
    assert result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.max(r.residual(result.eigenvalues, result.right)) < 1e-10


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_unrecoverable_expected_dimension_is_not_convergence(mode):
    r = MatrixResolvent(np.diag([1.0, 5.0]))
    result = feast(
        r, Contour.circle(1.0, 0.5, npts=32),
        NumpySpan.random(2, 2, seed=0), mode=mode, expected_dim=2,
        nrefresh=2, niterations=10, report_residuals=True, seed=0,
    )
    assert not result.converged
    assert result.iterations == 3
    assert result.right.dim == len(result.eigenvalues) == 1
    if result.left is not None:
        assert result.left.dim == 1
    residual = r.residual(result.eigenvalues, result.right, result.left)
    assert np.max(residual) < 1e-10


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_refresh_recovers_missing_eigenvector(mode):
    r = MatrixResolvent(np.diag([1.0, 2.0, 10.0, 20.0]))
    y = NumpySpan(np.eye(4)[:, :1])
    result = feast(
        r, Contour.circle(1.5, 0.75, npts=16), y,
        mode=mode, yl=y.copy(), expected_dim=2, seed=1,
    )
    assert result.converged and result.iterations >= 3
    assert result.right.dim == len(result.eigenvalues) == 2
    assert np.allclose(np.sort(result.eigenvalues.real), [1, 2], atol=1e-10)
    assert np.linalg.matrix_rank(result.right.data) == 2
    assert np.max(r.residual(
        result.eigenvalues, result.right, result.left,
    )) < 1e-9


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_iteration_limit_returns_only_computed_pairs(mode):
    r = MatrixResolvent(np.diag([1.0, 5.0]))
    y = NumpySpan(np.eye(2)[:, :1])
    result = feast(
        r, Contour.circle(1.0, 0.5), y, mode=mode, yl=y.copy(),
        expected_dim=2, niterations=1, seed=0,
    )
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    if result.left is not None:
        assert result.left.dim == 1
    assert np.max(r.residual(
        result.eigenvalues, result.right, result.left,
    )) < 1e-12


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_failed_solve_after_augmentation_preserves_last_pairs(restart, mode):
    class FailingResolvent(MatrixResolvent):
        def __init__(self):
            super().__init__(np.diag([1.0, 5.0, 10.0]))
            self.calls = 0

        def rayleigh(self, yl, yr):
            self.calls += 1
            a, b = super().rayleigh(yl, yr)
            return (a, b) if self.calls == 1 else (a, 0 * b)

    r = FailingResolvent()
    initial = NumpySpan(np.eye(3)[:, :1])
    result = feast(
        r, Contour.circle(1.0, 0.5), initial, yl=initial.copy(),
        mode=mode, niterations=1 if restart else 2,
        nrestarts=2 if restart else 1,
        expected_dim=None if restart else 2, seed=0,
    )
    assert result.iterations == r.calls == 2
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [1])
    assert np.max(r.residual(
        result.eigenvalues, result.right, result.left,
    )) < 1e-12
    if mode == "biorth":
        assert result.left.dim == 1


def test_first_failed_solve_returns_empty_pairs():
    result = feast(
        MatrixResolvent(np.eye(2), -np.eye(2)),
        Contour.circle(-1.0, 0.5), NumpySpan.random(2, 1, seed=0),
    )
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 0


@pytest.mark.parametrize("nrefresh", [0, 1])
def test_zero_filtered_span_can_refresh(nrefresh):
    result = feast(
        MatrixResolvent(np.diag([1.0, 10.0])), Contour.circle(1.0, 0.5),
        NumpySpan.zeros(2, 2), nrefresh=nrefresh, seed=0,
    )
    assert result.converged == bool(nrefresh)
    assert result.right.dim == len(result.eigenvalues) == nrefresh


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
@pytest.mark.parametrize("check_contour", [0, 1, 3, 100])
def test_convergence_uses_only_in_contour_pairs(mode, check_contour):
    r = MatrixResolvent(np.diag([0.25, 2.0]))
    contour = Contour.circle(0, 1, npts=4)
    y = NumpySpan(np.eye(2))
    result = feast(
        r, contour, y, mode=mode, yl=y.copy(),
        check_contour=check_contour, niterations=4,
    )
    assert result.converged
    assert result.iterations == 2
    assert np.allclose(result.eigenvalues, [0.25])
    assert result.right.dim == 1
    if result.left is not None:
        assert result.left.dim == 1
    assert np.max(r.residual(
        result.eigenvalues, result.right, result.left,
    )) < 1e-12
    assert len(result.history.ritz_values[0]) == 2
    assert np.isinf(result.history.hausdorff[0])
    assert result.history.hausdorff[-1] < 1e-12


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
@pytest.mark.parametrize("check_contour", [0, 1, 3])
def test_all_exterior_pairs_return_empty_nonconverged(mode, check_contour):
    y = NumpySpan(np.eye(2))
    result = feast(
        MatrixResolvent(np.diag([2.0, 3.0])),
        Contour.circle(0, 1, npts=4), y, yl=y.copy(), mode=mode,
        check_contour=check_contour, niterations=4,
    )
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 0
    if result.left is not None:
        assert result.left.dim == 0
    assert np.all(np.isinf(result.history.hausdorff))


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_exterior_pairs_do_not_satisfy_expected_dimension(mode):
    y = NumpySpan(np.eye(2))
    result = feast(
        MatrixResolvent(np.diag([0.25, 2.0])),
        Contour.circle(0, 1, npts=4), y, yl=y.copy(), mode=mode,
        expected_dim=2, check_contour=1, niterations=4,
    )
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [0.25])
    assert all(len(values) == 2 for values in result.history.ritz_values)


@pytest.mark.parametrize("mode", ["hermitian", "right", "biorth"])
def test_iteration_limit_filters_returned_pairs(mode):
    y = NumpySpan(np.eye(2))
    result = feast(
        MatrixResolvent(np.diag([0.25, 2.0])),
        Contour.circle(0, 1, npts=4), y, yl=y.copy(), mode=mode,
        check_contour=0, niterations=1,
    )
    assert not result.converged
    assert result.right.dim == len(result.eigenvalues) == 1
    assert np.allclose(result.eigenvalues, [0.25])
    if result.left is not None:
        assert result.left.dim == 1


@pytest.mark.parametrize("ritz_sets", [
    [[0.25, 0.9999], [0.25, 1.0001], [0.25, 1.0001]],
    [[0.25, 2.0], [2.0, 3.0], [0.25, 2.0], [0.25, 2.0]],
    [[0.25, 2.0], [0.25, 3.0]],
])
def test_convergence_tracks_in_contour_sets(monkeypatch, ritz_sets):
    values = iter(ritz_sets)

    def small_eig(a, b, mode):
        return np.array(next(values)), None, np.eye(2)

    monkeypatch.setattr("feast.driver._small_eig", small_eig)
    result = feast(
        MatrixResolvent(np.diag([0.25, 2.0])),
        Contour.circle(0, 1, npts=4), NumpySpan(np.eye(2)),
        check_contour=0, stop_tol=1e-3, niterations=len(ritz_sets),
    )
    assert result.converged
    assert result.iterations == len(ritz_sets)
    assert np.allclose(result.eigenvalues, [0.25])
