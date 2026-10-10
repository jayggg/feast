"""Validation of the Theorem 3.10 eigenspace-error estimator machinery.

Setting: A_true is Hermitian with an eigenvalue cluster {1.0, 1.05};
the "discrete" operator A_h is a small Hermitian perturbation of it,
living in the same space. FEAST runs on A_h; the ExactSourceError
estimator (Assumption 3.9 with C0 = 1, Y = V = C^n) then estimates the
gap to the true eigenspace.
"""

import numpy as np
import pytest
from scipy.linalg import subspace_angles

from feast import (
    Contour,
    SourceErrorEstimator,
    estimate_eigenspace_error,
    feast,
)
from feast.backends.numpy_backend import (
    ExactSourceError,
    MatrixResolvent,
    NumpySpan,
)

CLUSTER = [1.0, 1.05]
OTHERS = [3.0, 4.0, 5.0, 6.0]


@pytest.fixture(scope="module")
def setting():
    rng = np.random.default_rng(4)
    n = len(CLUSTER) + len(OTHERS)
    q, _ = np.linalg.qr(rng.random((n, n)) + 1j * rng.random((n, n)))
    A_true = q @ np.diag(CLUSTER + OTHERS) @ q.conj().T
    A_true = 0.5 * (A_true + A_true.conj().T)
    pert = rng.random((n, n)) + 1j * rng.random((n, n))
    A_h = A_true + 1e-4 * 0.5 * (pert + pert.conj().T)

    C = Contour.circle(center=1.025, radius=0.5, npts=8)
    R_h = MatrixResolvent(A_h)
    res = feast(
        R_h, C, NumpySpan.random(n, 4, seed=0),
        mode="hermitian", stop_tol=1e-13, niterations=60,
    )
    assert res.converged and res.right.dim == 2
    est = estimate_eigenspace_error(
        ExactSourceError(MatrixResolvent(A_true)), R_h, C, res.right
    )
    return A_true, A_h, C, res, est


def test_exact_source_error_is_protocol():
    R = MatrixResolvent(np.eye(3))
    assert isinstance(ExactSourceError(R), SourceErrorEstimator)


def test_eta_is_supremum_of_error_rayleigh_quotient(setting):
    """eta must equal sup ||(S_r - S_rh) e|| / ||e|| over e in E_h;
    check against brute-force random sampling (which can only be <=)."""
    A_true, A_h, C, res, est = setting
    from feast.projector import SpectralProjectorApprox

    Pt = SpectralProjectorApprox(MatrixResolvent(A_true), C)
    Ph = SpectralProjectorApprox(MatrixResolvent(A_h), C)
    E = res.right
    diff = Pt.apply(E).data - Ph.apply(E).data

    rng = np.random.default_rng(1)
    best = 0.0
    for _ in range(500):
        x = rng.standard_normal(E.dim) + 1j * rng.standard_normal(E.dim)
        best = max(best,
                   np.linalg.norm(diff @ x) / np.linalg.norm(E.data @ x))
    assert best <= est.eta * (1 + 1e-10)
    assert best >= est.eta * 0.99  # sampling should come close


def test_gap_bounded_by_estimator(setting):
    """Theorem 3.10: gap(E_h, E) <= C * eta, with modest C here."""
    A_true, A_h, C, res, est = setting
    lam, V = np.linalg.eigh(A_true)
    true_basis = V[:, [i for i, x in enumerate(lam)
                       if any(abs(x - c) < 1e-8 for c in CLUSTER)]]
    gap = np.sin(subspace_angles(res.right.data, true_basis)).max()
    assert est.eta > 0
    assert gap <= 20 * est.eta
    # both should reflect the 1e-4 perturbation scale
    assert est.eta < 1e-2


def test_worst_vector_attains_eta(setting):
    _, _, _, res, est = setting
    assert est.worst.dim == 1
    assert abs(np.linalg.norm(est.worst.data) - 1.0) < 1e-10
    assert abs(np.linalg.norm(est.eworst.data) - est.eta) < 1e-10 * (
        1 + est.eta
    )


def test_rayleigh_values_sorted_and_bounded(setting):
    _, _, _, res, est = setting
    lam = est.rayleigh_values
    assert len(lam) == 2
    assert lam[-1] >= lam[0] >= -1e-12
    assert np.isclose(np.sqrt(max(lam[-1], 0)), est.eta)


def test_empty_span_rejected():
    R = MatrixResolvent(np.eye(3))
    C = Contour.circle(0, 1, npts=4)
    s = NumpySpan.zeros(3, 1)
    s.remove([0])
    with pytest.raises(ValueError):
        estimate_eigenspace_error(ExactSourceError(R), R, C, s)
