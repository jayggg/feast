import numpy as np

from feast.backends.numpy_backend import NumpySpan
from feast.subspace import analyze_gram, apply_cleaning


def random_unitary(m, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.random((m, m)) + 1j * rng.random((m, m)))
    return q


def test_full_rank_gram_kept():
    g = np.diag([1.0, 0.5, 0.1])
    a = analyze_gram(g, eta_tol=1e-8)
    assert a.ok and a.rank == 3 and a.cut == []


def test_relative_threshold_cuts():
    # 1e-16 would survive an absolute tol but is 1e-12 relative to s[0]
    g = np.diag([1e-4, 1e-16, 1e-20])
    a = analyze_gram(g, eta_tol=1e-8)
    assert a.rank == 1 and a.cut == [1, 2]


def test_ill_conditioned_rotation_is_unitary():
    # Gram with condition number 1e16: rotations must stay unitary
    d = np.diag([1.0, 1e-16])
    u0 = random_unitary(2, 1)
    w0 = random_unitary(2, 2)
    a = analyze_gram(u0 @ d @ w0.conj().T, eta_tol=1e-8)
    assert np.allclose(a.u @ a.u.conj().T, np.eye(2), atol=1e-12)
    assert np.allclose(a.w @ a.w.conj().T, np.eye(2), atol=1e-12)
    assert a.rank == 1


def test_apply_cleaning_biorthonormalizes():
    # Build spans with known B=I Gram matrix, dirty second direction
    rng = np.random.default_rng(3)
    n, m = 20, 3
    yl = NumpySpan(rng.random((n, m)) + 1j * rng.random((n, m)))
    yr = NumpySpan(rng.random((n, m)) + 1j * rng.random((n, m)))
    yr.data[:, 2] = 1e-14 * yr.data[:, 2]  # near-kernel direction
    gram = yl.data.conj().T @ yr.data
    a = analyze_gram(gram, eta_tol=1e-8)
    apply_cleaning(a, yl, yr)
    assert yl.dim == yr.dim == a.rank
    new_gram = yl.data.conj().T @ yr.data
    assert np.allclose(new_gram, np.eye(a.rank), atol=1e-10)


def test_zero_and_nonfinite_gram_rejected():
    assert not analyze_gram(np.zeros((3, 3))).ok
    g = np.eye(3, dtype=complex)
    g[0, 0] = np.nan
    assert not analyze_gram(g).ok
