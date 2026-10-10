import numpy as np

from feast.backends.numpy_backend import NumpySpan
from feast.span import Span


def test_numpyspan_satisfies_protocol():
    assert isinstance(NumpySpan.zeros(5, 2), Span)


def test_random_reproducible():
    a = NumpySpan.random(10, 3, seed=42)
    b = NumpySpan.random(10, 3, seed=42)
    assert np.array_equal(a.data, b.data)


def test_rotate_scale_remove():
    s = NumpySpan(np.eye(3))
    s.rotate(np.array([[1, 1], [0, 1], [0, 0]], dtype=complex))
    assert s.dim == 2
    s.scale(1, 2.0)
    assert np.allclose(s.data[:, 1], [2, 2, 0])
    s.remove([0])
    assert s.dim == 1 and np.allclose(s.data[:, 0], [2, 2, 0])


def test_append_random_deterministic():
    a = NumpySpan.zeros(8, 1)
    b = NumpySpan.zeros(8, 1)
    a.append_random(2, np.random.default_rng(7))
    b.append_random(2, np.random.default_rng(7))
    assert a.dim == 3
    assert np.array_equal(a.data, b.data)


def test_copy_is_deep():
    a = NumpySpan.random(4, 2, seed=0)
    b = a.copy()
    b.scale(0, 5.0)
    assert not np.array_equal(a.data, b.data)
