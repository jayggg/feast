import numpy as np
import pytest

from feast import Contour


def test_circle_nodes_on_circle():
    C = Contour.circle(center=2 + 1j, radius=0.5, npts=8)
    assert C.npts == 8
    assert np.allclose(abs(C.z - (2 + 1j)), 0.5)


def test_odd_npts_rounded_up():
    C = Contour.circle(center=0, radius=1, npts=7)
    assert C.npts == 8


def test_shifted_nodes_avoid_real_axis():
    C = Contour.circle(center=1.0, radius=1.0, npts=8)
    assert np.all(np.abs(C.z.imag) > 1e-12)


def test_weights_reproduce_cauchy_integral():
    # (2 pi i)^-1  ∮ dz/(z - a) = 1 for a inside, 0 for a far outside
    C = Contour.circle(center=0.0, radius=1.0, npts=16)
    for a, expected in [(0.1 + 0.2j, 1.0), (5.0, 0.0)]:
        val = np.sum(C.w / (C.z - a))
        assert abs(val - expected) < 1e-8


def test_within_circle():
    C = Contour.circle(center=3.0, radius=1.5, npts=8)
    lam = np.array([3.0, 4.4, 4.6, 3 + 1.4j, 3 + 1.6j])
    assert list(C.within(lam)) == [True, True, False, True, False]


def test_filter_inside_vs_outside():
    C = Contour.circle(center=0.0, radius=1.0, npts=16)
    f = C.filter(np.array([0.0, 0.5, 3.0]))
    assert abs(f[0] - 1) < 1e-8
    assert abs(f[1] - 1) < 1e-3
    assert abs(f[2]) < 0.1


def test_ellipse_within_and_cauchy():
    C = Contour.ellipse(center=0.0, radius=1.0, rhoinv=0.5, npts=32)
    assert C.within(np.array([0.9]))[0]
    assert not C.within(np.array([0.9 + 0.5j]))[0]
    val = np.sum(C.w / (C.z - 0.1))
    assert abs(val - 1) < 1e-8


def test_custom_contour_shape_check():
    with pytest.raises(ValueError):
        Contour.custom(np.ones(3), np.ones(4), lambda z: z.real > 0)


def test_circle_boundary_tolerance():
    # a point exactly ON the contour counts as inside (deterministically),
    # regardless of rounding in |z - c|
    C = Contour.circle(center=1.0 + 1.0j, radius=1.0, npts=8)
    on_boundary = np.array([1.0 + 0j, 1j, 1 + 2j, 2 + 1j])
    assert C.within(on_boundary).all()
    # clearly-outside points are still outside
    assert not C.within(np.array([-1.0 + 0j, 3.0 + 1j])).any()
    # margin is relative and tiny: 2 parts in 1e9 outside stays outside
    assert not C.within(np.array([2 + 1j + 2e-9]))[0]


def test_ellipse_boundary_tolerance():
    C = Contour.ellipse(center=0.0, radius=1.0, rhoinv=0.5, npts=16)
    e = (2.0 - 0.5) / (2.0 + 0.5)  # minor/major axis ratio
    on_boundary = np.array([1.0 + 0j, -1.0 + 0j, 1j * e, -1j * e])
    assert C.within(on_boundary).all()
    assert not C.within(np.array([1.0 + 1e-8, 1j * e * (1 + 1e-8)])).any()
