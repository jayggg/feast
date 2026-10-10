import numpy as np
import pytest

from feast.zoom import segment_list, simple_multiple_zoom


def test_segment_list():
    assert segment_list([2, 3, 4, 9, 11, 12, 13, 17, 18]) == [
        [2, 3, 4], [9], [11, 12, 13], [17, 18]
    ]
    assert segment_list([]) == []


@pytest.mark.parametrize(
    "ew", [[1.0, 3.0, 3.01, 5.0, 7.0], [5.0, 1.0, 3.01, 7.0, 3.0]]
)
def test_zoom_contours_enclose_their_values(ew):
    # Sorted and unsorted input must give the same, correct contours.
    ew_in = list(ew)
    simple, multiple = simple_multiple_zoom(0.0, 8.0, ew, delta=0.1)
    assert ew == ew_in  # input not modified
    ew = np.asarray(ew)

    assert sorted(ew[simple["index"]]) == [1.0, 5.0, 7.0]
    for i, c, r in zip(simple["index"], simple["center"], simple["radius"]):
        inside = np.abs(ew - c) < r
        assert inside[i] and inside.sum() == 1

    (idx,) = multiple["index"]
    assert sorted(ew[idx]) == [3.0, 3.01]
    (c,), (r,) = multiple["center"], multiple["radius"]
    inside = np.abs(ew - c) < r
    assert inside[idx].all() and inside.sum() == 2


def test_zoom_rejects_values_outside_interval():
    with pytest.raises(ValueError):
        simple_multiple_zoom(0.0, 1.0, [0.5, 2.0])
