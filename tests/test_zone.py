import numpy as np

from src.zone import (
    get_zone_center,
    get_zone_polygon,
    is_point_inside_zone,
)


def test_zone_polygon_scales_with_resolution():
    polygon_small = get_zone_polygon(1000, 1000)
    polygon_large = get_zone_polygon(2000, 2000)

    expected_small = np.array(
        [
            [300, 550],
            [700, 550],
            [850, 920],
            [150, 920],
        ],
        dtype=np.int32,
    )

    assert np.array_equal(polygon_small, expected_small)
    assert np.array_equal(polygon_large, expected_small * 2)


def test_zone_center_is_polygon_mean():
    polygon = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.int32,
    )

    center = get_zone_center(polygon)

    assert np.allclose(center, [5.0, 5.0])


def test_point_inside_zone_returns_true():
    polygon = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.int32,
    )

    assert is_point_inside_zone((5, 5), polygon) is True


def test_point_outside_zone_returns_false():
    polygon = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.int32,
    )

    assert is_point_inside_zone((15, 5), polygon) is False


def test_point_on_zone_boundary_counts_as_inside():
    polygon = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.int32,
    )

    assert is_point_inside_zone((5, 0), polygon) is True