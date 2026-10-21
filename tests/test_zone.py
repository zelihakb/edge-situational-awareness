import numpy as np
import pytest

from src.zone import (
    get_zone_center,
    get_zone_polygon,
    is_point_inside_zone,
    get_signed_zone_distance,
    load_zone_config,
    save_zone_config,
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

def test_signed_zone_distance():
    polygon = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.int32,
    )

    assert get_signed_zone_distance(
        (5, 5), polygon
    ) == 5.0

    assert get_signed_zone_distance(
        (15, 5), polygon
    ) == -5.0

    assert get_signed_zone_distance(
        (0, 5), polygon
    ) == 0.0

def test_zone_config_roundtrip(tmp_path):
    points = [
        (0.1, 0.2),
        (0.8, 0.2),
        (0.8, 0.8),
        (0.1, 0.8),
    ]

    config_path = tmp_path / "zone.json"

    save_zone_config(config_path, points)
    loaded_points = load_zone_config(config_path)

    assert loaded_points == points

    polygon = get_zone_polygon(
        200,
        100,
        loaded_points,
    )

    expected = np.array(
        [
            [20, 20],
            [160, 20],
            [160, 80],
            [20, 80],
        ],
        dtype=np.int32,
    )

    assert np.array_equal(polygon, expected)


@pytest.mark.parametrize(
    "points",
    [
        [(0.1, 0.1), (0.2, 0.2)],
        [(0.1, 0.1), (1.2, 0.2), (0.8, 0.8)],
        [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
    ],
)
def test_invalid_zone_config_is_rejected(tmp_path, points):
    config_path = tmp_path / "invalid_zone.json"

    with pytest.raises(ValueError):
        save_zone_config(config_path, points)

    assert not config_path.exists()

def test_self_intersecting_polygon_is_rejected(tmp_path):
    # We use a crossing polygon with non-zero area.
    points = [
        (0.1, 0.1),
        (0.9, 0.7),
        (0.1, 0.9),
        (0.8, 0.1),
    ]

    config_path = tmp_path / "crossed_zone.json"

    with pytest.raises(ValueError, match="intersect"):
        save_zone_config(config_path, points)

    assert not config_path.exists()


def test_concave_polygon_is_accepted(tmp_path):
    # We allow concave polygons as long as edges do not cross.
    points = [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.9, 0.9),
        (0.5, 0.5),
        (0.1, 0.9),
    ]

    config_path = tmp_path / "concave_zone.json"

    save_zone_config(config_path, points)

    assert load_zone_config(config_path) == points
