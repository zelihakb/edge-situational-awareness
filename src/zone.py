import json
from pathlib import Path

import cv2
import numpy as np


# Normalized coordinates keep the demo zone resolution-independent.
ZONE_NORMALIZED = [
    (0.30, 0.55),
    (0.70, 0.55),
    (0.85, 0.92),
    (0.15, 0.92),
]


def get_zone_polygon(
    frame_width: int,
    frame_height: int,
    normalized_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Convert normalized zone coordinates to pixel coordinates."""

    coordinates = (
        ZONE_NORMALIZED
        if normalized_points is None
        else validate_normalized_points(normalized_points)
    )

    points = [
        (int(x * frame_width), int(y * frame_height))
        for x, y in coordinates
    ]

    return np.array(points, dtype=np.int32)

def get_zone_center(polygon: np.ndarray) -> np.ndarray:
    """Return the image-space center of a zone polygon."""
    return np.mean(
        polygon,
        axis=0,
    ).astype(np.float32)


def is_point_inside_zone(
    point: tuple[int, int],
    polygon: np.ndarray,
) -> bool:
    """Return True when a point is inside or on the polygon boundary."""
    return cv2.pointPolygonTest(
        polygon,
        point,
        False,
    ) >= 0

def get_signed_zone_distance(
    point: tuple[int, int],
    polygon: np.ndarray,
) -> float:
    """Return signed pixel distance to the nearest zone boundary."""

    return float(
        cv2.pointPolygonTest(
            polygon,
            point,
            True,
        )
    )

def draw_zone(
    frame: np.ndarray,
    polygon: np.ndarray,
) -> None:
    """Draw the controlled zone and its label."""

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=3,
    )

    x, y = polygon[0]

    cv2.putText(
        frame,
        "CONTROLLED ZONE",
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
def validate_normalized_points(
    points: object,
) -> list[tuple[float, float]]:
    """Validate a normalized polygon before using it."""

    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("A zone requires at least three points.")

    normalized = []

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each point must contain x and y.")

        x, y = point

        if type(x) not in (int, float) or type(y) not in (int, float):
            raise ValueError("Coordinates must be numeric.")

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Coordinates must be finite.")

        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")

        normalized.append((float(x), float(y)))

    polygon = np.array(normalized, dtype=np.float32)

    if (
        len(set(normalized)) != len(normalized)
        or cv2.contourArea(polygon) <= 1e-6
    ):
        raise ValueError("Polygon must have distinct, non-collinear points.")

    return normalized


def load_zone_config(
    path: Path,
) -> list[tuple[float, float]]:
    """Load and validate normalized polygon coordinates."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or data.get("name") != "CONTROLLED_ZONE":
        raise ValueError("Invalid zone configuration name.")

    return validate_normalized_points(
        data.get("normalized_points")
    )


def save_zone_config(
    path: Path,
    normalized_points: list[tuple[float, float]],
) -> None:
    """Validate and save a normalized zone configuration."""

    points = validate_normalized_points(normalized_points)

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": "CONTROLLED_ZONE",
        "normalized_points": points,
    }

    path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
