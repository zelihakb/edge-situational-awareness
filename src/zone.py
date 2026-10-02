import cv2
import numpy as np


# Normalized coordinates keep the demo zone resolution-independent.
ZONE_NORMALIZED = [
    (0.30, 0.55),
    (0.70, 0.55),
    (0.85, 0.92),
    (0.15, 0.92),
]


def get_zone_polygon(frame_width: int, frame_height: int) -> np.ndarray:
    """Convert normalized zone coordinates to pixel coordinates."""
    points = [
        (int(x_ratio * frame_width), int(y_ratio * frame_height))
        for x_ratio, y_ratio in ZONE_NORMALIZED
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