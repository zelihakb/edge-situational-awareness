import cv2
import numpy as np


ZONE_NORMALIZED = [
    (0.30, 0.55),
    (0.70, 0.55),
    (0.85, 0.92),
    (0.15, 0.92),
]


def get_zone_polygon(
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    points = []

    for x_ratio, y_ratio in ZONE_NORMALIZED:
        x = int(x_ratio * frame_width)
        y = int(y_ratio * frame_height)

        points.append((x, y))

    return np.array(points, dtype=np.int32)
def is_point_inside_zone(
    point: tuple[int, int],
    polygon: np.ndarray,
) -> bool:
    result = cv2.pointPolygonTest(
        polygon,
        point,
        False,
    )

    return result >= 0
def draw_zone(
    frame: np.ndarray,
    polygon: np.ndarray,
) -> None:
    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=3,
    )

    first_point = polygon[0]

    cv2.putText(
        frame,
        "CONTROLLED ZONE",
        (
            int(first_point[0]),
            int(first_point[1]) - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )