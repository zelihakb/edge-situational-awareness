from collections import defaultdict, deque

import cv2
import numpy as np


class TrackVisualizer:
    """Draw per-track debug points, motion labels, and trajectories."""

    def __init__(
        self,
        trail_length: int,
    ) -> None:

        self.track_history: dict[
            int,
            deque[tuple[int, int]],
        ] = defaultdict(
            lambda: deque(
                maxlen=trail_length
            )
        )

    def draw_track(
        self,
        frame: np.ndarray,
        track_id: int,
        center: tuple[int, int],
        zone_point: tuple[int, int],
        inside_zone: bool,
        motion_direction: str,
    ) -> None:
        """Draw zone status, motion direction, and trajectory."""

        point_color = (
            (0, 255, 0)
            if inside_zone
            else (0, 0, 255)
        )

        cv2.circle(
            frame,
            zone_point,
            radius=6,
            color=point_color,
            thickness=-1,
        )

        cv2.putText(
            frame,
            motion_direction,
            (
                zone_point[0] + 8,
                zone_point[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        history = self.track_history[
            track_id
        ]

        history.append(
            center
        )

        if len(history) < 2:
            return

        points = np.asarray(
            history,
            dtype=np.int32,
        ).reshape(
            (-1, 1, 2)
        )

        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=(255, 255, 255),
            thickness=2,
        )


def draw_pipeline_overlay(
    frame: np.ndarray,
    rolling_fps: float,
    tracking_latency_ms: float,
    inference_latency_ms: float,
    zone_entry_count: int,
    zone_exit_count: int,
) -> None:
    """Draw high-level pipeline metrics and event counters."""

    lines = [
        (
            f"Rolling FPS: {rolling_fps:.1f}",
            (255, 255, 255),
        ),
        (
            f"Track call: {tracking_latency_ms:.1f} ms",
            (255, 255, 255),
        ),
        (
            f"Inference: {inference_latency_ms:.1f} ms",
            (255, 255, 255),
        ),
        (
            f"ZONE ENTRY: {zone_entry_count}",
            (0, 255, 0),
        ),
        (
            f"ZONE EXIT: {zone_exit_count}",
            (0, 0, 255),
        ),
    ]

    for index, (
        text,
        color,
    ) in enumerate(lines):

        y = (
            35
            + index * 35
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if index < 3 else 0.7,
            color,
            2,
            cv2.LINE_AA,
        )