import numpy as np

from src.events import TrackEventEngine


def test_stationary_track_remains_stable():
    """Verify that a stationary track remains stable."""

    engine = TrackEventEngine(
        source_fps=24.0,
        zone_center=np.array([100.0, 100.0]),
        motion_min_delta=2.0,
        zone_confirm_frames=3,
        loitering_threshold_seconds=3.0,
        motion_history_length=8,
    )

    for frame_number in range(10):
        motion_direction, events = engine.update(
            track_id=1,
            zone_point=(50, 50),
            inside_zone=False,
            frame_number=frame_number,
        )

        assert motion_direction == "STABLE"
        assert events == []

def test_small_position_jitter_remains_stable():
    """Ignore small position changes caused by tracking jitter."""

    engine = TrackEventEngine(
        source_fps=24.0,
        zone_center=np.array([100.0, 100.0]),
        motion_min_delta=2.0,
        zone_confirm_frames=3,
        loitering_threshold_seconds=3.0,
        motion_history_length=8,
    )

    for frame_number in range(10):
        x_position = 50 + (frame_number % 2)

        motion_direction, events = engine.update(
            track_id=1,
            zone_point=(x_position, 100),
            inside_zone=False,
            frame_number=frame_number,
        )

        assert motion_direction == "STABLE"
        assert events == []