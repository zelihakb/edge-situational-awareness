import numpy as np
import pytest

from src.events import TrackEventEngine


def make_engine(
    fps: float = 24.0,
    confirm_frames: int = 3,
    loitering_seconds: float = 3.0,
) -> TrackEventEngine:
    """Create an event engine with simple deterministic test settings."""
    return TrackEventEngine(
        source_fps=fps,
        zone_center=np.array([0.0, 0.0], dtype=np.float32),
        motion_min_delta=10.0,
        zone_confirm_frames=confirm_frames,
        loitering_threshold_seconds=loitering_seconds,
        motion_history_length=8,
    )


def update(
    engine: TrackEventEngine,
    inside: bool,
    frame_number: int,
    track_id: int = 1,
) -> list[dict]:
    """Update one synthetic track and return only emitted events."""
    _, events = engine.update(
        track_id=track_id,
        zone_point=(0, 0),
        inside_zone=inside,
        frame_number=frame_number,
    )
    return events


def test_first_observation_inside_does_not_emit_entry():
    engine = make_engine()

    events = update(
        engine,
        inside=True,
        frame_number=0,
    )

    assert events == []
    assert engine.zone_state[1] is True
    assert engine.zone_entry_count == 0


def test_one_or_two_inside_frames_do_not_confirm_entry():
    engine = make_engine()

    update(engine, inside=False, frame_number=0)

    events_1 = update(engine, inside=True, frame_number=1)
    events_2 = update(engine, inside=True, frame_number=2)

    assert events_1 == []
    assert events_2 == []
    assert engine.zone_state[1] is False
    assert engine.zone_entry_count == 0


def test_three_inside_frames_confirm_single_entry():
    engine = make_engine()

    update(engine, inside=False, frame_number=0)

    update(engine, inside=True, frame_number=1)
    update(engine, inside=True, frame_number=2)
    events = update(engine, inside=True, frame_number=3)

    assert events == [
        {
            "event_type": "ZONE_ENTRY",
            "frame_number": 4,
            "track_id": 1,
        }
    ]

    assert engine.zone_state[1] is True
    assert engine.zone_entry_count == 1


def test_boundary_jitter_resets_entry_candidate():
    engine = make_engine()

    update(engine, inside=False, frame_number=0)

    update(engine, inside=True, frame_number=1)
    update(engine, inside=True, frame_number=2)

    # Raw state returns to the confirmed OUTSIDE state,
    # so the pending entry candidate must be cancelled.
    events = update(engine, inside=False, frame_number=3)

    assert events == []
    assert engine.zone_state[1] is False
    assert engine.zone_candidate_count[1] == 0

    update(engine, inside=True, frame_number=4)
    update(engine, inside=True, frame_number=5)
    events = update(engine, inside=True, frame_number=6)

    assert len(events) == 1
    assert events[0]["event_type"] == "ZONE_ENTRY"
    assert events[0]["frame_number"] == 7


def test_three_outside_frames_confirm_single_exit():
    engine = make_engine()

    # Initial observation inside does not create an ENTRY.
    update(engine, inside=True, frame_number=0)

    update(engine, inside=False, frame_number=1)
    update(engine, inside=False, frame_number=2)
    events = update(engine, inside=False, frame_number=3)

    assert events == [
        {
            "event_type": "ZONE_EXIT",
            "frame_number": 4,
            "track_id": 1,
        }
    ]

    assert engine.zone_state[1] is False
    assert engine.zone_exit_count == 1


@pytest.mark.parametrize(
    ("fps", "expected_event_frame"),
    [
        (24.0, 73),
        (30.0, 91),
    ],
)
def test_loitering_uses_video_time(
    fps: float,
    expected_event_frame: int,
):
    engine = make_engine(
        fps=fps,
        loitering_seconds=3.0,
    )

    # Frame 0 establishes the track inside the zone.
    update(engine, inside=True, frame_number=0)

    emitted_events = []

    for frame_number in range(1, int(fps * 3) + 1):
        emitted_events.extend(
            update(
                engine,
                inside=True,
                frame_number=frame_number,
            )
        )

    loitering_events = [
        event
        for event in emitted_events
        if event["event_type"] == "LOITERING"
    ]

    assert len(loitering_events) == 1
    assert loitering_events[0]["frame_number"] == expected_event_frame
    assert loitering_events[0]["duration_seconds"] == pytest.approx(3.0)


def test_continuous_stay_emits_only_one_loitering_event():
    engine = make_engine(
        fps=24.0,
        loitering_seconds=3.0,
    )

    update(engine, inside=True, frame_number=0)

    emitted_events = []

    # Stay inside much longer than the loitering threshold.
    for frame_number in range(1, 150):
        emitted_events.extend(
            update(
                engine,
                inside=True,
                frame_number=frame_number,
            )
        )

    loitering_events = [
        event
        for event in emitted_events
        if event["event_type"] == "LOITERING"
    ]

    assert len(loitering_events) == 1


def test_exit_and_reentry_start_new_loitering_period():
    engine = make_engine(
        fps=24.0,
        loitering_seconds=1.0,
    )

    emitted_events = []

    # First continuous stay.
    emitted_events.extend(
        update(engine, inside=True, frame_number=0)
    )

    for frame_number in range(1, 25):
        emitted_events.extend(
            update(engine, inside=True, frame_number=frame_number)
        )

    # Confirm EXIT with three consecutive outside frames.
    for frame_number in range(25, 28):
        emitted_events.extend(
            update(engine, inside=False, frame_number=frame_number)
        )

    # Confirm re-entry with three consecutive inside frames.
    for frame_number in range(28, 31):
        emitted_events.extend(
            update(engine, inside=True, frame_number=frame_number)
        )

    # Remain inside long enough to loiter again.
    for frame_number in range(31, 55):
        emitted_events.extend(
            update(engine, inside=True, frame_number=frame_number)
        )

    event_types = [
        event["event_type"]
        for event in emitted_events
    ]

    assert event_types.count("LOITERING") == 2
    assert event_types.count("ZONE_EXIT") == 1
    assert event_types.count("ZONE_ENTRY") == 1
def test_stale_track_state_is_removed_without_exit():
    engine = make_engine(loitering_seconds=1.0)

    for frame in range(25):
        update(engine, inside=True, frame_number=frame)

    assert 1 in engine.loitering_reported

    expired = engine.expire_stale_tracks(
        frame_number=29,
        max_missing_frames=3,
    )

    assert expired == [1]
    assert 1 not in engine.last_seen_frame
    assert 1 not in engine.zone_state
    assert 1 not in engine.zone_candidate_state
    assert 1 not in engine.zone_candidate_count
    assert 1 not in engine.zone_enter_frame
    assert 1 not in engine.loitering_reported
    assert 1 not in engine.motion_position_history
    assert 1 not in engine.motion_distance_history
    assert engine.zone_exit_count == 0


def test_short_tracking_gap_preserves_state():
    engine = make_engine()

    update(engine, inside=False, frame_number=0)

    expired = engine.expire_stale_tracks(
        frame_number=3,
        max_missing_frames=3,
    )

    assert expired == []
    assert 1 in engine.zone_state

    update(engine, inside=True, frame_number=3)
    update(engine, inside=True, frame_number=4)
    events = update(
        engine, inside=True, frame_number=5
    )

    assert len(events) == 1
    assert events[0]["event_type"] == "ZONE_ENTRY"


def test_expired_track_starts_with_fresh_state():
    engine = make_engine()

    update(engine, inside=True, frame_number=0)

    engine.expire_stale_tracks(
        frame_number=4,
        max_missing_frames=3,
    )

    events = update(
        engine, inside=True, frame_number=4
    )

    assert events == []
    assert engine.zone_enter_frame[1] == 5
    assert engine.zone_entry_count == 0


def test_invalid_stale_timeout_is_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.expire_stale_tracks(
            frame_number=10,
            max_missing_frames=0,
        )
