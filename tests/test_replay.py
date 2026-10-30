import json
from pathlib import Path

from src.replay_events import load_replay, replay_events


FIXTURES_DIR = Path(__file__).parent / "fixtures"

REPLAY_PATH = FIXTURES_DIR / "sample_tracks.jsonl"
EXPECTED_EVENTS_PATH = FIXTURES_DIR / "expected_events.json"


def test_real_tracking_replay_matches_expected_events():
    """Replay real tracker observations and compare emitted events."""

    metadata, track_records = load_replay(
        REPLAY_PATH
    )

    actual_events = replay_events(
        metadata,
        track_records,
    )

    expected_events = json.loads(
        EXPECTED_EVENTS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert actual_events == expected_events


def test_replay_applies_stale_track_timeout():
    metadata, _ = load_replay(REPLAY_PATH)

    records = [
        {
            "track_id": 1,
            "zone_point": [100, 100],
            "inside_zone": True,
            "frame_number": 0,
        },
        {
            "track_id": 1,
            "zone_point": [100, 100],
            "inside_zone": False,
            "frame_number": 10,
        },
        {
            "track_id": 1,
            "zone_point": [100, 100],
            "inside_zone": False,
            "frame_number": 11,
        },
        {
            "track_id": 1,
            "zone_point": [100, 100],
            "inside_zone": False,
            "frame_number": 12,
        },
    ]

    without_cleanup = replay_events(metadata, records)

    with_cleanup = replay_events(
        {
            **metadata,
            "track_stale_timeout_frames": 3,
        },
        records,
    )

    assert [
        event["event_type"]
        for event in without_cleanup
    ] == ["ZONE_EXIT"]

    assert with_cleanup == []