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