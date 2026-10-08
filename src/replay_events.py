import json
from pathlib import Path

import numpy as np

from src.config import REPLAY_CAPTURE_PATH
from src.events import TrackEventEngine


def load_replay(
    replay_path: Path,
) -> tuple[dict, list[dict]]:
    """Load replay metadata and recorded track observations from JSONL."""

    metadata: dict | None = None
    track_records: list[dict] = []

    with replay_path.open(
        "r",
        encoding="utf-8",
    ) as replay_file:
        for line in replay_file:
            record = json.loads(line)

            if record["record_type"] == "metadata":
                metadata = record
                continue

            if record["record_type"] == "track":
                track_records.append(record)

    if metadata is None:
        raise ValueError(
            "Replay file does not contain metadata."
        )

    return metadata, track_records


def build_event_engine(
    metadata: dict,
) -> TrackEventEngine:
    """Create an event engine using the settings stored in the replay."""

    return TrackEventEngine(
        source_fps=float(
            metadata["source_fps"]
        ),
        zone_center=np.asarray(
            metadata["zone_center"],
            dtype=np.float32,
        ),
        motion_min_delta=float(
            metadata["motion_min_delta"]
        ),
        zone_confirm_frames=int(
            metadata["zone_confirm_frames"]
        ),
        loitering_threshold_seconds=float(
            metadata[
                "loitering_threshold_seconds"
            ]
        ),
        motion_history_length=int(
            metadata["motion_history_length"]
        ),
    )


def replay_events(
    metadata: dict,
    track_records: list[dict],
) -> list[dict]:
    """Replay recorded tracks through the event engine."""

    event_engine = build_event_engine(
        metadata
    )

    emitted_events: list[dict] = []

    for record in track_records:
        _, events = event_engine.update(
            track_id=int(
                record["track_id"]
            ),
            zone_point=(
                int(record["zone_point"][0]),
                int(record["zone_point"][1]),
            ),
            inside_zone=bool(
                record["inside_zone"]
            ),
            frame_number=int(
                record["frame_number"]
            ),
        )

        emitted_events.extend(
            events
        )

    return emitted_events


def main() -> None:
    metadata, track_records = load_replay(
        REPLAY_CAPTURE_PATH
    )

    events = replay_events(
        metadata,
        track_records,
    )

    print(
        f"Replay records : {len(track_records)}"
    )

    print(
        f"Replay events  : {len(events)}"
    )

    print(
        "=" * 60
    )

    for event in events:
        if event["event_type"] == "LOITERING":
            print(
                f"{event['event_type']} | "
                f"Frame: {event['frame_number']} | "
                f"Track ID: {event['track_id']} | "
                f"Duration: "
                f"{event['duration_seconds']:.1f}s"
            )

        else:
            print(
                f"{event['event_type']} | "
                f"Frame: {event['frame_number']} | "
                f"Track ID: {event['track_id']}"
            )


if __name__ == "__main__":
    main()