import json

from src.run_manifest import (
    build_run_manifest,
    save_run_manifest,
)
from src.storage import EventStore


def test_different_runs_keep_their_own_configuration(tmp_path):
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    zone_path = tmp_path / "zone.json"

    video.write_bytes(b"sample video")
    model.write_bytes(b"sample model")

    runs = [
        ("run_first", [[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]),
        ("run_second", [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]]),
    ]

    store = EventStore(tmp_path / "events.db")

    try:
        for run_id, points in runs:
            zone_path.write_text(
                json.dumps({
                    "name": "CONTROLLED_ZONE",
                    "normalized_points": points,
                }),
                encoding="utf-8",
            )

            manifest = build_run_manifest(
                input_video=video,
                model_path=model,
                zone_config_path=zone_path,
                frame_width=1280,
                frame_height=720,
                source_fps=24.0,
                total_frames=240,
                settings={"zone_confirm_frames": 3},
                run_id=run_id,
            )

            save_run_manifest(
                manifest,
                tmp_path / "evidence" / run_id / "manifest.json",
            )

            store.log_event(
                video_time_seconds=0.5,
                frame_number=12,
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                event_type="ZONE_ENTRY",
                zone_name=manifest["zone_config"]["name"],
                run_id=run_id,
            )

        rows = store.connection.execute(
            "SELECT run_id FROM events ORDER BY event_id"
        ).fetchall()

        assert rows == [
            ("run_first",),
            ("run_second",),
        ]

        for run_id, expected_points in runs:
            path = tmp_path / "evidence" / run_id / "manifest.json"
            saved = json.loads(path.read_text(encoding="utf-8"))

            assert saved["run_id"] == run_id
            assert (
                saved["zone_config"]["normalized_points"]
                == expected_points
            )
    finally:
        store.close()