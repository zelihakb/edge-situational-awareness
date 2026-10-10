import cv2
import numpy as np

from src.evidence import save_event_snapshot


def test_event_snapshot_is_saved(tmp_path):
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    event = {
        "event_type": "ZONE_ENTRY",
        "frame_number": 31,
        "track_id": 2,
    }

    snapshot_path = save_event_snapshot(
        frame=frame,
        output_dir=tmp_path,
        event=event,
    )

    assert snapshot_path.exists()

    assert (
        snapshot_path.name
        == "frame_000031_track_2_zone_entry.jpg"
    )

    saved_frame = cv2.imread(
        str(snapshot_path)
    )

    assert saved_frame is not None
    assert saved_frame.shape == frame.shape
def test_snapshots_from_different_runs_are_preserved(tmp_path):
        event = {
            "event_type": "ZONE_ENTRY",
            "frame_number": 31,
            "track_id": 2,
    }

        first_frame = np.zeros(
                (100, 200, 3),
                dtype=np.uint8,
            )

        second_frame = np.full(
                (100, 200, 3),
                255,
                dtype=np.uint8,
            )

        first_snapshot = save_event_snapshot(
            frame=first_frame,
            output_dir=tmp_path / "run_first",
            event=event,
        )

        original_bytes = first_snapshot.read_bytes()

        second_snapshot = save_event_snapshot(
            frame=second_frame,
            output_dir=tmp_path / "run_second",
            event=event,
        )

        assert first_snapshot != second_snapshot
        assert first_snapshot.name == second_snapshot.name

        assert first_snapshot.is_file()
        assert second_snapshot.is_file()

        assert first_snapshot.read_bytes() == original_bytes
        assert first_snapshot.read_bytes() != second_snapshot.read_bytes()