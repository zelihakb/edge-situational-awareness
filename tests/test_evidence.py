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