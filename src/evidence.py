from pathlib import Path

import cv2
import numpy as np


def save_event_snapshot(
    frame: np.ndarray,
    output_dir: Path,
    event: dict,
) -> Path:
    """Save one visual snapshot for an emitted event."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_type = event["event_type"].lower()
    frame_number = int(event["frame_number"])
    track_id = int(event["track_id"])

    filename = (
        f"frame_{frame_number:06d}_"
        f"track_{track_id}_"
        f"{event_type}.jpg"
    )

    snapshot_path = output_dir / filename

    success = cv2.imwrite(
        str(snapshot_path),
        frame,
    )

    if not success:
        raise RuntimeError(
            f"Could not save event snapshot: "
            f"{snapshot_path}"
        )

    return snapshot_path