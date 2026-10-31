import json
from hashlib import sha256

import pytest

from src.run_manifest import (
    build_run_manifest,
    save_run_manifest,
)


def test_manifest_captures_run_configuration(tmp_path):
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    zone = tmp_path / "zone.json"

    video.write_bytes(b"sample video")
    model.write_bytes(b"sample model")

    zone_config = {
        "name": "CONTROLLED_ZONE",
        "normalized_points": [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.5, 0.9],
        ],
    }

    zone.write_text(
        json.dumps(zone_config),
        encoding="utf-8",
    )

    manifest = build_run_manifest(
        input_video=video,
        model_path=model,
        zone_config_path=zone,
        frame_width=1280,
        frame_height=720,
        source_fps=24.0,
        total_frames=240,
        settings={"zone_confirm_frames": 3},
        run_id="run_test123",
    )

    assert manifest["run_id"] == "run_test123"
    assert manifest["input_video"]["resolution"] == [1280, 720]
    assert manifest["input_video"]["source_fps"] == 24.0

    assert manifest["input_video"]["sha256"] == (
        sha256(b"sample video").hexdigest()
    )

    assert manifest["model"]["sha256"] == (
        sha256(b"sample model").hexdigest()
    )

    assert manifest["model"]["tracker"] == "bytetrack.yaml"
    assert manifest["zone_config"] == zone_config
    assert manifest["settings"]["zone_confirm_frames"] == 3
    assert manifest["created_at_utc"].endswith("+00:00")


def test_manifest_can_be_saved_and_loaded(tmp_path):
    manifest = {
        "run_id": "run_test123",
        "settings": {"threshold": 0.35},
    }

    path = tmp_path / "runs" / "run_test123" / "manifest.json"

    saved_path = save_run_manifest(manifest, path)

    assert saved_path == path
    assert json.loads(
        path.read_text(encoding="utf-8")
    ) == manifest


def test_manifest_cannot_overwrite_existing_run(tmp_path):
    path = tmp_path / "manifest.json"

    save_run_manifest({"run_id": "first"}, path)

    with pytest.raises(FileExistsError):
        save_run_manifest({"run_id": "second"}, path)

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["run_id"] == "first"