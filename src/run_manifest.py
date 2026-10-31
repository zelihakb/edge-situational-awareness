import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


def calculate_file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash without loading the entire file."""

    digest = sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_run_manifest(
    *,
    input_video: Path,
    model_path: Path,
    zone_config_path: Path,
    frame_width: int,
    frame_height: int,
    source_fps: float,
    total_frames: int,
    settings: dict,
    run_id: str | None = None,
    tracker: str = "bytetrack.yaml",
) -> dict:
    """Capture the input files and settings for one processing run."""

    zone_config = json.loads(
        zone_config_path.read_text(encoding="utf-8")
    )

    return {
        "run_id": run_id or f"run_{uuid4().hex}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": {
            "filename": input_video.name,
            "sha256": calculate_file_hash(input_video),
            "resolution": [frame_width, frame_height],
            "source_fps": source_fps,
            "total_frames": total_frames,
        },
        "model": {
            "filename": model_path.name,
            "sha256": calculate_file_hash(model_path),
            "tracker": tracker,
        },
        "zone_config": zone_config,
        "settings": settings.copy(),
    }


def save_run_manifest(manifest: dict, path: Path) -> Path:
    """Save a manifest without overwriting an existing run."""

    content = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        file.write(content + "\n")

    return path