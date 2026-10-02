from pathlib import Path
import shutil
import subprocess

import cv2


def open_video(
    input_path: Path,
) -> cv2.VideoCapture:
    """Open a video file and validate that OpenCV can read it."""

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: "
            f"{input_path}"
        )

    return capture


def get_video_metadata(
    capture: cv2.VideoCapture,
) -> tuple[int, int, float, int]:
    """Read width, height, FPS, and frame count."""

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if source_fps <= 0:
        source_fps = 30.0

    return (
        frame_width,
        frame_height,
        source_fps,
        total_frames,
    )


def create_video_writer(
    output_path: Path,
    frame_width: int,
    frame_height: int,
    source_fps: float,
) -> cv2.VideoWriter:
    """Create the temporary MJPG writer."""

    codec = (
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        source_fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create temporary video: "
            f"{output_path}"
        )

    return writer


def convert_to_mp4(
    input_path: Path,
    output_path: Path,
) -> None:
    """Convert temporary MJPG AVI output to H.264 MP4."""

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    print()
    print("=" * 60)
    print("FFMPEG CONVERSION")
    print("=" * 60)

    subprocess.run(
        command,
        check=True,
    )

    print(
        "MP4 conversion completed."
    )


def cleanup_temp_video(
    temp_path: Path,
    output_path: Path,
) -> None:
    """Delete the temporary AVI only after a valid MP4 exists."""

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and temp_path.exists()
    ):
        temp_path.unlink()