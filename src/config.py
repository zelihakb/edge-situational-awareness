from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "test_video.mp4"
)

TEMP_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "tracked_video_temp.avi"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "tracked_video.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "yolo11n.pt"
)

EVENT_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "events.db"
)


# =========================================================
# MODEL SETTINGS
# =========================================================

# COCO class IDs:
# person, bicycle, car, motorcycle, bus, truck
TARGET_CLASSES = [
    0,
    1,
    2,
    3,
    5,
    7,
]

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 640


# =========================================================
# TRACKING / EVENT SETTINGS
# =========================================================

TRAIL_LENGTH = 40
ZONE_CONFIRM_FRAMES = 3
LOITERING_THRESHOLD_SECONDS = 3.0
MOTION_HISTORY_LENGTH = 8
MOTION_MIN_DELTA_RATIO = 0.01


# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

WARMUP_RUNS = 3
FPS_WINDOW = 30
BENCHMARK_SKIP_FRAMES = 5