from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
    PROJECT_ROOT,
    TARGET_CLASSES,
)


INPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "test.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)


def main() -> None:
    """Run a simple image-detection smoke test."""

    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.predict(
        source=str(INPUT_IMAGE),
        device=0,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        classes=TARGET_CLASSES,
        save=True,
        project=str(OUTPUT_DIR),
        name="python_detection",
    )

    print(
        "Detection completed."
    )

    print(
        f"Detected objects: "
        f"{len(results[0].boxes)}"
    )


if __name__ == "__main__":
    main()