import cv2
import numpy as np

from config import INPUT_VIDEO, ZONE_CONFIG_PATH
from zone import (
    load_zone_config,
    save_zone_config,
    validate_normalized_points,
)


def main():
    # Read the first frame to use as our editing canvas.
    capture = cv2.VideoCapture(str(INPUT_VIDEO))

    if not capture.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {INPUT_VIDEO}"
        )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError("Cannot read the first video frame.")

    height, width = frame.shape[:2]

    # Fit the preview on a laptop screen.
    scale = min(960 / width, 540 / height, 1.0)

    display_width = round(width * scale)
    display_height = round(height * scale)

    preview = cv2.resize(
        frame,
        (display_width, display_height),
    )

    # Start with our currently saved polygon.
    points = list(load_zone_config(ZONE_CONFIG_PATH))

    window_name = "SP14 - Zone Editor"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            normalized_x = x / display_width
            normalized_y = y / display_height

            points.append(
                (
                    min(1.0, max(0.0, normalized_x)),
                    min(1.0, max(0.0, normalized_y)),
                )
            )

    cv2.setMouseCallback(window_name, on_mouse)

    print("\nZONE EDITOR")
    print("C: Clear all points")
    print("Left click: Add a point")
    print("U: Undo last point")
    print("S: Save polygon")
    print("Q or ESC: Exit without saving")

    try:
        while True:
            canvas = preview.copy()

            pixel_points = [
                (
                    min(display_width - 1, round(x * display_width)),
                    min(display_height - 1, round(y * display_height)),
                )
                for x, y in points
            ]

            if len(pixel_points) >= 2:
                polygon = np.array(
                    pixel_points,
                    dtype=np.int32,
                )

                cv2.polylines(
                    canvas,
                    [polygon],
                    isClosed=len(pixel_points) >= 3,
                    color=(0, 255, 255),
                    thickness=2,
                )

            for index, point in enumerate(pixel_points):
                cv2.circle(
                    canvas,
                    point,
                    5,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    canvas,
                    str(index + 1),
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            cv2.rectangle(
                canvas,
                (0, 0),
                (display_width, 35),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                canvas,
                "C: Clear | Click: Add | U: Undo | S: Save | Q: Quit",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("c"):
                points.clear()

            elif key == ord("u"):
                if points:
                    points.pop()

            elif key == ord("s"):
                try:
                    validated_points = validate_normalized_points(
                        points
                    )
                except ValueError as error:
                    print(f"Cannot save polygon: {error}")
                else:
                    save_zone_config(
                        ZONE_CONFIG_PATH,
                        validated_points,
                    )

                    print(f"Zone saved: {ZONE_CONFIG_PATH}")
                    break

            elif key in (ord("q"), 27):
                print("Editor closed without saving.")
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()