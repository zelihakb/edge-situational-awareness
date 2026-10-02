import subprocess
import time

import numpy as np
import torch
from ultralytics import YOLO

from config import (
    BENCHMARK_SKIP_FRAMES,
    CONFIDENCE_THRESHOLD,
    EVENT_DB_PATH,
    FPS_WINDOW,
    IMAGE_SIZE,
    INPUT_VIDEO,
    LOITERING_THRESHOLD_SECONDS,
    MODEL_PATH,
    MOTION_HISTORY_LENGTH,
    MOTION_MIN_DELTA_RATIO,
    OUTPUT_VIDEO,
    TARGET_CLASSES,
    TEMP_VIDEO,
    TRAIL_LENGTH,
    WARMUP_RUNS,
    ZONE_CONFIRM_FRAMES,
)
from events import TrackEventEngine
from metrics import PipelineMetrics
from storage import EventStore
from video_io import (
    cleanup_temp_video,
    convert_to_mp4,
    create_video_writer,
    get_video_metadata,
    open_video,
)
from visualization import (
    TrackVisualizer,
    draw_pipeline_overlay,
)
from zone import (
    draw_zone,
    get_zone_center,
    get_zone_polygon,
    is_point_inside_zone,
)


def synchronize_cuda() -> None:
    """Wait for queued CUDA work for accurate latency measurements."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warm_up_model(
    model: YOLO,
    device: int | str,
    frame_width: int,
    frame_height: int,
) -> None:
    """Run dummy inference passes before benchmark measurements."""

    print(
        "GPU/model warm-up started..."
    )

    dummy_frame = np.zeros(
        (
            frame_height,
            frame_width,
            3,
        ),
        dtype=np.uint8,
    )

    for run_number in range(
        WARMUP_RUNS
    ):

        model.predict(
            source=dummy_frame,
            device=device,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=TARGET_CLASSES,
            verbose=False,
        )

        synchronize_cuda()

        print(
            f"Warm-up: "
            f"{run_number + 1}/"
            f"{WARMUP_RUNS}"
        )

    print(
        "Warm-up completed."
    )

    print(
        "=" * 60
    )


def print_event(
    event: dict,
) -> None:
    """Print one structured event."""

    event_type = (
        event[
            "event_type"
        ]
    )

    if event_type == "LOITERING":

        print(
            f"{event_type} | "
            f"Frame: "
            f"{event['frame_number']} | "
            f"Track ID: "
            f"{event['track_id']} | "
            f"Duration: "
            f"{event['duration_seconds']:.1f}s"
        )

        return

    print(
        f"{event_type} | "
        f"Frame: "
        f"{event['frame_number']} | "
        f"Track ID: "
        f"{event['track_id']}"
    )


def print_video_info(
    frame_width: int,
    frame_height: int,
    source_fps: float,
    total_frames: int,
) -> None:
    """Print input and runtime information."""

    print()

    print(
        "=" * 60
    )

    print(
        "VIDEO TRACKING"
    )

    print(
        "=" * 60
    )

    print(
        f"Input       : "
        f"{INPUT_VIDEO}"
    )

    print(
        f"Resolution  : "
        f"{frame_width}"
        f"x"
        f"{frame_height}"
    )

    print(
        f"Source FPS  : "
        f"{source_fps:.2f}"
    )

    print(
        f"Frames      : "
        f"{total_frames}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "Device      : CPU"
        )

    print(
        "=" * 60
    )


def main() -> None:

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: "
            f"{INPUT_VIDEO}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: "
            f"{MODEL_PATH}"
        )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device: int | str = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    # -----------------------------------------------------
    # MODEL + VIDEO SETUP
    # -----------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    video_capture = open_video(
        INPUT_VIDEO
    )

    (
        frame_width,
        frame_height,
        source_fps,
        total_frames,
    ) = get_video_metadata(
        video_capture
    )

    zone_polygon = get_zone_polygon(
        frame_width,
        frame_height,
    )

    zone_center = get_zone_center(
        zone_polygon
    )

    frame_diagonal = np.hypot(
        frame_width,
        frame_height,
    )

    motion_min_delta = (
        MOTION_MIN_DELTA_RATIO
        * frame_diagonal
    )

    print_video_info(
        frame_width,
        frame_height,
        source_fps,
        total_frames,
    )

    warm_up_model(
        model,
        device,
        frame_width,
        frame_height,
    )

    try:

        video_writer = (
            create_video_writer(
                TEMP_VIDEO,
                frame_width,
                frame_height,
                source_fps,
            )
        )

    except Exception:

        video_capture.release()
        raise

    # -----------------------------------------------------
    # PIPELINE COMPONENTS
    # -----------------------------------------------------

    event_engine = (
        TrackEventEngine(
            source_fps=source_fps,
            zone_center=zone_center,
            motion_min_delta=(
                motion_min_delta
            ),
            zone_confirm_frames=(
                ZONE_CONFIRM_FRAMES
            ),
            loitering_threshold_seconds=(
                LOITERING_THRESHOLD_SECONDS
            ),
            motion_history_length=(
                MOTION_HISTORY_LENGTH
            ),
        )
    )

    metrics = PipelineMetrics(
        fps_window=FPS_WINDOW,
        benchmark_skip_frames=(
            BENCHMARK_SKIP_FRAMES
        ),
    )

    visualizer = TrackVisualizer(
        trail_length=TRAIL_LENGTH
    )
    event_store = EventStore(
    EVENT_DB_PATH
)

    # -----------------------------------------------------
    # FRAME LOOP
    # -----------------------------------------------------

    try:

        while True:

            frame_pipeline_start = (
                time.perf_counter()
            )

            success, frame = (
                video_capture.read()
            )

            if not success:
                break

            # ---------------------------------------------
            # YOLO + BYTETRACK
            # ---------------------------------------------

            synchronize_cuda()

            tracking_start = (
                time.perf_counter()
            )

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                classes=TARGET_CLASSES,
                verbose=False,
            )

            synchronize_cuda()

            tracking_latency_ms = (
                time.perf_counter()
                - tracking_start
            ) * 1000

            result = results[0]

            inference_latency_ms = float(
                result.speed.get(
                    "inference",
                    0.0,
                )
            )

            # ---------------------------------------------
            # ANNOTATION
            # ---------------------------------------------

            annotated_frame = (
                result.plot()
            )

            draw_zone(
                annotated_frame,
                zone_polygon,
            )

            boxes = (
                result.boxes
            )

            # ---------------------------------------------
            # TRACK PROCESSING
            # ---------------------------------------------

            if (
                boxes is not None
                and boxes.id is not None
            ):

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                box_centers = (
                    boxes.xywh
                    .cpu()
                    .tolist()
                )

                bounding_boxes = (
                    boxes.xyxy
                    .cpu()
                    .tolist()
                )
                class_ids = (
                    boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .tolist()
                )

                for (
                    box,
                    bbox,
                    track_id,
                    class_id,
                    confidence,
                ) in zip(
                    box_centers,
                    bounding_boxes,
                    track_ids,
                    class_ids,
                    confidences,
                ):

                    center = (
                        int(box[0]),
                        int(box[1]),
                    )

                    (
                        x1,
                        _y1,
                        x2,
                        y2,
                    ) = bbox

                    # Bottom-center approximates ground contact.
                    zone_point = (
                        int(
                            (
                                x1
                                + x2
                            )
                            / 2
                        ),
                        int(y2),
                    )

                    inside_zone = (
                        is_point_inside_zone(
                            zone_point,
                            zone_polygon,
                        )
                    )

                    (
                        motion_direction,
                        events,
                    ) = event_engine.update(
                        track_id=track_id,
                        zone_point=zone_point,
                        inside_zone=inside_zone,
                        frame_number=(
                            metrics.frame_count
                        ),
                    )

                    for event in events:

                        print_event(
                            event
                        )

                        class_name = str(
                            model.names[
                                class_id
                            ]
                        )

                        video_time_seconds = (
                            (
                                event[
                                    "frame_number"
                                ]
                                - 1
                            )
                            / source_fps
                        )

                        event_id = event_store.log_event(
                            video_time_seconds=(
                                video_time_seconds
                            ),
                            frame_number=(
                                event[
                                    "frame_number"
                                ]
                            ),
                            track_id=(
                                event[
                                    "track_id"
                                ]
                            ),
                            class_id=(
                                class_id
                            ),
                            class_name=(
                                class_name
                            ),
                            confidence=float(
                                confidence
                            ),
                            event_type=(
                                event[
                                    "event_type"
                                ]
                            ),
                            zone_name=(
                                "CONTROLLED_ZONE"
                            ),
                            duration_seconds=(
                                event.get(
                                    "duration_seconds"
                                )
                            ),
                        )

                        print(
                            f"DB_EVENT | "
                            f"ID: {event_id}"
                        )

                    visualizer.draw_track(
                        frame=annotated_frame,
                        track_id=track_id,
                        center=center,
                        zone_point=zone_point,
                        inside_zone=inside_zone,
                        motion_direction=(
                            motion_direction
                        ),
                    )

            # ---------------------------------------------
            # PIPELINE OVERLAY
            # ---------------------------------------------

            draw_pipeline_overlay(
                frame=annotated_frame,
                rolling_fps=(
                    metrics.rolling_fps
                ),
                tracking_latency_ms=(
                    tracking_latency_ms
                ),
                inference_latency_ms=(
                    inference_latency_ms
                ),
                zone_entry_count=(
                    event_engine.zone_entry_count
                ),
                zone_exit_count=(
                    event_engine.zone_exit_count
                ),
            )

            # ---------------------------------------------
            # WRITE + METRICS
            # ---------------------------------------------

            video_writer.write(
                annotated_frame
            )

            frame_end = (
                time.perf_counter()
            )

            metrics.record_frame(
                frame_pipeline_start=(
                    frame_pipeline_start
                ),
                frame_end=frame_end,
                tracking_latency_ms=(
                    tracking_latency_ms
                ),
                inference_latency_ms=(
                    inference_latency_ms
                ),
            )

            # ---------------------------------------------
            # TERMINAL PROGRESS
            # ---------------------------------------------

            if (
                metrics.frame_count
                % 30
                == 0
            ):

                if total_frames > 0:

                    progress = (
                        metrics.frame_count
                        / total_frames
                    ) * 100

                    print(
                        f"{metrics.frame_count}/"
                        f"{total_frames} frames "
                        f"({progress:.1f}%) | "
                        f"Rolling FPS: "
                        f"{metrics.rolling_fps:.1f}"
                    )

                else:

                    print(
                        f"{metrics.frame_count} frames | "
                        f"Rolling FPS: "
                        f"{metrics.rolling_fps:.1f}"
                    )

    finally:

        video_capture.release()
        video_writer.release()
        event_store.close()
    # -----------------------------------------------------
    # BENCHMARK
    # -----------------------------------------------------

    metrics.print_summary()

    # -----------------------------------------------------
    # FINAL MP4
    # -----------------------------------------------------

    try:

        convert_to_mp4(
            TEMP_VIDEO,
            OUTPUT_VIDEO,
        )

    except (
        subprocess.CalledProcessError
    ) as error:

        print()

        print(
            "FFmpeg conversion failed."
        )

        print(
            f"Temporary AVI was kept at: "
            f"{TEMP_VIDEO}"
        )

        raise RuntimeError(
            "MP4 conversion failed."
        ) from error

    cleanup_temp_video(
        TEMP_VIDEO,
        OUTPUT_VIDEO,
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ALL PROCESSING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Final video : "
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"File size   : "
        f"{OUTPUT_VIDEO.stat().st_size / 1024**2:.2f} MB"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()