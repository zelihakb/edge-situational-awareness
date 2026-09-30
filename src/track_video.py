from collections import defaultdict, deque
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from zone import (
    draw_zone,
    get_zone_polygon,
    is_point_inside_zone,
)


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

MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"


# =========================================================
# MODEL SETTINGS
# =========================================================

# COCO class IDs:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

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

TRAIL_LENGTH = 40

# Zone state değişikliği gerçek event sayılmadan önce
# yeni durumun kaç gözlem boyunca devam etmesi gerektiği.
ZONE_CONFIRM_FRAMES = 3


# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

WARMUP_RUNS = 3

FPS_WINDOW = 30

# İlk gerçek frameler benchmark ortalamasına
# dahil edilmiyor.
BENCHMARK_SKIP_FRAMES = 5


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def synchronize_cuda() -> None:
    """
    CUDA aktifse GPU işlemlerinin bitmesini bekler.

    Doğru latency ölçümü için kullanılır.
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warm_up_model(
    model: YOLO,
    device: int | str,
    frame_width: int,
    frame_height: int,
) -> None:
    """
    Benchmark başlamadan önce modeli birkaç kez çalıştırır.

    Böylece ilk inference initialization maliyeti
    benchmark sonucunu gereksiz şekilde bozmaz.
    """

    print(
        "GPU/model warm-up başlatılıyor..."
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
        "Warm-up tamamlandı."
    )

    print(
        "=" * 60
    )


def convert_to_mp4(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Geçici MJPG AVI dosyasını
    H.264 MP4 dosyasına dönüştürür.
    """

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg bulunamadı. "
            "Terminalde "
            "'ffmpeg -version' "
            "çalışıyor mu kontrol et."
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

    print(
        "=" * 60
    )

    print(
        "FFMPEG DÖNÜŞÜMÜ"
    )

    print(
        "=" * 60
    )

    subprocess.run(
        command,
        check=True,
    )

    print(
        "MP4 dönüşümü tamamlandı."
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # 1. INPUT CHECK
    # -----------------------------------------------------

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Video bulunamadı:\n"
            f"{INPUT_VIDEO}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO modeli bulunamadı:\n"
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
    # 2. MODEL LOAD
    # -----------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # -----------------------------------------------------
    # 3. VIDEO INPUT
    # -----------------------------------------------------

    video_capture = (
        cv2.VideoCapture(
            str(INPUT_VIDEO)
        )
    )

    if not video_capture.isOpened():
        raise RuntimeError(
            "Video OpenCV tarafından "
            "açılamadı."
        )

    frame_width = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source_fps = float(
        video_capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    zone_polygon = (
        get_zone_polygon(
            frame_width,
            frame_height,
        )
    )

    if source_fps <= 0:
        source_fps = 30.0

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
        f"Girdi       : "
        f"{INPUT_VIDEO}"
    )

    print(
        f"Çözünürlük  : "
        f"{frame_width}"
        f"x"
        f"{frame_height}"
    )

    print(
        f"Kaynak FPS  : "
        f"{source_fps:.2f}"
    )

    print(
        f"Kare sayısı : "
        f"{total_frames}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:
        print(
            "Cihaz       : CPU"
        )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # 4. WARM-UP
    # -----------------------------------------------------

    warm_up_model(
        model=model,
        device=device,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # -----------------------------------------------------
    # 5. VIDEO OUTPUT
    # -----------------------------------------------------

    codec = (
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    video_writer = (
        cv2.VideoWriter(
            str(TEMP_VIDEO),
            codec,
            source_fps,
            (
                frame_width,
                frame_height,
            ),
        )
    )

    if not video_writer.isOpened():

        video_capture.release()

        raise RuntimeError(
            "Geçici çıktı videosu "
            "oluşturulamadı."
        )

    # -----------------------------------------------------
    # 6. TRACK / ZONE STATE
    # -----------------------------------------------------

    track_history = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )

    # Her track'in onaylanmış
    # inside/outside durumu.
    zone_state = {}

    # Debounce için aday durum.
    zone_candidate_state = {}

    # Aday durum kaç kez üst üste
    # gözlendi?
    zone_candidate_count = {}

    zone_entry_count = 0
    zone_exit_count = 0

    # -----------------------------------------------------
    # 7. METRICS
    # -----------------------------------------------------

    frame_number = 0

    fps_timestamps = deque(
        maxlen=FPS_WINDOW
    )

    rolling_fps = 0.0

    benchmark_frame_count = 0

    total_tracking_latency_ms = 0.0

    total_inference_latency_ms = 0.0

    total_end_to_end_latency_ms = 0.0

    processing_start = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # 8. FRAME LOOP
    # -----------------------------------------------------

    try:

        while True:

            # End-to-end timer
            # frame read dahil.
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
                (
                    time.perf_counter()
                    - tracking_start
                )
                * 1000
            )

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

            boxes = result.boxes

            # ---------------------------------------------
            # TRACKS + ZONE EVENT ENGINE
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

                for (
                    box,
                    bbox,
                    track_id,
                ) in zip(
                    box_centers,
                    bounding_boxes,
                    track_ids,
                ):

                    center_x = int(
                        box[0]
                    )

                    center_y = int(
                        box[1]
                    )

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = bbox

                    # Bounding box'ın
                    # bottom-center noktası.
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

                    # -------------------------------------
                    # DEBOUNCED STATE TRANSITION
                    # -------------------------------------

                    if (
                        track_id
                        not in zone_state
                    ):

                        # Nesneyi ilk kez görüyorsak
                        # bunu ENTRY/EXIT saymıyoruz.
                        zone_state[
                            track_id
                        ] = inside_zone

                        zone_candidate_state[
                            track_id
                        ] = inside_zone

                        zone_candidate_count[
                            track_id
                        ] = 0

                    else:

                        confirmed_state = (
                            zone_state[
                                track_id
                            ]
                        )

                        # Ham durum, onaylanmış
                        # durumla aynıysa:
                        if (
                            inside_zone
                            == confirmed_state
                        ):

                            # Bekleyen transition
                            # varsa iptal et.
                            zone_candidate_state[
                                track_id
                            ] = confirmed_state

                            zone_candidate_count[
                                track_id
                            ] = 0

                        else:

                            # Ham durum değişti.
                            #
                            # Daha önce de aynı yeni
                            # durumu görmüşsek streak +1.
                            if (
                                zone_candidate_state.get(
                                    track_id
                                )
                                == inside_zone
                            ):

                                zone_candidate_count[
                                    track_id
                                ] += 1

                            else:

                                # Yeni candidate başladı.
                                zone_candidate_state[
                                    track_id
                                ] = inside_zone

                                zone_candidate_count[
                                    track_id
                                ] = 1

                            # Candidate yeterli sayıda
                            # gözlendiyse gerçek event.
                            if (
                                zone_candidate_count[
                                    track_id
                                ]
                                >= ZONE_CONFIRM_FRAMES
                            ):

                                # OUTSIDE -> INSIDE
                                if (
                                    not confirmed_state
                                    and inside_zone
                                ):

                                    zone_entry_count += 1

                                    print(
                                        f"ZONE_ENTRY | "
                                        f"Frame: "
                                        f"{frame_number + 1} | "
                                        f"Track ID: "
                                        f"{track_id}"
                                    )

                                # INSIDE -> OUTSIDE
                                elif (
                                    confirmed_state
                                    and not inside_zone
                                ):

                                    zone_exit_count += 1

                                    print(
                                        f"ZONE_EXIT | "
                                        f"Frame: "
                                        f"{frame_number + 1} | "
                                        f"Track ID: "
                                        f"{track_id}"
                                    )

                                # ÖNEMLİ:
                                # confirmed state SADECE
                                # confirmation tamamlanınca
                                # değişiyor.
                                zone_state[
                                    track_id
                                ] = inside_zone

                                zone_candidate_state[
                                    track_id
                                ] = inside_zone

                                zone_candidate_count[
                                    track_id
                                ] = 0

                    # -------------------------------------
                    # VISUAL ZONE DEBUG POINT
                    # -------------------------------------

                    # Burada confirmed state değil,
                    # ham geometry sonucu gösteriliyor.
                    if inside_zone:

                        point_color = (
                            0,
                            255,
                            0,
                        )

                    else:

                        point_color = (
                            0,
                            0,
                            255,
                        )

                    cv2.circle(
                        annotated_frame,
                        zone_point,
                        radius=6,
                        color=point_color,
                        thickness=-1,
                    )

                    # -------------------------------------
                    # TRAJECTORY
                    # -------------------------------------

                    track_history[
                        track_id
                    ].append(
                        (
                            center_x,
                            center_y,
                        )
                    )

                    points = np.array(
                        track_history[
                            track_id
                        ],
                        dtype=np.int32,
                    ).reshape(
                        (
                            -1,
                            1,
                            2,
                        )
                    )

                    if (
                        len(points)
                        >= 2
                    ):

                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(
                                255,
                                255,
                                255,
                            ),
                            thickness=2,
                        )

            # ---------------------------------------------
            # FRAME COUNTER
            # ---------------------------------------------

            frame_number += 1

            # ---------------------------------------------
            # METRIC / EVENT OVERLAY
            # ---------------------------------------------

            cv2.putText(
                annotated_frame,
                (
                    f"Rolling FPS: "
                    f"{rolling_fps:.1f}"
                ),
                (
                    20,
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"Track call: "
                    f"{tracking_latency_ms:.1f} ms"
                ),
                (
                    20,
                    70,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"Inference: "
                    f"{inference_latency_ms:.1f} ms"
                ),
                (
                    20,
                    105,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"ZONE ENTRY: "
                    f"{zone_entry_count}"
                ),
                (
                    20,
                    140,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    255,
                    0,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"ZONE EXIT: "
                    f"{zone_exit_count}"
                ),
                (
                    20,
                    175,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            # ---------------------------------------------
            # WRITE FRAME
            # ---------------------------------------------

            video_writer.write(
                annotated_frame
            )

            # ---------------------------------------------
            # END-TO-END TIMING
            # ---------------------------------------------

            frame_end = (
                time.perf_counter()
            )

            end_to_end_latency_ms = (
                (
                    frame_end
                    - frame_pipeline_start
                )
                * 1000
            )

            # ---------------------------------------------
            # ROLLING FPS
            # ---------------------------------------------

            fps_timestamps.append(
                frame_end
            )

            if (
                len(fps_timestamps)
                >= 2
            ):

                time_span = (
                    fps_timestamps[-1]
                    - fps_timestamps[0]
                )

                if time_span > 0:

                    rolling_fps = (
                        (
                            len(
                                fps_timestamps
                            )
                            - 1
                        )
                        / time_span
                    )

            # ---------------------------------------------
            # BENCHMARK ACCUMULATION
            # ---------------------------------------------

            if (
                frame_number
                > BENCHMARK_SKIP_FRAMES
            ):

                benchmark_frame_count += 1

                total_tracking_latency_ms += (
                    tracking_latency_ms
                )

                total_inference_latency_ms += (
                    inference_latency_ms
                )

                total_end_to_end_latency_ms += (
                    end_to_end_latency_ms
                )

            # ---------------------------------------------
            # TERMINAL PROGRESS
            # ---------------------------------------------

            if (
                frame_number
                % 30
                == 0
            ):

                if total_frames > 0:

                    progress = (
                        frame_number
                        / total_frames
                    ) * 100

                    print(
                        f"{frame_number}/"
                        f"{total_frames} kare "
                        f"(%{progress:.1f}) | "
                        f"Rolling FPS: "
                        f"{rolling_fps:.1f}"
                    )

                else:

                    print(
                        f"{frame_number} kare | "
                        f"Rolling FPS: "
                        f"{rolling_fps:.1f}"
                    )

    finally:

        video_capture.release()

        video_writer.release()

    # =========================================================
    # 9. BENCHMARK RESULTS
    # =========================================================

    total_processing_time = (
        time.perf_counter()
        - processing_start
    )

    if frame_number == 0:

        raise RuntimeError(
            "Videodan hiçbir frame "
            "işlenemedi."
        )

    overall_pipeline_fps = (
        frame_number
        / total_processing_time
    )

    if (
        benchmark_frame_count
        > 0
    ):

        average_tracking_latency = (
            total_tracking_latency_ms
            / benchmark_frame_count
        )

        average_inference_latency = (
            total_inference_latency_ms
            / benchmark_frame_count
        )

        average_end_to_end_latency = (
            total_end_to_end_latency_ms
            / benchmark_frame_count
        )

    else:

        average_tracking_latency = 0.0
        average_inference_latency = 0.0
        average_end_to_end_latency = 0.0

    print()

    print(
        "=" * 60
    )

    print(
        "TRACKING BENCHMARK"
    )

    print(
        "=" * 60
    )

    print(
        f"İşlenen kare                 : "
        f"{frame_number}"
    )

    print(
        f"Benchmark kare               : "
        f"{benchmark_frame_count}"
    )

    print(
        f"Toplam işlem süresi          : "
        f"{total_processing_time:.2f} saniye"
    )

    print(
        f"Overall pipeline FPS         : "
        f"{overall_pipeline_fps:.2f}"
    )

    print(
        f"Final rolling FPS            : "
        f"{rolling_fps:.2f}"
    )

    print(
        f"Ort. inference latency       : "
        f"{average_inference_latency:.2f} ms"
    )

    print(
        f"Ort. track-call latency      : "
        f"{average_tracking_latency:.2f} ms"
    )

    print(
        f"Ort. end-to-end latency      : "
        f"{average_end_to_end_latency:.2f} ms"
    )

    print(
        "=" * 60
    )

    # =========================================================
    # 10. AUTOMATIC MP4 CONVERSION
    # =========================================================

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
            "FFmpeg dönüşümü "
            "başarısız oldu."
        )

        print(
            f"Geçici AVI korundu:\n"
            f"{TEMP_VIDEO}"
        )

        raise RuntimeError(
            "MP4 dönüşümü başarısız."
        ) from error

    if (
        OUTPUT_VIDEO.exists()
        and OUTPUT_VIDEO.stat().st_size > 0
        and TEMP_VIDEO.exists()
    ):

        TEMP_VIDEO.unlink()

    print()

    print(
        "=" * 60
    )

    print(
        "TÜM İŞLEMLER TAMAMLANDI"
    )

    print(
        "=" * 60
    )

    print(
        f"Final video : "
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"Dosya boyutu: "
        f"{OUTPUT_VIDEO.stat().st_size / 1024**2:.2f} MB"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()