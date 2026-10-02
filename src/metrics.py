from collections import deque
import time


class PipelineMetrics:
    """Track rolling FPS and aggregate latency statistics."""

    def __init__(
        self,
        fps_window: int,
        benchmark_skip_frames: int,
    ) -> None:

        self.fps_timestamps = deque(
            maxlen=fps_window
        )

        self.benchmark_skip_frames = (
            benchmark_skip_frames
        )

        self.frame_count = 0
        self.rolling_fps = 0.0

        self.benchmark_frame_count = 0

        self.total_tracking_latency_ms = 0.0
        self.total_inference_latency_ms = 0.0
        self.total_end_to_end_latency_ms = 0.0

        self.processing_start = (
            time.perf_counter()
        )

    def record_frame(
        self,
        frame_pipeline_start: float,
        frame_end: float,
        tracking_latency_ms: float,
        inference_latency_ms: float,
    ) -> None:
        """Record timing data after one frame has been fully processed."""

        self.frame_count += 1

        end_to_end_latency_ms = (
            frame_end
            - frame_pipeline_start
        ) * 1000

        self.fps_timestamps.append(
            frame_end
        )

        self._update_rolling_fps()

        if (
            self.frame_count
            <= self.benchmark_skip_frames
        ):
            return

        self.benchmark_frame_count += 1

        self.total_tracking_latency_ms += (
            tracking_latency_ms
        )

        self.total_inference_latency_ms += (
            inference_latency_ms
        )

        self.total_end_to_end_latency_ms += (
            end_to_end_latency_ms
        )

    def _update_rolling_fps(
        self,
    ) -> None:
        """Update FPS using timestamps in the rolling window."""

        if len(self.fps_timestamps) < 2:
            return

        time_span = (
            self.fps_timestamps[-1]
            - self.fps_timestamps[0]
        )

        if time_span <= 0:
            return

        self.rolling_fps = (
            len(self.fps_timestamps) - 1
        ) / time_span

    def get_summary(
        self,
    ) -> dict[str, float | int]:
        """Return aggregate benchmark values."""

        total_processing_time = (
            time.perf_counter()
            - self.processing_start
        )

        if self.frame_count == 0:
            raise RuntimeError(
                "No video frames were processed."
            )

        overall_pipeline_fps = (
            self.frame_count
            / total_processing_time
        )

        if self.benchmark_frame_count > 0:

            average_tracking_latency = (
                self.total_tracking_latency_ms
                / self.benchmark_frame_count
            )

            average_inference_latency = (
                self.total_inference_latency_ms
                / self.benchmark_frame_count
            )

            average_end_to_end_latency = (
                self.total_end_to_end_latency_ms
                / self.benchmark_frame_count
            )

        else:

            average_tracking_latency = 0.0
            average_inference_latency = 0.0
            average_end_to_end_latency = 0.0

        return {
            "frame_count": self.frame_count,
            "benchmark_frame_count": (
                self.benchmark_frame_count
            ),
            "total_processing_time": (
                total_processing_time
            ),
            "overall_pipeline_fps": (
                overall_pipeline_fps
            ),
            "rolling_fps": self.rolling_fps,
            "average_inference_latency": (
                average_inference_latency
            ),
            "average_tracking_latency": (
                average_tracking_latency
            ),
            "average_end_to_end_latency": (
                average_end_to_end_latency
            ),
        }

    def print_summary(
        self,
    ) -> None:
        """Print benchmark results."""

        summary = self.get_summary()

        print()
        print("=" * 60)
        print("TRACKING BENCHMARK")
        print("=" * 60)

        print(
            f"Processed frames              : "
            f"{summary['frame_count']}"
        )

        print(
            f"Benchmark frames              : "
            f"{summary['benchmark_frame_count']}"
        )

        print(
            f"Total processing time         : "
            f"{summary['total_processing_time']:.2f} s"
        )

        print(
            f"Overall pipeline FPS          : "
            f"{summary['overall_pipeline_fps']:.2f}"
        )

        print(
            f"Final rolling FPS             : "
            f"{summary['rolling_fps']:.2f}"
        )

        print(
            f"Average inference latency     : "
            f"{summary['average_inference_latency']:.2f} ms"
        )

        print(
            f"Average track-call latency    : "
            f"{summary['average_tracking_latency']:.2f} ms"
        )

        print(
            f"Average end-to-end latency    : "
            f"{summary['average_end_to_end_latency']:.2f} ms"
        )

        print("=" * 60)