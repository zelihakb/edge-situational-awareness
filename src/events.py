from collections import defaultdict, deque

import numpy as np


class TrackEventEngine:
    """Manage per-track motion, zone transitions, and loitering state."""

    def __init__(
        self,
        source_fps: float,
        zone_center: np.ndarray,
        motion_min_delta: float,
        zone_confirm_frames: int,
        loitering_threshold_seconds: float,
        motion_history_length: int,
    ) -> None:
        self.source_fps = source_fps
        self.zone_center = zone_center
        self.motion_min_delta = motion_min_delta
        self.zone_confirm_frames = zone_confirm_frames
        self.loitering_threshold_seconds = loitering_threshold_seconds
        self.motion_history_length = motion_history_length

        # Confirmed inside/outside state per track.
        self.zone_state: dict[int, bool] = {}

        # Candidate state used for transition debounce.
        self.zone_candidate_state: dict[int, bool] = {}
        self.zone_candidate_count: dict[int, int] = {}

        # Time-based state for loitering detection.
        self.zone_enter_frame: dict[int, int] = {}
        self.loitering_reported: set[int] = set()

        # Recent image-space distance to the zone center.
        self.motion_distance_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(
                maxlen=self.motion_history_length
            )
        )

        self.zone_entry_count = 0
        self.zone_exit_count = 0

    def update(
        self,
        track_id: int,
        zone_point: tuple[int, int],
        inside_zone: bool,
        frame_number: int,
    ) -> tuple[str, list[dict]]:
        """Update one track and return its motion label plus any new events."""

        current_frame = frame_number + 1
        events: list[dict] = []

        motion_direction = self._update_motion(
            track_id,
            zone_point,
        )

        self._update_zone_state(
            track_id,
            inside_zone,
            current_frame,
            events,
        )

        self._update_loitering(
            track_id,
            current_frame,
            events,
        )

        return motion_direction, events

    def _update_motion(
        self,
        track_id: int,
        zone_point: tuple[int, int],
    ) -> str:
        """Estimate image-space motion relative to the zone center."""

        current_distance = float(
            np.linalg.norm(
                np.asarray(
                    zone_point,
                    dtype=np.float32,
                )
                - self.zone_center
            )
        )

        history = self.motion_distance_history[
            track_id
        ]

        history.append(
            current_distance
        )

        if len(history) < self.motion_history_length:
            return "STABLE"

        distance_change = (
            history[-1]
            - history[0]
        )

        if abs(distance_change) < self.motion_min_delta:
            return "STABLE"

        return (
            "APPROACHING"
            if distance_change < 0
            else "MOVING_AWAY"
        )

    def _update_zone_state(
        self,
        track_id: int,
        inside_zone: bool,
        current_frame: int,
        events: list[dict],
    ) -> None:
        """Apply debounced zone transitions and emit entry/exit events."""

        if track_id not in self.zone_state:

            # Initial observation establishes state without emitting an event.
            self.zone_state[
                track_id
            ] = inside_zone

            self.zone_candidate_state[
                track_id
            ] = inside_zone

            self.zone_candidate_count[
                track_id
            ] = 0

            if inside_zone:
                self.zone_enter_frame[
                    track_id
                ] = current_frame

            return

        confirmed_state = self.zone_state[
            track_id
        ]

        if inside_zone == confirmed_state:

            # Raw state matches confirmed state, so cancel any candidate.
            self.zone_candidate_state[
                track_id
            ] = confirmed_state

            self.zone_candidate_count[
                track_id
            ] = 0

            return

        if (
            self.zone_candidate_state.get(
                track_id
            )
            == inside_zone
        ):
            self.zone_candidate_count[
                track_id
            ] += 1

        else:

            # A new candidate transition starts here.
            self.zone_candidate_state[
                track_id
            ] = inside_zone

            self.zone_candidate_count[
                track_id
            ] = 1

        if (
            self.zone_candidate_count[
                track_id
            ]
            < self.zone_confirm_frames
        ):
            return

        # OUTSIDE -> INSIDE
        if (
            not confirmed_state
            and inside_zone
        ):

            self.zone_enter_frame[
                track_id
            ] = current_frame

            self.loitering_reported.discard(
                track_id
            )

            self.zone_entry_count += 1

            events.append(
                {
                    "event_type": "ZONE_ENTRY",
                    "frame_number": current_frame,
                    "track_id": track_id,
                }
            )

        # INSIDE -> OUTSIDE
        elif (
            confirmed_state
            and not inside_zone
        ):

            self.zone_enter_frame.pop(
                track_id,
                None,
            )

            self.loitering_reported.discard(
                track_id
            )

            self.zone_exit_count += 1

            events.append(
                {
                    "event_type": "ZONE_EXIT",
                    "frame_number": current_frame,
                    "track_id": track_id,
                }
            )

        # Confirm the transition only after debounce succeeds.
        self.zone_state[
            track_id
        ] = inside_zone

        self.zone_candidate_state[
            track_id
        ] = inside_zone

        self.zone_candidate_count[
            track_id
        ] = 0

    def _update_loitering(
        self,
        track_id: int,
        current_frame: int,
        events: list[dict],
    ) -> None:
        """Emit one loitering event per continuous stay inside the zone."""

        if (
            self.zone_state.get(track_id) is not True
            or track_id not in self.zone_enter_frame
            or track_id in self.loitering_reported
        ):
            return

        time_inside_seconds = (
            current_frame
            - self.zone_enter_frame[
                track_id
            ]
        ) / self.source_fps

        if (
            time_inside_seconds
            < self.loitering_threshold_seconds
        ):
            return

        self.loitering_reported.add(
            track_id
        )

        events.append(
            {
                "event_type": "LOITERING",
                "frame_number": current_frame,
                "track_id": track_id,
                "duration_seconds": time_inside_seconds,
            }
        )