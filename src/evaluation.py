def calculate_event_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    """Calculate precision, recall and F1 from event counts.

Counts must be non-negative integers.
Undefined ratios with zero denominators are reported as 0.0.
"""

    if any(type(n) is not int or n < 0 for n in (tp, fp, fn)):
        raise ValueError("Event counts must be non-negative integers.")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
def _validate_event_records(events: list[dict], source: str) -> None:
    """Validate event records before matching."""

    if not isinstance(events, list):
        raise ValueError(f"{source} must be a list.")

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"{source}[{index}] must be a dictionary."
            )

        event_type = event.get("event_type")
        frame_number = event.get("frame_number")

        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError(
                f"{source}[{index}] has an invalid event_type."
            )

        if type(frame_number) is not int or frame_number < 0:
            raise ValueError(
                f"{source}[{index}] has an invalid frame_number."
            )
def match_event_counts(
    ground_truth: list[dict],
    predictions: list[dict],
    tolerance_frames: int = 3,
) -> dict[str, int]:
    """Match events by type and frame within a tolerance."""

    if type(tolerance_frames) is not int or tolerance_frames < 0:
        raise ValueError("Frame tolerance must be a non-negative integer.")
    _validate_event_records(ground_truth, "ground_truth")
    _validate_event_records(predictions, "predictions")

    matched = 0

    event_types = {
        event["event_type"]
        for event in ground_truth + predictions
    }

    for event_type in event_types:
        truth_frames = sorted(
            event["frame_number"]
            for event in ground_truth
            if event["event_type"] == event_type
        )

        predicted_frames = sorted(
            event["frame_number"]
            for event in predictions
            if event["event_type"] == event_type
        )

        rows = len(truth_frames) + 1
        cols = len(predicted_frames) + 1

        dp = [
            [(0, 0) for _ in range(cols)]
            for _ in range(rows)
        ]

        for i in range(1, rows):
            for j in range(1, cols):
                best = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

                frame_difference = abs(
                    truth_frames[i - 1]
                    - predicted_frames[j - 1]
                )

                if frame_difference <= tolerance_frames:
                    previous = dp[i - 1][j - 1]

                    best = max(
                        best,
                        (
                            previous[0] + 1,
                            previous[1] - frame_difference,
                        ),
                    )

                dp[i][j] = best

        matched += dp[-1][-1][0]

    return {
        "tp": matched,
        "fp": len(predictions) - matched,
        "fn": len(ground_truth) - matched,
    }