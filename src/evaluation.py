def calculate_event_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    """Calculate event detection precision, recall and F1."""

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