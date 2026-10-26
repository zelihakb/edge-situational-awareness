import pytest

from src.evaluation import calculate_event_metrics,match_event_counts


def test_event_metrics():
    metrics = calculate_event_metrics(
        tp=3,
        fp=1,
        fn=2,
    )

    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.60)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_zero_event_counts():
    metrics = calculate_event_metrics(0, 0, 0)

    assert metrics == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        calculate_event_metrics(-1, 0, 0)

def test_event_matching_with_frame_tolerance():
    ground_truth = [
        {"event_type": "ZONE_ENTRY", "frame_number": 10},
        {"event_type": "ZONE_EXIT", "frame_number": 30},
    ]

    predictions = [
        {"event_type": "ZONE_ENTRY", "frame_number": 12},
        {"event_type": "ZONE_EXIT", "frame_number": 37},
    ]

    counts = match_event_counts(
        ground_truth,
        predictions,
        tolerance_frames=3,
    )

    assert counts == {"tp": 1, "fp": 1, "fn": 1}


def test_event_matching_does_not_double_count():
    ground_truth = [
        {"event_type": "ZONE_ENTRY", "frame_number": 10},
    ]

    predictions = [
        {"event_type": "ZONE_ENTRY", "frame_number": 9},
        {"event_type": "ZONE_ENTRY", "frame_number": 11},
    ]

    counts = match_event_counts(ground_truth, predictions)

    assert counts == {"tp": 1, "fp": 1, "fn": 0}


def test_event_matching_maximizes_matches():
    ground_truth = [
        {"event_type": "ZONE_ENTRY", "frame_number": 10},
        {"event_type": "ZONE_ENTRY", "frame_number": 13},
    ]

    predictions = [
        {"event_type": "ZONE_ENTRY", "frame_number": 12},
        {"event_type": "ZONE_ENTRY", "frame_number": 15},
    ]

    counts = match_event_counts(
        ground_truth,
        predictions,
        tolerance_frames=2,
    )

    assert counts == {"tp": 2, "fp": 0, "fn": 0}


def test_different_event_types_do_not_match():
    ground_truth = [
        {"event_type": "ZONE_ENTRY", "frame_number": 10},
    ]

    predictions = [
        {"event_type": "ZONE_EXIT", "frame_number": 10},
    ]

    counts = match_event_counts(ground_truth, predictions)

    assert counts == {"tp": 0, "fp": 1, "fn": 1}