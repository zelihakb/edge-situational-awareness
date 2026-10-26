import pytest

from src.evaluation import calculate_event_metrics


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