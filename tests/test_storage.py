import sqlite3

import pytest

from src.storage import EventStore


def log_sample_event(store: EventStore) -> int:
    """Insert one deterministic event for storage tests."""
    return store.log_event(
        video_time_seconds=3.0,
        frame_number=73,
        track_id=6,
        class_id=0,
        class_name="person",
        confidence=0.81,
        event_type="LOITERING",
        zone_name="CONTROLLED_ZONE",
        duration_seconds=3.0,
    )


def test_log_event_persists_expected_values(tmp_path):
    database_path = tmp_path / "events.db"
    store = EventStore(database_path)

    event_id = log_sample_event(store)

    row = store.connection.execute(
        """
        SELECT
            event_id,
            video_time_seconds,
            frame_number,
            track_id,
            class_id,
            class_name,
            confidence,
            event_type,
            zone_name,
            duration_seconds,
            review_status,
            reviewed_at
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    store.close()

    assert row == (
        1,
        3.0,
        73,
        6,
        0,
        "person",
        0.81,
        "LOITERING",
        "CONTROLLED_ZONE",
        3.0,
        "review",
        None,
    )


def test_new_event_defaults_to_review_status(tmp_path):
    store = EventStore(tmp_path / "events.db")

    event_id = log_sample_event(store)

    row = store.connection.execute(
        """
        SELECT review_status, reviewed_at
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    store.close()

    assert row == ("review", None)


@pytest.mark.parametrize(
    "review_status",
    [
        "normal",
        "confirmed",
    ],
)
def test_reviewed_event_sets_timestamp(
    tmp_path,
    review_status,
):
    store = EventStore(tmp_path / "events.db")

    event_id = log_sample_event(store)

    updated = store.update_review_status(
        event_id,
        review_status,
    )

    row = store.connection.execute(
        """
        SELECT review_status, reviewed_at
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    store.close()

    assert updated is True
    assert row[0] == review_status
    assert row[1] is not None
    assert row[1].endswith("+00:00")


def test_resetting_status_to_review_clears_review_timestamp(tmp_path):
    store = EventStore(tmp_path / "events.db")

    event_id = log_sample_event(store)

    store.update_review_status(
        event_id,
        "confirmed",
    )

    store.update_review_status(
        event_id,
        "review",
    )

    row = store.connection.execute(
        """
        SELECT review_status, reviewed_at
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()

    store.close()

    assert row == ("review", None)


def test_invalid_review_status_raises_value_error(tmp_path):
    store = EventStore(tmp_path / "events.db")

    event_id = log_sample_event(store)

    with pytest.raises(
        ValueError,
        match="Invalid review status",
    ):
        store.update_review_status(
            event_id,
            "dangerous",
        )

    store.close()


def test_updating_missing_event_returns_false(tmp_path):
    store = EventStore(tmp_path / "events.db")

    updated = store.update_review_status(
        event_id=999,
        review_status="confirmed",
    )

    store.close()

    assert updated is False


def test_legacy_database_is_migrated_without_losing_events(tmp_path):
    database_path = tmp_path / "legacy_events.db"

    connection = sqlite3.connect(database_path)

    connection.execute(
        """
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            video_time_seconds REAL NOT NULL,
            frame_number INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            event_type TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            duration_seconds REAL
        )
        """
    )

    connection.execute(
        """
        INSERT INTO events (
            created_at,
            video_time_seconds,
            frame_number,
            track_id,
            class_id,
            class_name,
            confidence,
            event_type,
            zone_name,
            duration_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-10-01T12:00:00+00:00",
            1.25,
            31,
            2,
            0,
            "person",
            0.87,
            "ZONE_ENTRY",
            "CONTROLLED_ZONE",
            None,
        ),
    )

    connection.commit()
    connection.close()

    # Opening the legacy database should automatically apply
    # the lightweight review-column migration.
    store = EventStore(database_path)

    columns = {
        row[1]
        for row in store.connection.execute(
            "PRAGMA table_info(events)"
        )
    }

    row = store.connection.execute(
        """
        SELECT
            event_id,
            event_type,
            review_status,
            reviewed_at
        FROM events
        """
    ).fetchone()

    store.close()

    assert "review_status" in columns
    assert "reviewed_at" in columns

    assert row == (
        1,
        "ZONE_ENTRY",
        "review",
        None,
    )