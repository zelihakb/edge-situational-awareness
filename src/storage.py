import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class EventStore:
    """Persist observable pipeline events in SQLite."""

    def __init__(
        self,
        database_path: Path,
    ) -> None:

        self.database_path = database_path

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.database_path
        )

        self._create_schema()

    def _create_schema(
        self,
    ) -> None:
        """Create the event table and apply lightweight schema migrations."""

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
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
                duration_seconds REAL,
                review_status TEXT NOT NULL DEFAULT 'review',
                reviewed_at TEXT
            )
            """
        )

        # Existing Sprint 5 databases do not yet have review columns.
        existing_columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info(events)"
            )
        }

        if "review_status" not in existing_columns:
            self.connection.execute(
                """
                ALTER TABLE events
                ADD COLUMN review_status TEXT NOT NULL DEFAULT 'review'
                """
            )

        if "reviewed_at" not in existing_columns:
            self.connection.execute(
                """
                ALTER TABLE events
                ADD COLUMN reviewed_at TEXT
                """
            )

        self.connection.commit()

    def log_event(
        self,
        *,
        video_time_seconds: float,
        frame_number: int,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        event_type: str,
        zone_name: str,
        duration_seconds: float | None = None,
    ) -> int:
        """Insert one event and return its generated database ID."""

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        cursor = self.connection.execute(
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
                created_at,
                video_time_seconds,
                frame_number,
                track_id,
                class_id,
                class_name,
                confidence,
                event_type,
                zone_name,
                duration_seconds,
            ),
        )

        self.connection.commit()

        return int(
            cursor.lastrowid
        )
    def update_review_status(
        self,
        event_id: int,
        review_status: str,
    ) -> bool:
        """Update the human review status of an existing event."""

        valid_statuses = {
            "review",
            "normal",
            "confirmed",
        }

        if review_status not in valid_statuses:
            raise ValueError(
                f"Invalid review status: {review_status}"
            )

        reviewed_at = None

        if review_status != "review":
            reviewed_at = datetime.now(
                timezone.utc
            ).isoformat()

        cursor = self.connection.execute(
            """
            UPDATE events
            SET
                review_status = ?,
                reviewed_at = ?
            WHERE event_id = ?
            """,
            (
                review_status,
                reviewed_at,
                event_id,
            ),
        )

        self.connection.commit()

        return cursor.rowcount > 0
    def close(
        self,
    ) -> None:
        """Close the SQLite connection."""

        self.connection.close()