import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class EventStore:
    """Persist observable pipeline events in SQLite."""

    def __init__(
        self,
        database_path: Path,
    ) -> None:

        self.database_path = (
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = (
            sqlite3.connect(
                self.database_path
            )
        )

        self._create_schema()

    def _create_schema(
        self,
    ) -> None:
        """Create the event table if it does not already exist."""

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
                duration_seconds REAL
            )
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

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

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

    def close(
        self,
    ) -> None:
        """Close the SQLite connection."""

        self.connection.close()