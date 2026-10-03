import sys

from config import EVENT_DB_PATH
from storage import EventStore


VALID_STATUSES = {
    "review",
    "normal",
    "confirmed",
}


def main() -> None:
    """Update the human review status of one stored event."""

    if len(sys.argv) != 3:
        print(
            "Usage: "
            "python src/review_event.py "
            "<event_id> "
            "<review|normal|confirmed>"
        )
        raise SystemExit(1)

    event_id_raw = sys.argv[1]
    review_status = sys.argv[2].lower()

    try:
        event_id = int(
            event_id_raw
        )
    except ValueError:
        print(
            f"Invalid event_id: "
            f"{event_id_raw}"
        )
        raise SystemExit(1)

    if review_status not in VALID_STATUSES:
        print(
            f"Invalid status: "
            f"{review_status}"
        )

        print(
            "Valid statuses: "
            "review, normal, confirmed"
        )

        raise SystemExit(1)

    event_store = EventStore(
        EVENT_DB_PATH
    )

    try:
        updated = (
            event_store.update_review_status(
                event_id=event_id,
                review_status=review_status,
            )
        )

    finally:
        event_store.close()

    if not updated:
        print(
            f"Event ID {event_id} "
            f"was not found."
        )

        raise SystemExit(1)

    print(
        f"Event ID {event_id} "
        f"updated to "
        f"'{review_status}'."
    )


if __name__ == "__main__":
    main()