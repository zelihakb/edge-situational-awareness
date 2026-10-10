import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    EVENT_DB_PATH,
    OUTPUT_VIDEO,
    PROJECT_ROOT,
)
from storage import EventStore


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=("GPU-Accelerated Video Analytics & Event Detection"),
    page_icon="📹",
    layout="wide",
)


# =========================================================
# DATA ACCESS
# =========================================================

@st.cache_data
def load_events() -> pd.DataFrame:
    """Load stored events from SQLite into a DataFrame."""

    if not EVENT_DB_PATH.exists():
        return pd.DataFrame()

    connection = sqlite3.connect(
        EVENT_DB_PATH
    )

    try:
        dataframe = pd.read_sql_query(
            """
            SELECT
                event_id,
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
                review_status,
                reviewed_at,
                evidence_path
            FROM events
            ORDER BY event_id DESC
            """,
            connection,
        )

    finally:
        connection.close()

    return dataframe


def update_event_status(
    event_id: int,
    review_status: str,
) -> bool:
    """Update one event using the shared persistence layer."""

    event_store = EventStore(
        EVENT_DB_PATH
    )

    try:
        return event_store.update_review_status(
            event_id=event_id,
            review_status=review_status,
        )

    finally:
        event_store.close()


# =========================================================
# HEADER
# =========================================================

st.title(
    "GPU-Accelerated Video Analytics & Event Detection"
)

st.caption(
    "Object tracking, controlled-zone events, "
    "motion analysis, persistence, and human-in-the-loop review."
)


events_df = load_events()


# =========================================================
# EMPTY DATABASE STATE
# =========================================================

if events_df.empty:

    st.warning(
        "No stored events were found."
    )

    st.info(
        "Run the video analytics pipeline first:\n\n"
        "`python src/track_video.py`"
    )

    st.stop()


# =========================================================
# SUMMARY METRICS
# =========================================================

total_events = len(
    events_df
)

pending_reviews = int(
    (
        events_df[
            "review_status"
        ]
        == "review"
    ).sum()
)

confirmed_events = int(
    (
        events_df[
            "review_status"
        ]
        == "confirmed"
    ).sum()
)

normal_events = int(
    (
        events_df[
            "review_status"
        ]
        == "normal"
    ).sum()
)


metric_col_1, metric_col_2, metric_col_3, metric_col_4 = (
    st.columns(4)
)

metric_col_1.metric(
    "Total Events",
    total_events,
)

metric_col_2.metric(
    "Pending Review",
    pending_reviews,
)

metric_col_3.metric(
    "Confirmed",
    confirmed_events,
)

metric_col_4.metric(
    "Normal",
    normal_events,
)


st.divider()


# =========================================================
# FILTERS
# =========================================================

st.subheader(
    "Event Explorer"
)

filter_col_1, filter_col_2, filter_col_3 = (
    st.columns(3)
)


event_types = sorted(
    events_df[
        "event_type"
    ]
    .dropna()
    .unique()
    .tolist()
)

review_statuses = sorted(
    events_df[
        "review_status"
    ]
    .dropna()
    .unique()
    .tolist()
)

object_classes = sorted(
    events_df[
        "class_name"
    ]
    .dropna()
    .unique()
    .tolist()
)


selected_event_types = (
    filter_col_1.multiselect(
        "Event Type",
        options=event_types,
        default=event_types,
    )
)

selected_review_statuses = (
    filter_col_2.multiselect(
        "Review Status",
        options=review_statuses,
        default=review_statuses,
    )
)

selected_object_classes = (
    filter_col_3.multiselect(
        "Object Class",
        options=object_classes,
        default=object_classes,
    )
)


filtered_df = events_df[
    events_df[
        "event_type"
    ].isin(
        selected_event_types
    )
    & events_df[
        "review_status"
    ].isin(
        selected_review_statuses
    )
    & events_df[
        "class_name"
    ].isin(
        selected_object_classes
    )
].copy()
analytics_df = filtered_df.copy()

# =========================================================
# DISPLAY FORMATTING
# =========================================================

filtered_df[
    "video_time_seconds"
] = (
    filtered_df[
        "video_time_seconds"
    ]
    .round(2)
)

filtered_df[
    "confidence"
] = (
    filtered_df[
        "confidence"
    ]
    .mul(100)
    .round(1)
)
filtered_df[
    "reviewed_at"
] = pd.to_datetime(
    filtered_df[
        "reviewed_at"
    ],
    errors="coerce",
    utc=True,
).dt.strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

filtered_df[
    "reviewed_at"
] = filtered_df[
    "reviewed_at"
].fillna(
    "Pending"
)

filtered_df.rename(
    columns={
        "event_id": "Event ID",
        "video_time_seconds": "Video Time (s)",
        "frame_number": "Frame",
        "track_id": "Track ID",
        "class_name": "Class",
        "confidence": "Confidence (%)",
        "event_type": "Event Type",
        "zone_name": "Zone",
        "duration_seconds": "Duration (s)",
        "review_status": "Review Status",
        "reviewed_at": "Reviewed At",
    },
    inplace=True,
)


display_columns = [
    "Event ID",
    "Video Time (s)",
    "Frame",
    "Track ID",
    "Class",
    "Confidence (%)",
    "Event Type",
    "Zone",
    "Duration (s)",
    "Review Status",
    "Reviewed At",
]

table_df = (
    filtered_df[
        display_columns
    ]
    .reset_index(
        drop=True
    )
)

edited_df = st.data_editor(
    table_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "Event ID",
        "Video Time (s)",
        "Frame",
        "Track ID",
        "Class",
        "Confidence (%)",
        "Event Type",
        "Zone",
        "Duration (s)",
        "Reviewed At",
    ],
    column_config={
        "Review Status": (
            st.column_config.SelectboxColumn(
                "Review Status",
                options=[
                    "review",
                    "normal",
                    "confirmed",
                ],
                required=True,
            )
        )
    },
)

if st.button(
    "Apply Review Changes",
    type="primary",
):

    updates = 0

    for row_index in range(
        len(table_df)
    ):

        old_status = (
            table_df.loc[
                row_index,
                "Review Status",
            ]
        )

        new_status = (
            edited_df.loc[
                row_index,
                "Review Status",
            ]
        )

        if old_status == new_status:
            continue

        event_id = int(
            edited_df.loc[
                row_index,
                "Event ID",
            ]
        )

        if update_event_status(
            event_id=event_id,
            review_status=new_status,
        ):
            updates += 1

    if updates > 0:

        load_events.clear()

        st.success(
            f"{updates} event(s) updated."
        )

        st.rerun()

    else:

        st.info(
            "No review status changes detected."
        )


st.caption(
    f"Showing {len(filtered_df)} "
    f"of {total_events} stored events."
)
csv_data = (
    filtered_df.to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)

st.download_button(
    label="Export Filtered Events as CSV",
    data=csv_data,
    file_name="filtered_events.csv",
    mime="text/csv",
)
# =========================================================
# EVENT EVIDENCE VIEWER
# =========================================================

st.subheader("Event Evidence Viewer")

if table_df.empty:
    st.info("No events match the current filters.")

else:
    selected_event_id = st.selectbox(
        "Select an Event",
        options=table_df["Event ID"].tolist(),
        format_func=lambda event_id: f"Event #{event_id}",
    )

    selected_event = events_df.loc[
        events_df["event_id"] == selected_event_id
    ].iloc[0]

    evidence_path = selected_event["evidence_path"]

    if pd.isna(evidence_path) or not str(evidence_path).strip():
        st.info("No visual evidence is available for this event.")

    else:
        snapshot_path = PROJECT_ROOT / evidence_path

        if snapshot_path.is_file():
            st.image(
                str(snapshot_path),
                caption=(
                    f"Event #{selected_event_id} — "
                    f"{selected_event['event_type']}"
                ),
                use_container_width=True,
            )

        else:
            st.warning("The evidence image could not be found.")
st.divider()


# =========================================================
# ANALYTICS
# =========================================================

st.subheader(
    "Event Analytics"
)

chart_col_1, chart_col_2 = (
    st.columns(2)
)


event_counts = (
    analytics_df[
        "event_type"
    ]
    .value_counts()
    .rename_axis(
        "Event Type"
    )
    .reset_index(
        name="Count"
    )
)


event_chart = (
    px.bar(
        event_counts,
        x="Event Type",
        y="Count",
        title="Events by Type",
        text_auto=True,
    )
)

event_chart.update_layout(
    showlegend=False,
)

chart_col_1.plotly_chart(
    event_chart,
    use_container_width=True,
)


review_counts = (
    analytics_df[
        "review_status"
    ]
    .value_counts()
    .rename_axis(
        "Review Status"
    )
    .reset_index(
        name="Count"
    )
)


review_chart = (
    px.pie(
        review_counts,
        names="Review Status",
        values="Count",
        title="Human Review Status",
        hole=0.45,
    )
)

chart_col_2.plotly_chart(
    review_chart,
    use_container_width=True,
)
timeline_df = (
    analytics_df[
        [
            "video_time_seconds",
            "event_type",
            "track_id",
            "class_name",
            "review_status",
        ]
    ]
    .copy()
)

timeline_chart = px.scatter(
    timeline_df,
    x="video_time_seconds",
    y="event_type",
    color="review_status",
    symbol="class_name",
    hover_data={
        "track_id": True,
        "class_name": True,
        "review_status": True,
        "video_time_seconds": ":.2f",
    },
    title="Event Timeline",
    labels={
        "video_time_seconds": (
            "Video Time (seconds)"
        ),
        "event_type": (
            "Event Type"
        ),
        "review_status": (
            "Review Status"
        ),
        "class_name": (
            "Object Class"
        ),
    },
)

timeline_chart.update_layout(
    height=420,
)

st.plotly_chart(
    timeline_chart,
    use_container_width=True,
)

st.divider()


# =========================================================
# VIDEO PREVIEW
# =========================================================

st.subheader(
    "Processed Video"
)

if (
    OUTPUT_VIDEO.exists()
    and OUTPUT_VIDEO.stat().st_size > 0
):

    st.video(
        str(
            OUTPUT_VIDEO
        )
    )

else:

    st.info(
        "Processed video not found. "
        "Run the tracking pipeline to generate it."
    )


# =========================================================
# SYSTEM NOTES
# =========================================================

with st.expander(
    "System Notes"
):

    st.markdown(
        """
        - Object detection is performed with YOLO.
        - Persistent object IDs are maintained with ByteTrack.
        - Zone membership uses the bottom-center of each bounding box.
        - Zone transitions use temporal debounce to reduce boundary jitter.
        - Loitering is based on observed video time.
        - Motion direction is estimated in image space relative to the zone center.
        - Event classification is limited to observable behavior.
        - Human review determines whether an event is normal or confirmed.
        """
    )