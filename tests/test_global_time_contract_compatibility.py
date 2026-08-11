from __future__ import annotations

import pandas as pd

from cryptolab.time_contract import (
    point_in_time_filter,
)


def _make_frame(
    event_column: str,
    event_times: list[str],
    available_times: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            event_column: pd.to_datetime(
                event_times,
                utc=True,
            ),
            "available_at": pd.to_datetime(
                available_times,
                utc=True,
            ),
            "available_at_quality": [
                "derived",
                "exact",
                "exact",
            ],
            "ingested_at": pd.to_datetime(
                [
                    "2026-08-11T10:00:10Z",
                    "2026-08-11T10:05:10Z",
                    "2026-08-11T10:10:10Z",
                ],
                utc=True,
            ),
            "ingested_at_quality": [
                "exact",
                "exact",
                "exact",
            ],
        }
    )


def test_global_as_of_rule_is_available_at_based():
    df = _make_frame(
        event_column="event_time",
        event_times=[
            "2026-08-11T10:00:00Z",
            "2026-08-11T10:05:00Z",
            "2026-08-11T10:10:00Z",
        ],
        available_times=[
            "2026-08-11T10:00:00Z",
            "2026-08-11T10:05:05Z",
            "2026-08-11T10:10:05Z",
        ],
    )

    result = point_in_time_filter(
        df,
        as_of_time=pd.Timestamp(
            "2026-08-11T10:05:03Z"
        ),
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "event_time"
        ]
        == pd.Timestamp(
            "2026-08-11T10:00:00Z"
        )
    )


def test_event_time_alone_does_not_make_record_usable():
    df = pd.DataFrame(
        {
            "event_time": [
                pd.Timestamp(
                    "2026-08-11T10:00:00Z"
                )
            ],
            "available_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:05Z"
                )
            ],
            "available_at_quality": [
                "exact"
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:06Z"
                )
            ],
            "ingested_at_quality": [
                "exact"
            ],
        }
    )

    result = point_in_time_filter(
        df,
        as_of_time=pd.Timestamp(
            "2026-08-11T10:00:02Z"
        ),
    )

    assert result.empty


def test_record_becomes_usable_at_available_at():
    df = pd.DataFrame(
        {
            "event_time": [
                pd.Timestamp(
                    "2026-08-11T10:00:00Z"
                )
            ],
            "available_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:05Z"
                )
            ],
            "available_at_quality": [
                "exact"
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:06Z"
                )
            ],
            "ingested_at_quality": [
                "exact"
            ],
        }
    )

    result = point_in_time_filter(
        df,
        as_of_time=pd.Timestamp(
            "2026-08-11T10:00:05Z"
        ),
    )

    assert len(result) == 1


def test_ingested_at_after_as_of_does_not_change_causal_rule():
    """
    available_at is the canonical research availability field.

    ingested_at is lineage evidence and does not redefine
    the external information-availability boundary.
    """

    df = pd.DataFrame(
        {
            "event_time": [
                pd.Timestamp(
                    "2026-08-11T10:00:00Z"
                )
            ],
            "available_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:05Z"
                )
            ],
            "available_at_quality": [
                "exact"
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T10:30:00Z"
                )
            ],
            "ingested_at_quality": [
                "exact"
            ],
        }
    )

    result = point_in_time_filter(
        df,
        as_of_time=pd.Timestamp(
            "2026-08-11T10:10:00Z"
        ),
    )

    assert len(result) == 1


def test_derived_availability_is_still_filterable():
    df = pd.DataFrame(
        {
            "event_time": [
                pd.Timestamp(
                    "2026-08-11T10:00:00Z"
                )
            ],
            "available_at": [
                pd.Timestamp(
                    "2026-08-11T10:00:00Z"
                )
            ],
            "available_at_quality": [
                "derived"
            ],
            "ingested_at": [
                pd.NaT
            ],
            "ingested_at_quality": [
                "unknown"
            ],
        }
    )

    result = point_in_time_filter(
        df,
        as_of_time=pd.Timestamp(
            "2026-08-11T10:00:00Z"
        ),
    )

    assert len(result) == 1
