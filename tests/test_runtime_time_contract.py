from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.time_contract import (
    TimeContractError,
    attach_runtime_time_metadata,
)


FIXED_INGESTION_TIME = pd.Timestamp(
    "2026-08-11T00:00:10Z"
)


def fixed_clock() -> pd.Timestamp:
    return FIXED_INGESTION_TIME


def test_rest_runtime_metadata():
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-11T00:00:00Z",
                "2026-08-11T00:00:05Z",
            ],
            "value": [
                1.0,
                2.0,
            ],
        }
    )

    result = attach_runtime_time_metadata(
        source,
        event_time_column="timestamp",
        clock=fixed_clock,
    )

    expected_available = pd.to_datetime(
        source["timestamp"],
        utc=True,
    )

    pd.testing.assert_series_equal(
        result["available_at"],
        expected_available,
        check_names=False,
    )

    assert (
        result[
            "available_at_quality"
        ]
        .eq("derived")
        .all()
    )

    assert (
        result[
            "ingested_at"
        ]
        .eq(FIXED_INGESTION_TIME)
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("exact")
        .all()
    )


def test_runtime_batch_has_one_ingestion_time():
    source = pd.DataFrame(
        {
            "trade_time": pd.to_datetime(
                [
                    "2026-08-11T00:00:01Z",
                    "2026-08-11T00:00:02Z",
                    "2026-08-11T00:00:03Z",
                ],
                utc=True,
            ),
        }
    )

    result = attach_runtime_time_metadata(
        source,
        event_time_column="trade_time",
        clock=fixed_clock,
    )

    assert (
        result[
            "ingested_at"
        ].nunique()
        == 1
    )


def test_websocket_available_at_can_be_exact():
    source = pd.DataFrame(
        {
            "event_time": [
                "2026-08-11T00:00:00.100Z",
            ],
            "value": [
                100.0,
            ],
        }
    )

    receive_time = pd.Timestamp(
        "2026-08-11T00:00:00.250Z"
    )

    persist_time = pd.Timestamp(
        "2026-08-11T00:00:00.300Z"
    )

    result = attach_runtime_time_metadata(
        source,
        event_time_column="event_time",
        available_at=receive_time,
        ingested_at=persist_time,
    )

    assert (
        result.iloc[0][
            "available_at"
        ]
        == receive_time
    )

    assert (
        result.iloc[0][
            "available_at_quality"
        ]
        == "exact"
    )

    assert (
        result.iloc[0][
            "ingested_at"
        ]
        == persist_time
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )


def test_runtime_metadata_does_not_modify_event_time():
    original_time = pd.Timestamp(
        "2026-08-11T01:00:00Z"
    )

    source = pd.DataFrame(
        {
            "funding_time": [
                original_time,
            ],
        }
    )

    result = attach_runtime_time_metadata(
        source,
        event_time_column="funding_time",
        clock=fixed_clock,
    )

    assert (
        result.iloc[0][
            "funding_time"
        ]
        == original_time
    )


def test_missing_event_time_column_rejected():
    source = pd.DataFrame(
        {
            "value": [
                1.0,
            ],
        }
    )

    with pytest.raises(
        TimeContractError
    ):
        attach_runtime_time_metadata(
            source,
            event_time_column="timestamp",
            clock=fixed_clock,
        )


def test_invalid_event_timestamp_rejected():
    source = pd.DataFrame(
        {
            "timestamp": [
                "not-a-timestamp",
            ],
        }
    )

    with pytest.raises(
        TimeContractError
    ):
        attach_runtime_time_metadata(
            source,
            event_time_column="timestamp",
            clock=fixed_clock,
        )


def test_runtime_available_at_can_be_unknown():
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-11T00:00:00Z",
            ],
        }
    )

    result = attach_runtime_time_metadata(
        source,
        event_time_column="timestamp",
        available_at_from_event=False,
        clock=fixed_clock,
    )

    assert pd.isna(
        result.iloc[0][
            "available_at"
        ]
    )

    assert (
        result.iloc[0][
            "available_at_quality"
        ]
        == "unknown"
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )
