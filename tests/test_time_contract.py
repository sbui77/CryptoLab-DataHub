from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.time_contract import (
    TimeContract,
    TimeContractError,
    TimeQuality,
    derive_historical_time_metadata,
    get_historical_time_policy,
    latest_point_in_time,
    normalize_time_columns,
    point_in_time_filter,
    validate_time_contract,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
            ],
            "available_at": [
                "2026-01-01T00:05:00Z",
                "2026-01-02T12:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
            "ingested_at": [
                "2026-01-01T00:06:00Z",
                "2026-01-02T12:01:00Z",
                "2026-01-04T00:01:00Z",
            ],
            "value": [
                10,
                20,
                30,
            ],
        }
    )


def test_normalize_time_columns():
    contract = TimeContract(
        event_time_column="event_time",
    )

    df = normalize_time_columns(
        make_df(),
        contract,
    )

    assert (
        str(df["event_time"].dt.tz)
        == "UTC"
    )

    assert (
        str(df["available_at"].dt.tz)
        == "UTC"
    )


def test_point_in_time_filter_blocks_future_availability():
    df = point_in_time_filter(
        make_df(),
        as_of_time="2026-01-02T06:00:00Z",
    )

    assert len(df) == 1
    assert df.iloc[0]["value"] == 10


def test_point_in_time_filter_includes_available_record():
    df = point_in_time_filter(
        make_df(),
        as_of_time="2026-01-02T12:00:00Z",
    )

    assert len(df) == 2

    assert df["value"].tolist() == [
        10,
        20,
    ]


def test_latest_point_in_time():
    row = latest_point_in_time(
        make_df(),
        as_of_time="2026-01-02T13:00:00Z",
        event_time_column="event_time",
    )

    assert row is not None
    assert row["value"] == 20


def test_ingested_before_available_rejected():
    df = make_df()

    df.loc[
        1,
        "ingested_at",
    ] = "2026-01-02T11:00:00Z"

    contract = TimeContract(
        event_time_column="event_time",
    )

    with pytest.raises(
        TimeContractError
    ):
        validate_time_contract(
            df,
            contract,
        )


def test_missing_available_at_rejected():
    df = make_df().drop(
        columns=[
            "available_at",
        ]
    )

    contract = TimeContract(
        event_time_column="event_time",
    )

    with pytest.raises(
        TimeContractError
    ):
        normalize_time_columns(
            df,
            contract,
        )


def test_event_time_may_be_after_available_at():
    df = pd.DataFrame(
        {
            "event_time": [
                "2026-01-02T00:00:00Z",
            ],
            "available_at": [
                "2026-01-01T12:00:00Z",
            ],
            "ingested_at": [
                "2026-01-01T12:01:00Z",
            ],
        }
    )

    contract = TimeContract(
        event_time_column="event_time",
    )

    validate_time_contract(
        df,
        contract,
    )


@pytest.mark.parametrize(
    (
        "dataset",
        "event_column",
    ),
    [
        (
            "aggtrades",
            "trade_time",
        ),
        (
            "open_interest",
            "timestamp",
        ),
        (
            "funding",
            "funding_time",
        ),
        (
            "basis",
            "timestamp",
        ),
        (
            "taker_flow",
            "timestamp",
        ),
        (
            "liquidations",
            "event_time",
        ),
    ],
)
def test_historical_policy_registry(
    dataset,
    event_column,
):
    policy = get_historical_time_policy(
        dataset
    )

    assert (
        policy.event_time_column
        == event_column
    )

    assert (
        policy.availability_quality
        == TimeQuality.DERIVED
    )

    assert (
        policy.ingestion_quality
        == TimeQuality.UNKNOWN
    )


def test_unknown_historical_policy_rejected():
    with pytest.raises(
        TimeContractError
    ):
        get_historical_time_policy(
            "does_not_exist"
        )


def test_historical_metadata_is_explicitly_derived():
    source = pd.DataFrame(
        {
            "trade_time": [
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:02Z",
            ],
            "price": [
                100.0,
                101.0,
            ],
        }
    )

    result = derive_historical_time_metadata(
        source,
        dataset="aggtrades",
    )

    expected = pd.to_datetime(
        source["trade_time"],
        utc=True,
    )

    pd.testing.assert_series_equal(
        result["available_at"],
        expected,
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
        .isna()
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("unknown")
        .all()
    )


def test_historical_migration_does_not_fake_ingestion_time():
    source = pd.DataFrame(
        {
            "event_time": [
                "2026-01-01T00:00:00Z",
            ],
            "liquidation_notional": [
                1000.0,
            ],
        }
    )

    result = derive_historical_time_metadata(
        source,
        dataset="liquidations",
    )

    assert pd.isna(
        result.iloc[0][
            "ingested_at"
        ]
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )


def test_derived_historical_data_respects_as_of():
    source = pd.DataFrame(
        {
            "funding_time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T08:00:00Z",
                "2026-01-01T16:00:00Z",
            ],
            "funding_rate": [
                0.0001,
                0.0002,
                0.0003,
            ],
        }
    )

    migrated = derive_historical_time_metadata(
        source,
        dataset="funding",
    )

    filtered = point_in_time_filter(
        migrated,
        as_of_time="2026-01-01T12:00:00Z",
    )

    assert len(filtered) == 2

    assert filtered[
        "funding_rate"
    ].tolist() == [
        0.0001,
        0.0002,
    ]
