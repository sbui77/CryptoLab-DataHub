from __future__ import annotations

import pandas as pd

from cryptolab.pipelines.liquidation_heartbeat import (
    build_hourly_liquidation_coverage,
    normalize_liquidation_heartbeat,
)


def make_heartbeat(
    count: int,
    connected: bool = True,
    freq: str = "30s",
) -> pd.DataFrame:
    times = pd.date_range(
        "2026-08-10T10:00:00Z",
        periods=count,
        freq=freq,
    )

    return pd.DataFrame(
        {
            "exchange": [
                "binance"
            ] * count,

            "symbol": [
                "BTCUSDT"
            ] * count,

            "heartbeat_time": times,

            "connected": [
                connected
            ] * count,

            "messages_received": list(
                range(count)
            ),

            "liquidation_rows": [
                0
            ] * count,

            "reconnect_count": [
                0
            ] * count,
        }
    )


def test_normalize_heartbeat():
    df = normalize_liquidation_heartbeat(
        make_heartbeat(
            3
        )
    )

    assert len(df) == 3

    assert (
        df[
            "heartbeat_time"
        ].dt.tz
        is not None
    )


def test_full_hour_covered():
    df = make_heartbeat(
        120
    )

    result = (
        build_hourly_liquidation_coverage(
            df,
            minimum_coverage_ratio=0.80,
            heartbeat_interval_seconds=30,
        )
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "collector_coverage_ratio",
        ]
        == 1.0
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_collector_covered",
            ]
        )
        is True
    )


def test_80_percent_is_covered():
    df = make_heartbeat(
        96
    )

    result = (
        build_hourly_liquidation_coverage(
            df,
            minimum_coverage_ratio=0.80,
            heartbeat_interval_seconds=30,
        )
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_collector_covered",
            ]
        )
        is True
    )


def test_below_threshold_not_covered():
    df = make_heartbeat(
        95
    )

    result = (
        build_hourly_liquidation_coverage(
            df,
            minimum_coverage_ratio=0.80,
            heartbeat_interval_seconds=30,
        )
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_collector_covered",
            ]
        )
        is False
    )


def test_disconnected_heartbeats_not_counted():
    connected = make_heartbeat(
        60,
        connected=True,
    )

    disconnected = make_heartbeat(
        60,
        connected=False,
    )

    df = pd.concat(
        [
            connected,
            disconnected,
        ],
        ignore_index=True,
    )

    df[
        "heartbeat_time"
    ] = pd.date_range(
        "2026-08-10T10:00:00Z",
        periods=120,
        freq="30s",
    )

    result = (
        build_hourly_liquidation_coverage(
            df,
            minimum_coverage_ratio=0.80,
            heartbeat_interval_seconds=30,
        )
    )

    assert (
        result.loc[
            0,
            "connected_heartbeat_count",
        ]
        == 60
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_collector_covered",
            ]
        )
        is False
    )


def test_as_of_contract():
    df = make_heartbeat(
        120
    )

    result = (
        build_hourly_liquidation_coverage(
            df
        )
    )

    assert (
        result.loc[
            0,
            "as_of_time",
        ]
        == (
            result.loc[
                0,
                "open_time",
            ]
            + pd.Timedelta(
                hours=1
            )
        )
    )
