from __future__ import annotations

from datetime import timedelta

import pandas as pd


_TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,

    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,

    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,

    "1w": 7 * 24 * 60 * 60,
}


def timeframe_seconds(
    timeframe: str,
) -> int:
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return _TIMEFRAME_SECONDS[timeframe]


def timeframe_milliseconds(
    timeframe: str,
) -> int:
    return timeframe_seconds(timeframe) * 1000


def timeframe_timedelta(
    timeframe: str,
) -> timedelta:
    return timedelta(
        seconds=timeframe_seconds(timeframe)
    )


def pandas_frequency(
    timeframe: str,
) -> str:
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",

        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",

        "1d": "1D",
        "3d": "3D",

        "1w": "7D",
    }

    if timeframe not in mapping:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return mapping[timeframe]


def datetime_to_ms(
    value: pd.Timestamp,
) -> int:
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")

    return int(value.timestamp() * 1000)
