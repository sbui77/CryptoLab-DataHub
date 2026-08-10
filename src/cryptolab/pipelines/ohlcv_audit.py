from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cryptolab.pipelines.ohlcv_storage import (
    read_ohlcv,
)
from cryptolab.quality.ohlcv import (
    find_ohlcv_gaps,
    validate_ohlcv,
)


@dataclass
class OHLCVAuditResult:
    exchange: str
    symbol: str
    timeframe: str

    rows: int

    first_open_time: pd.Timestamp | None
    last_open_time: pd.Timestamp | None

    duplicate_count: int

    gap_count: int
    gap_group_count: int

    structurally_valid: bool
    continuous: bool
    valid: bool


def _count_gap_groups(
    gaps: pd.DataFrame,
    timeframe: str,
) -> int:
    if gaps.empty:
        return 0

    timeframe_map = {
        "1m": pd.Timedelta(minutes=1),
        "3m": pd.Timedelta(minutes=3),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
        "3d": pd.Timedelta(days=3),
        "1w": pd.Timedelta(days=7),
    }

    if timeframe not in timeframe_map:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    delta = timeframe_map[timeframe]

    times = (
        gaps["missing_open_time"]
        .sort_values()
        .tolist()
    )

    groups = 1

    previous = times[0]

    for current in times[1:]:
        if current - previous != delta:
            groups += 1

        previous = current

    return groups


def audit_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> OHLCVAuditResult:

    df = read_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    if df.empty:
        return OHLCVAuditResult(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            rows=0,
            first_open_time=None,
            last_open_time=None,
            duplicate_count=0,
            gap_count=0,
            gap_group_count=0,
            structurally_valid=False,
            continuous=False,
            valid=False,
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "exchange",
                "symbol",
                "timeframe",
                "open_time",
            ]
        ).sum()
    )

    gaps = find_ohlcv_gaps(
        df,
        timeframe,
    )

    gap_count = len(gaps)

    gap_group_count = _count_gap_groups(
        gaps,
        timeframe,
    )

    try:
        validate_ohlcv(df)
        structurally_valid = True

    except Exception:
        structurally_valid = False

    continuous = (
        gap_count == 0
    )

    valid = (
        structurally_valid
        and duplicate_count == 0
    )

    return OHLCVAuditResult(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        rows=len(df),
        first_open_time=df["open_time"].min(),
        last_open_time=df["open_time"].max(),
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        gap_group_count=gap_group_count,
        structurally_valid=structurally_valid,
        continuous=continuous,
        valid=valid,
    )