from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OpenInterestAuditResult:
    exchange: str
    symbol: str
    period: str

    rows: int

    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    duplicate_count: int
    gap_count: int
    invalid_base_count: int
    invalid_quote_count: int
    timestamp_disorder_count: int

    structurally_valid: bool
    continuous: bool
    valid: bool


PERIOD_TO_DELTA = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


def audit_open_interest(
    df: pd.DataFrame,
) -> OpenInterestAuditResult:
    """
    Audit canonical historical Open Interest data.
    """

    if df.empty:
        return OpenInterestAuditResult(
            exchange="unknown",
            symbol="unknown",
            period="unknown",
            rows=0,
            first_timestamp=None,
            last_timestamp=None,
            duplicate_count=0,
            gap_count=0,
            invalid_base_count=0,
            invalid_quote_count=0,
            timestamp_disorder_count=0,
            structurally_valid=False,
            continuous=False,
            valid=False,
        )

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    exchange = str(
        work["exchange"].iloc[0]
    )

    symbol = str(
        work["symbol"].iloc[0]
    )

    period = str(
        work["period"].iloc[0]
    )

    duplicate_count = int(
        work["timestamp"]
        .duplicated()
        .sum()
    )

    invalid_base_count = int(
        (
            work["open_interest_base"].isna()
            | (
                work["open_interest_base"]
                < 0
            )
        ).sum()
    )

    invalid_quote_count = int(
        (
            work["open_interest_quote"].isna()
            | (
                work["open_interest_quote"]
                < 0
            )
        ).sum()
    )

    diff = (
        work["timestamp"]
        .diff()
    )

    timestamp_disorder_count = int(
        (
            diff
            < pd.Timedelta(0)
        ).sum()
    )

    expected_delta = (
        PERIOD_TO_DELTA[
            period
        ]
    )

    gap_count = int(
        (
            diff
            > expected_delta
        ).sum()
    )

    structurally_valid = (
        duplicate_count == 0
        and invalid_base_count == 0
        and invalid_quote_count == 0
        and timestamp_disorder_count == 0
    )

    continuous = (
        gap_count == 0
    )

    return OpenInterestAuditResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        rows=len(work),
        first_timestamp=pd.Timestamp(
            work["timestamp"].iloc[0]
        ),
        last_timestamp=pd.Timestamp(
            work["timestamp"].iloc[-1]
        ),
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        invalid_base_count=invalid_base_count,
        invalid_quote_count=invalid_quote_count,
        timestamp_disorder_count=timestamp_disorder_count,
        structurally_valid=structurally_valid,
        continuous=continuous,
        valid=structurally_valid,
    )
