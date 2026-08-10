from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundingRateAuditResult:
    exchange: str
    symbol: str

    rows: int

    first_funding_time: pd.Timestamp | None
    last_funding_time: pd.Timestamp | None

    duplicate_count: int
    invalid_rate_count: int
    invalid_mark_price_count: int
    timestamp_disorder_count: int

    min_interval_hours: float | None
    median_interval_hours: float | None
    max_interval_hours: float | None

    regular_count: int
    special_count: int
    unknown_rate_type_count: int

    structurally_valid: bool
    valid: bool


def audit_funding_rate(
    df: pd.DataFrame,
) -> FundingRateAuditResult:
    """
    Audit canonical funding-rate history.

    Funding intervals are reported but not forced to 8h,
    because exchange-level funding interval adjustments
    may occur.
    """

    if df.empty:
        return FundingRateAuditResult(
            exchange="unknown",
            symbol="unknown",
            rows=0,
            first_funding_time=None,
            last_funding_time=None,
            duplicate_count=0,
            invalid_rate_count=0,
            invalid_mark_price_count=0,
            timestamp_disorder_count=0,
            min_interval_hours=None,
            median_interval_hours=None,
            max_interval_hours=None,
            regular_count=0,
            special_count=0,
            unknown_rate_type_count=0,
            structurally_valid=False,
            valid=False,
        )

    work = (
        df.copy()
        .sort_values("funding_time")
        .reset_index(drop=True)
    )

    exchange = str(
        work["exchange"].iloc[0]
    )

    symbol = str(
        work["symbol"].iloc[0]
    )

    duplicate_count = int(
        work[
            "funding_time"
        ]
        .duplicated()
        .sum()
    )

    invalid_rate_count = int(
        work[
            "funding_rate"
        ]
        .isna()
        .sum()
    )

    invalid_mark_price_count = int(
        (
            work["mark_price"].notna()
            & (
                work["mark_price"]
                <= 0
            )
        ).sum()
    )

    interval = (
        work[
            "funding_time"
        ]
        .diff()
    )

    timestamp_disorder_count = int(
        (
            interval
            < pd.Timedelta(0)
        ).sum()
    )

    interval_hours = (
        interval
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )

    if interval_hours.empty:
        min_interval_hours = None
        median_interval_hours = None
        max_interval_hours = None
    else:
        min_interval_hours = float(
            interval_hours.min()
        )

        median_interval_hours = float(
            interval_hours.median()
        )

        max_interval_hours = float(
            interval_hours.max()
        )

    rate_type = (
        work["rate_type"]
        .astype("string")
    )

    regular_count = int(
        rate_type.eq(
            "Regular"
        ).sum()
    )

    special_count = int(
        rate_type.eq(
            "Special"
        ).sum()
    )

    unknown_rate_type_count = int(
        (
            ~rate_type.isin(
                [
                    "Regular",
                    "Special",
                ]
            )
        ).sum()
    )

    structurally_valid = (
        duplicate_count == 0
        and invalid_rate_count == 0
        and invalid_mark_price_count == 0
        and timestamp_disorder_count == 0
        and unknown_rate_type_count == 0
    )

    return FundingRateAuditResult(
        exchange=exchange,
        symbol=symbol,
        rows=len(work),
        first_funding_time=pd.Timestamp(
            work[
                "funding_time"
            ].iloc[0]
        ),
        last_funding_time=pd.Timestamp(
            work[
                "funding_time"
            ].iloc[-1]
        ),
        duplicate_count=duplicate_count,
        invalid_rate_count=invalid_rate_count,
        invalid_mark_price_count=(
            invalid_mark_price_count
        ),
        timestamp_disorder_count=(
            timestamp_disorder_count
        ),
        min_interval_hours=(
            min_interval_hours
        ),
        median_interval_hours=(
            median_interval_hours
        ),
        max_interval_hours=(
            max_interval_hours
        ),
        regular_count=regular_count,
        special_count=special_count,
        unknown_rate_type_count=(
            unknown_rate_type_count
        ),
        structurally_valid=(
            structurally_valid
        ),
        valid=structurally_valid,
    )
