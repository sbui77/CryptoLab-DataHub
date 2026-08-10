from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TakerFlowAuditResult:
    exchange: str
    symbol: str
    period: str

    rows: int

    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    duplicate_count: int
    gap_count: int

    invalid_buy_volume_count: int
    invalid_sell_volume_count: int
    invalid_ratio_count: int

    ratio_identity_mismatch_count: int

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


RATIO_ABS_TOLERANCE = 5e-4


def audit_taker_flow(
    df: pd.DataFrame,
) -> TakerFlowAuditResult:
    """
    Audit canonical Binance futures taker-flow data.
    """

    if df.empty:
        return TakerFlowAuditResult(
            exchange="unknown",
            symbol="unknown",
            period="unknown",
            rows=0,
            first_timestamp=None,
            last_timestamp=None,
            duplicate_count=0,
            gap_count=0,
            invalid_buy_volume_count=0,
            invalid_sell_volume_count=0,
            invalid_ratio_count=0,
            ratio_identity_mismatch_count=0,
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

    if period not in PERIOD_TO_DELTA:
        raise ValueError(
            f"Unsupported taker-flow period: {period}"
        )

    duplicate_count = int(
        work["timestamp"]
        .duplicated()
        .sum()
    )

    invalid_buy_volume_count = int(
        (
            work["buy_volume"].isna()
            | (
                work["buy_volume"]
                < 0
            )
        ).sum()
    )

    invalid_sell_volume_count = int(
        (
            work["sell_volume"].isna()
            | (
                work["sell_volume"]
                < 0
            )
        ).sum()
    )

    invalid_ratio_count = int(
        (
            work["buy_sell_ratio"].isna()
            | ~np.isfinite(
                work["buy_sell_ratio"]
            )
            | (
                work["buy_sell_ratio"]
                < 0
            )
        ).sum()
    )

    expected_ratio = np.where(
        work["sell_volume"] > 0,
        work["buy_volume"]
        / work["sell_volume"],
        np.nan,
    )

    ratio_valid = (
        (
            work["sell_volume"] == 0
        )
        |
        np.isclose(
            work["buy_sell_ratio"],
            expected_ratio,
            rtol=1e-4,
            atol=RATIO_ABS_TOLERANCE,
            equal_nan=False,
        )
    )

    ratio_identity_mismatch_count = int(
        (
            ~ratio_valid
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
        and invalid_buy_volume_count == 0
        and invalid_sell_volume_count == 0
        and invalid_ratio_count == 0
        and ratio_identity_mismatch_count == 0
        and timestamp_disorder_count == 0
    )

    continuous = (
        gap_count == 0
    )

    return TakerFlowAuditResult(
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
        invalid_buy_volume_count=(
            invalid_buy_volume_count
        ),
        invalid_sell_volume_count=(
            invalid_sell_volume_count
        ),
        invalid_ratio_count=(
            invalid_ratio_count
        ),
        ratio_identity_mismatch_count=(
            ratio_identity_mismatch_count
        ),
        timestamp_disorder_count=(
            timestamp_disorder_count
        ),
        structurally_valid=(
            structurally_valid
        ),
        continuous=continuous,
        valid=structurally_valid,
    )
