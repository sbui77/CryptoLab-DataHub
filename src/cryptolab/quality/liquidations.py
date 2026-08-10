from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidationAuditResult:
    exchange: str
    symbol: str

    rows: int

    first_event_time: pd.Timestamp | None
    last_event_time: pd.Timestamp | None

    duplicate_count: int

    invalid_side_count: int
    invalid_liquidation_side_count: int

    invalid_quantity_count: int
    invalid_price_count: int
    invalid_notional_count: int

    side_mapping_mismatch_count: int
    notional_identity_mismatch_count: int

    timestamp_disorder_count: int

    structurally_valid: bool
    valid: bool


def audit_liquidations(
    df: pd.DataFrame,
) -> LiquidationAuditResult:
    """
    Audit canonical liquidation event data.

    Continuity is intentionally NOT assessed because
    forceOrder is an event stream, not a fixed-interval series.
    """

    if df.empty:
        return LiquidationAuditResult(
            exchange="unknown",
            symbol="unknown",
            rows=0,
            first_event_time=None,
            last_event_time=None,
            duplicate_count=0,
            invalid_side_count=0,
            invalid_liquidation_side_count=0,
            invalid_quantity_count=0,
            invalid_price_count=0,
            invalid_notional_count=0,
            side_mapping_mismatch_count=0,
            notional_identity_mismatch_count=0,
            timestamp_disorder_count=0,
            structurally_valid=False,
            valid=False,
        )

    work = (
        df.copy()
        .sort_values(
            [
                "event_time",
                "order_time",
            ]
        )
        .reset_index(drop=True)
    )

    exchange = str(
        work["exchange"].iloc[0]
    )

    symbol = str(
        work["symbol"].iloc[0]
    )

    dedup_columns = [
        "event_time",
        "order_time",
        "side",
        "price",
        "original_quantity",
        "filled_quantity",
    ]

    duplicate_count = int(
        work.duplicated(
            subset=dedup_columns
        ).sum()
    )

    invalid_side_count = int(
        (
            ~work["side"]
            .isin(
                [
                    "BUY",
                    "SELL",
                ]
            )
        ).sum()
    )

    invalid_liquidation_side_count = int(
        (
            ~work[
                "liquidation_side"
            ]
            .isin(
                [
                    "long",
                    "short",
                ]
            )
        ).sum()
    )

    invalid_quantity_count = int(
        (
            work[
                "original_quantity"
            ].isna()
            | (
                work[
                    "original_quantity"
                ]
                <= 0
            )
            | work[
                "filled_quantity"
            ].isna()
            | (
                work[
                    "filled_quantity"
                ]
                < 0
            )
        ).sum()
    )

    invalid_price_count = int(
        (
            work["price"].isna()
            | (
                work["price"]
                < 0
            )
            | work[
                "average_price"
            ].isna()
            | (
                work[
                    "average_price"
                ]
                < 0
            )
        ).sum()
    )

    invalid_notional_count = int(
        (
            work[
                "liquidation_notional"
            ].isna()
            | (
                work[
                    "liquidation_notional"
                ]
                < 0
            )
        ).sum()
    )

    expected_liquidation_side = np.where(
        work["side"].eq("SELL"),
        "long",
        "short",
    )

    side_mapping_mismatch_count = int(
        (
            work[
                "liquidation_side"
            ]
            .astype(str)
            .to_numpy()
            != expected_liquidation_side
        ).sum()
    )

    effective_price = np.where(
        work["average_price"] > 0,
        work["average_price"],
        work["price"],
    )

    expected_notional = (
        work["filled_quantity"]
        * effective_price
    )

    notional_valid = np.isclose(
        work[
            "liquidation_notional"
        ],
        expected_notional,
        rtol=1e-12,
        atol=1e-8,
        equal_nan=False,
    )

    notional_identity_mismatch_count = int(
        (
            ~notional_valid
        ).sum()
    )

    time_diff = (
        work["event_time"]
        .diff()
    )

    timestamp_disorder_count = int(
        (
            time_diff
            < pd.Timedelta(0)
        ).sum()
    )

    structurally_valid = (
        duplicate_count == 0
        and invalid_side_count == 0
        and invalid_liquidation_side_count == 0
        and invalid_quantity_count == 0
        and invalid_price_count == 0
        and invalid_notional_count == 0
        and side_mapping_mismatch_count == 0
        and notional_identity_mismatch_count == 0
        and timestamp_disorder_count == 0
    )

    return LiquidationAuditResult(
        exchange=exchange,
        symbol=symbol,
        rows=len(work),
        first_event_time=pd.Timestamp(
            work["event_time"].iloc[0]
        ),
        last_event_time=pd.Timestamp(
            work["event_time"].iloc[-1]
        ),
        duplicate_count=duplicate_count,
        invalid_side_count=(
            invalid_side_count
        ),
        invalid_liquidation_side_count=(
            invalid_liquidation_side_count
        ),
        invalid_quantity_count=(
            invalid_quantity_count
        ),
        invalid_price_count=(
            invalid_price_count
        ),
        invalid_notional_count=(
            invalid_notional_count
        ),
        side_mapping_mismatch_count=(
            side_mapping_mismatch_count
        ),
        notional_identity_mismatch_count=(
            notional_identity_mismatch_count
        ),
        timestamp_disorder_count=(
            timestamp_disorder_count
        ),
        structurally_valid=(
            structurally_valid
        ),
        valid=structurally_valid,
    )
