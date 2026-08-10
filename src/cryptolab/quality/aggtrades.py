from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class AggTradesQualityError(ValueError):
    """Raised when aggTrades quality validation fails."""


@dataclass(frozen=True)
class AggTradesAuditResult:
    exchange: str
    symbol: str

    rows: int

    first_trade_time: pd.Timestamp | None
    last_trade_time: pd.Timestamp | None

    first_agg_trade_id: int | None
    last_agg_trade_id: int | None

    duplicate_count: int

    id_gap_count: int
    missing_id_count: int

    invalid_price_count: int
    invalid_quantity_count: int
    invalid_quote_quantity_count: int

    invalid_taker_side_count: int
    invalid_maker_mapping_count: int

    timestamp_disorder_count: int

    structurally_valid: bool
    continuous: bool
    valid: bool


def validate_aggtrades(
    df: pd.DataFrame,
) -> None:
    """
    Validate canonical aggTrades.

    Raises AggTradesQualityError on structural violations.

    This function does not require ID continuity because
    historical bootstrap may intentionally cover only
    a partial time range.
    """

    if df.empty:
        return

    required_columns = [
        "exchange",
        "symbol",
        "agg_trade_id",
        "price",
        "quantity",
        "quote_quantity",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
        "buyer_is_maker",
        "taker_side",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise AggTradesQualityError(
            f"Missing aggTrades columns: {missing_columns}"
        )

    if df["exchange"].nunique() != 1:
        raise AggTradesQualityError(
            "aggTrades contains multiple exchanges"
        )

    if df["symbol"].nunique() != 1:
        raise AggTradesQualityError(
            "aggTrades contains multiple symbols"
        )

    if df["agg_trade_id"].duplicated().any():
        raise AggTradesQualityError(
            "Duplicate agg_trade_id detected"
        )

    if (
        df["price"].isna().any()
        or (df["price"] <= 0).any()
    ):
        raise AggTradesQualityError(
            "Invalid price detected"
        )

    if (
        df["quantity"].isna().any()
        or (df["quantity"] <= 0).any()
    ):
        raise AggTradesQualityError(
            "Invalid quantity detected"
        )

    if (
        df["quote_quantity"].isna().any()
        or (df["quote_quantity"] <= 0).any()
    ):
        raise AggTradesQualityError(
            "Invalid quote_quantity detected"
        )

    if df["trade_time"].isna().any():
        raise AggTradesQualityError(
            "Invalid trade_time detected"
        )

    allowed_sides = {
        "buy",
        "sell",
    }

    invalid_side = (
        ~df["taker_side"]
        .isin(allowed_sides)
    )

    if invalid_side.any():
        raise AggTradesQualityError(
            "Invalid taker_side detected"
        )

    expected_side = np.where(
        df["buyer_is_maker"],
        "sell",
        "buy",
    )

    maker_mapping_valid = (
        df["taker_side"]
        .astype(str)
        .to_numpy()
        == expected_side
    )

    if not maker_mapping_valid.all():
        raise AggTradesQualityError(
            "buyer_is_maker -> taker_side mapping invalid"
        )

    expected_quote = (
        df["price"]
        * df["quantity"]
    )

    if not np.allclose(
        df["quote_quantity"],
        expected_quote,
        rtol=1e-12,
        atol=1e-8,
        equal_nan=False,
    ):
        raise AggTradesQualityError(
            "quote_quantity does not match price * quantity"
        )


def audit_aggtrades(
    df: pd.DataFrame,
) -> AggTradesAuditResult:
    """
    Audit canonical aggTrades.

    Unlike validate_aggtrades(), this function reports
    continuity problems instead of raising on them.
    """

    if df.empty:
        return AggTradesAuditResult(
            exchange="unknown",
            symbol="unknown",
            rows=0,
            first_trade_time=None,
            last_trade_time=None,
            first_agg_trade_id=None,
            last_agg_trade_id=None,
            duplicate_count=0,
            id_gap_count=0,
            missing_id_count=0,
            invalid_price_count=0,
            invalid_quantity_count=0,
            invalid_quote_quantity_count=0,
            invalid_taker_side_count=0,
            invalid_maker_mapping_count=0,
            timestamp_disorder_count=0,
            structurally_valid=False,
            continuous=False,
            valid=False,
        )

    work = (
        df.copy()
        .sort_values(
            [
                "agg_trade_id",
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

    duplicate_count = int(
        work["agg_trade_id"]
        .duplicated()
        .sum()
    )

    unique_ids = (
        work["agg_trade_id"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    id_diff = (
        unique_ids.diff()
    )

    id_gaps = (
        id_diff
        > 1
    )

    id_gap_count = int(
        id_gaps.sum()
    )

    missing_id_count = int(
        (
            id_diff[
                id_gaps
            ]
            - 1
        ).sum()
    )

    invalid_price_count = int(
        (
            work["price"].isna()
            | (work["price"] <= 0)
        ).sum()
    )

    invalid_quantity_count = int(
        (
            work["quantity"].isna()
            | (work["quantity"] <= 0)
        ).sum()
    )

    expected_quote = (
        work["price"]
        * work["quantity"]
    )

    quote_valid = np.isclose(
        work["quote_quantity"],
        expected_quote,
        rtol=1e-12,
        atol=1e-8,
        equal_nan=False,
    )

    invalid_quote_quantity_count = int(
        (~quote_valid).sum()
    )

    invalid_taker_side_count = int(
        (
            ~work["taker_side"]
            .isin(
                [
                    "buy",
                    "sell",
                ]
            )
        ).sum()
    )

    expected_side = np.where(
        work["buyer_is_maker"],
        "sell",
        "buy",
    )

    maker_mapping_valid = (
        work["taker_side"]
        .astype(str)
        .to_numpy()
        == expected_side
    )

    invalid_maker_mapping_count = int(
        (~maker_mapping_valid).sum()
    )

    time_diff = (
        work["trade_time"]
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
        and invalid_price_count == 0
        and invalid_quantity_count == 0
        and invalid_quote_quantity_count == 0
        and invalid_taker_side_count == 0
        and invalid_maker_mapping_count == 0
        and timestamp_disorder_count == 0
    )

    continuous = (
        id_gap_count == 0
    )

    valid = (
        structurally_valid
    )

    return AggTradesAuditResult(
        exchange=exchange,
        symbol=symbol,
        rows=len(work),
        first_trade_time=work["trade_time"].min(),
        last_trade_time=work["trade_time"].max(),
        first_agg_trade_id=int(
            unique_ids.iloc[0]
        ),
        last_agg_trade_id=int(
            unique_ids.iloc[-1]
        ),
        duplicate_count=duplicate_count,
        id_gap_count=id_gap_count,
        missing_id_count=missing_id_count,
        invalid_price_count=invalid_price_count,
        invalid_quantity_count=invalid_quantity_count,
        invalid_quote_quantity_count=invalid_quote_quantity_count,
        invalid_taker_side_count=invalid_taker_side_count,
        invalid_maker_mapping_count=invalid_maker_mapping_count,
        timestamp_disorder_count=timestamp_disorder_count,
        structurally_valid=structurally_valid,
        continuous=continuous,
        valid=valid,
    )
