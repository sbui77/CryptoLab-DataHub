from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


AGGTRADE_BASE_COLUMNS = [
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


AGGTRADE_TIME_COLUMNS = [
    "available_at",
    "available_at_quality",
    "ingested_at",
    "ingested_at_quality",
]


AGGTRADE_COLUMNS = (
    AGGTRADE_BASE_COLUMNS
    + AGGTRADE_TIME_COLUMNS
)


AGGTRADE_DTYPES = {
    "exchange": "string",
    "symbol": "string",
    "agg_trade_id": "int64",
    "price": "float64",
    "quantity": "float64",
    "quote_quantity": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "buyer_is_maker": "bool",
    "taker_side": "string",
    "available_at_quality": "string",
    "ingested_at_quality": "string",
}


VALID_TIME_QUALITIES = {
    "exact",
    "derived",
    "unknown",
}


class AggTradeSchemaError(ValueError):
    """
    Raised when aggTrades data violates the
    canonical schema.
    """


@dataclass(frozen=True)
class AggTradeSchemaInfo:
    """
    Canonical CryptoLab aggTrades schema metadata.
    """

    exchange: str = "binance"
    timestamp_column: str = "trade_time"
    primary_key: str = "agg_trade_id"
    price_column: str = "price"
    base_quantity_column: str = "quantity"
    quote_quantity_column: str = "quote_quantity"
    side_column: str = "taker_side"

    available_at_column: str = "available_at"
    available_at_quality_column: str = (
        "available_at_quality"
    )

    ingested_at_column: str = "ingested_at"
    ingested_at_quality_column: str = (
        "ingested_at_quality"
    )


def _empty_aggtrades() -> pd.DataFrame:
    """
    Return an empty DataFrame with the complete
    canonical aggTrades schema.
    """

    result = pd.DataFrame(
        columns=AGGTRADE_COLUMNS
    )

    result["trade_time"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["available_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["ingested_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    return result


def _normalize_point_in_time_metadata(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize point-in-time metadata.

    Backward compatibility
    ----------------------
    If all four point-in-time columns are absent,
    the input is treated as legacy aggTrades:

        available_at
            derived from trade_time

        available_at_quality
            derived

        ingested_at
            unknown / NaT

        ingested_at_quality
            unknown

    Partial metadata is rejected because silently
    mixing old and new semantics would weaken the
    point-in-time contract.
    """

    metadata_columns = set(
        AGGTRADE_TIME_COLUMNS
    )

    present = {
        column
        for column in metadata_columns
        if column in result.columns
    }

    if not present:
        result["available_at"] = (
            result["trade_time"]
        )

        result[
            "available_at_quality"
        ] = "derived"

        result["ingested_at"] = pd.Series(
            pd.NaT,
            index=result.index,
            dtype="datetime64[ns, UTC]",
        )

        result[
            "ingested_at_quality"
        ] = "unknown"

        return result

    if present != metadata_columns:
        missing = sorted(
            metadata_columns - present
        )

        raise AggTradeSchemaError(
            "Partial point-in-time metadata "
            "detected. Missing columns: "
            f"{missing}"
        )

    result["available_at"] = pd.to_datetime(
        result["available_at"],
        utc=True,
        errors="coerce",
    )

    result["ingested_at"] = pd.to_datetime(
        result["ingested_at"],
        utc=True,
        errors="coerce",
    )

    result[
        "available_at_quality"
    ] = (
        result[
            "available_at_quality"
        ]
        .astype("string")
        .str.lower()
    )

    result[
        "ingested_at_quality"
    ] = (
        result[
            "ingested_at_quality"
        ]
        .astype("string")
        .str.lower()
    )

    invalid_available_quality = ~result[
        "available_at_quality"
    ].isin(
        VALID_TIME_QUALITIES
    )

    if invalid_available_quality.any():
        raise AggTradeSchemaError(
            "Invalid available_at_quality "
            "values detected"
        )

    invalid_ingested_quality = ~result[
        "ingested_at_quality"
    ].isin(
        VALID_TIME_QUALITIES
    )

    if invalid_ingested_quality.any():
        raise AggTradeSchemaError(
            "Invalid ingested_at_quality "
            "values detected"
        )

    known_available = (
        result[
            "available_at_quality"
        ]
        != "unknown"
    )

    if (
        result.loc[
            known_available,
            "available_at",
        ]
        .isna()
        .any()
    ):
        raise AggTradeSchemaError(
            "Known available_at timestamp "
            "cannot be null"
        )

    unknown_available = (
        result[
            "available_at_quality"
        ]
        == "unknown"
    )

    if (
        result.loc[
            unknown_available,
            "available_at",
        ]
        .notna()
        .any()
    ):
        raise AggTradeSchemaError(
            "available_at_quality='unknown' "
            "requires available_at=NaT"
        )

    known_ingested = (
        result[
            "ingested_at_quality"
        ]
        != "unknown"
    )

    if (
        result.loc[
            known_ingested,
            "ingested_at",
        ]
        .isna()
        .any()
    ):
        raise AggTradeSchemaError(
            "Known ingested_at timestamp "
            "cannot be null"
        )

    unknown_ingested = (
        result[
            "ingested_at_quality"
        ]
        == "unknown"
    )

    if (
        result.loc[
            unknown_ingested,
            "ingested_at",
        ]
        .notna()
        .any()
    ):
        raise AggTradeSchemaError(
            "ingested_at_quality='unknown' "
            "requires ingested_at=NaT"
        )

    return result


def normalize_aggtrades(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize raw/parsed aggTrades into the
    canonical CryptoLab schema.

    Required semantic fields
    ------------------------
    exchange
    symbol
    agg_trade_id
    price
    quantity
    first_trade_id
    last_trade_id
    trade_time
    buyer_is_maker

    Derived market fields
    ---------------------
    quote_quantity
        price * quantity

    taker_side
        buy  when buyer_is_maker == False
        sell when buyer_is_maker == True

    Point-in-time metadata
    ----------------------
    New runtime observations preserve:

        available_at
        available_at_quality
        ingested_at
        ingested_at_quality

    Legacy observations are explicitly marked:

        available_at = trade_time
        available_at_quality = derived
        ingested_at = NaT
        ingested_at_quality = unknown
    """

    if df.empty:
        return _empty_aggtrades()

    required = [
        "exchange",
        "symbol",
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
        "buyer_is_maker",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise AggTradeSchemaError(
            "Missing aggTrades columns: "
            f"{missing}"
        )

    result = df.copy()

    # --------------------------------------------------------
    # Event timestamp
    # --------------------------------------------------------

    result["trade_time"] = pd.to_datetime(
        result["trade_time"],
        utc=True,
        errors="coerce",
    )

    if result["trade_time"].isna().any():
        raise AggTradeSchemaError(
            "Invalid trade_time values detected"
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    integer_columns = [
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
    ]

    for column in integer_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype(
            "int64"
        )

    float_columns = [
        "price",
        "quantity",
    ]

    for column in float_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype(
            "float64"
        )

    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    result["buyer_is_maker"] = (
        result[
            "buyer_is_maker"
        ]
        .astype(bool)
    )

    # --------------------------------------------------------
    # Derived quote notional
    # --------------------------------------------------------

    result["quote_quantity"] = (
        result["price"]
        * result["quantity"]
    )

    # --------------------------------------------------------
    # Aggressor / taker side
    # --------------------------------------------------------

    result["taker_side"] = (
        result[
            "buyer_is_maker"
        ]
        .map(
            {
                False: "buy",
                True: "sell",
            }
        )
        .astype("string")
    )

    # --------------------------------------------------------
    # Market metadata
    # --------------------------------------------------------

    result["exchange"] = (
        result["exchange"]
        .astype("string")
        .str.lower()
    )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.upper()
    )

    # --------------------------------------------------------
    # Point-in-time metadata
    # --------------------------------------------------------

    result = (
        _normalize_point_in_time_metadata(
            result
        )
    )

    # --------------------------------------------------------
    # Canonical string dtypes
    # --------------------------------------------------------

    result[
        "available_at_quality"
    ] = (
        result[
            "available_at_quality"
        ]
        .astype("string")
    )

    result[
        "ingested_at_quality"
    ] = (
        result[
            "ingested_at_quality"
        ]
        .astype("string")
    )

    # --------------------------------------------------------
    # Canonical ordering
    # --------------------------------------------------------

    result = result[
        AGGTRADE_COLUMNS
    ]

    return (
        result
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )
