from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
AGGTRADE_COLUMNS = [
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
}
class AggTradeSchemaError(ValueError):
    """Raised when aggTrades data violates the canonical schema."""
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
def normalize_aggtrades(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize raw/parsed aggTrades into the canonical
    CryptoLab schema.
    Expected semantic fields
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
    Derived
    -------
    quote_quantity
        price * quantity
    taker_side
        buy  when buyer_is_maker == False
        sell when buyer_is_maker == True
    """
    if df.empty:
        return pd.DataFrame(
            columns=AGGTRADE_COLUMNS
        )
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
            f"Missing aggTrades columns: {missing}"
        )
    result = df.copy()
    # --------------------------------------------------------
    # Timestamp
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
        ).astype("int64")
    float_columns = [
        "price",
        "quantity",
    ]
    for column in float_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("float64")
    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------
    result["buyer_is_maker"] = (
        result["buyer_is_maker"]
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
        result["buyer_is_maker"]
        .map(
            {
                False: "buy",
                True: "sell",
            }
        )
        .astype("string")
    )
    # --------------------------------------------------------
    # Metadata
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
        .reset_index(drop=True)
    )
