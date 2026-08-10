from __future__ import annotations

import json
from typing import Any

import pandas as pd


class BinanceLiquidationError(RuntimeError):
    """Raised when Binance liquidation data is invalid."""


def get_liquidation_stream_url(
    symbol: str = "BTCUSDT",
) -> str:
    """
    Return Binance USDⓈ-M liquidation WebSocket URL.
    """

    stream = (
        symbol
        .lower()
        + "@forceOrder"
    )

    return (
        "wss://fstream.binance.com/"
        f"market/ws/{stream}"
    )


def parse_liquidation_message(
    message: str | dict[str, Any],
) -> pd.DataFrame:
    """
    Parse one Binance forceOrder WebSocket event.

    Canonical columns
    -----------------
    exchange
    market
    symbol

    event_time
    order_time

    side
    liquidation_side

    order_type
    time_in_force

    original_quantity
    price
    average_price

    order_status

    last_filled_quantity
    filled_quantity

    liquidation_notional

    Interpretation
    --------------
    Binance side is the liquidation ORDER side.

    SELL liquidation order:
        closes a LONG position
        -> liquidation_side = long

    BUY liquidation order:
        closes a SHORT position
        -> liquidation_side = short
    """

    if isinstance(
        message,
        str,
    ):
        try:
            payload = json.loads(
                message
            )

        except json.JSONDecodeError as exc:
            raise BinanceLiquidationError(
                "Invalid liquidation JSON"
            ) from exc

    elif isinstance(
        message,
        dict,
    ):
        payload = message

    else:
        raise BinanceLiquidationError(
            "message must be JSON string or dict"
        )

    if (
        payload.get("e")
        != "forceOrder"
    ):
        raise BinanceLiquidationError(
            "Unexpected event type: "
            f"{payload.get('e')}"
        )

    order = payload.get(
        "o"
    )

    if not isinstance(
        order,
        dict,
    ):
        raise BinanceLiquidationError(
            "Missing liquidation order payload"
        )

    required = {
        "s",
        "S",
        "o",
        "f",
        "q",
        "p",
        "ap",
        "X",
        "l",
        "z",
        "T",
    }

    missing = (
        required
        - set(order)
    )

    if missing:
        raise BinanceLiquidationError(
            "Missing liquidation fields: "
            f"{sorted(missing)}"
        )

    side = str(
        order["S"]
    ).upper()

    if side == "SELL":
        liquidation_side = "long"

    elif side == "BUY":
        liquidation_side = "short"

    else:
        raise BinanceLiquidationError(
            f"Unknown liquidation side: {side}"
        )

    original_quantity = float(
        order["q"]
    )

    price = float(
        order["p"]
    )

    average_price = float(
        order["ap"]
    )

    last_filled_quantity = float(
        order["l"]
    )

    filled_quantity = float(
        order["z"]
    )

    # Prefer actual executed average price.
    # Fall back to order price if average price is zero.
    notional_price = (
        average_price
        if average_price > 0
        else price
    )

    liquidation_notional = (
        filled_quantity
        * notional_price
    )

    event_time = pd.to_datetime(
        payload["E"],
        unit="ms",
        utc=True,
    )

    order_time = pd.to_datetime(
        order["T"],
        unit="ms",
        utc=True,
    )

    row = {
        "exchange": "binance",
        "market": "usdm_perpetual",
        "symbol": str(
            order["s"]
        ).upper(),

        "event_time": event_time,
        "order_time": order_time,

        "side": side,
        "liquidation_side": (
            liquidation_side
        ),

        "order_type": str(
            order["o"]
        ),

        "time_in_force": str(
            order["f"]
        ),

        "original_quantity": (
            original_quantity
        ),

        "price": price,

        "average_price": (
            average_price
        ),

        "order_status": str(
            order["X"]
        ),

        "last_filled_quantity": (
            last_filled_quantity
        ),

        "filled_quantity": (
            filled_quantity
        ),

        "liquidation_notional": (
            liquidation_notional
        ),
    }

    return pd.DataFrame(
        [row]
    )
