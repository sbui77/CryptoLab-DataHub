from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator

import websocket

from cryptolab.sources.derivatives_liquidations import (
    get_liquidation_stream_url,
)


class BinanceLiquidationWebSocketError(
    RuntimeError
):
    """Raised when Binance liquidation WebSocket fails."""


@dataclass(frozen=True)
class LiquidationWebSocketMessage:
    """
    One item emitted by the liquidation WebSocket source.

    kind
    ----
    message:
        Real Binance liquidation payload.

    heartbeat:
        No liquidation payload was received during one
        receive-timeout interval, but the WebSocket connection
        is still alive.

    Heartbeat ticks allow the collector to prove connectivity
    independently of liquidation-event frequency.
    """

    received_at_monotonic: float
    kind: str
    payload: str | None


def stream_liquidation_messages(
    symbol: str = "BTCUSDT",
    receive_timeout_seconds: float = 10.0,
    connect_timeout_seconds: float = 15.0,
    create_connection_fn: Callable | None = None,
) -> Iterator[LiquidationWebSocketMessage]:
    """
    Connect to Binance liquidation WebSocket.

    This function represents ONE connection lifecycle.

    It emits:
        kind="message"
            when Binance sends a liquidation event.

        kind="heartbeat"
            when recv() times out while the connection remains
            active.

    Reconnection belongs to the collector layer.
    """

    if receive_timeout_seconds <= 0:
        raise ValueError(
            "receive_timeout_seconds must be positive"
        )

    if connect_timeout_seconds <= 0:
        raise ValueError(
            "connect_timeout_seconds must be positive"
        )

    symbol = symbol.upper()

    url = get_liquidation_stream_url(
        symbol=symbol
    )

    connection_factory = (
        create_connection_fn
        if create_connection_fn is not None
        else websocket.create_connection
    )

    ws = None

    try:
        ws = connection_factory(
            url,
            timeout=connect_timeout_seconds,
        )

        if hasattr(
            ws,
            "settimeout",
        ):
            ws.settimeout(
                receive_timeout_seconds
            )

        while True:
            try:
                message = ws.recv()

            except websocket.WebSocketTimeoutException:
                yield LiquidationWebSocketMessage(
                    received_at_monotonic=(
                        time.monotonic()
                    ),
                    kind="heartbeat",
                    payload=None,
                )

                continue

            if message is None:
                raise BinanceLiquidationWebSocketError(
                    "Binance liquidation WebSocket "
                    "returned None"
                )

            if isinstance(
                message,
                bytes,
            ):
                message = message.decode(
                    "utf-8"
                )

            if not isinstance(
                message,
                str,
            ):
                raise BinanceLiquidationWebSocketError(
                    "Unexpected liquidation WebSocket "
                    f"message type: {type(message).__name__}"
                )

            yield LiquidationWebSocketMessage(
                received_at_monotonic=(
                    time.monotonic()
                ),
                kind="message",
                payload=message,
            )

    except BinanceLiquidationWebSocketError:
        raise

    except (
        websocket.WebSocketException,
        OSError,
    ) as exc:
        raise BinanceLiquidationWebSocketError(
            "Binance liquidation WebSocket error: "
            f"{exc}"
        ) from exc

    finally:
        if ws is not None:
            try:
                ws.close()

            except Exception:
                pass
