from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


def _utc_now() -> datetime:
    """
    Return timezone-aware current UTC time.

    This clock is used for point-in-time receive metadata.
    """

    return datetime.now(
        timezone.utc
    )


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

    received_at_monotonic
    ---------------------
    Monotonic process clock used for collector control logic.

    received_at
    -----------
    UTC wall-clock timestamp captured at the WebSocket receive
    boundary.

    For kind="message", this timestamp is captured immediately
    after ws.recv() returns and BEFORE decoding/parsing.

    This is the canonical exact available_at timestamp for new
    liquidation observations.

    The field is optional for backward compatibility with older
    tests/custom stream factories. Production WebSocket messages
    emitted by stream_liquidation_messages always populate it.
    """

    received_at_monotonic: float
    kind: str
    payload: str | None
    received_at: datetime | None = None


def stream_liquidation_messages(
    symbol: str = "BTCUSDT",
    receive_timeout_seconds: float = 10.0,
    connect_timeout_seconds: float = 15.0,
    create_connection_fn: Callable | None = None,
    utc_now_fn: Callable[[], datetime] = _utc_now,
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

    Point-in-time contract
    ----------------------
    For every real message:

        received_at

    is captured immediately after recv() returns.

    This timestamp represents the earliest local time at which
    CryptoLab can prove that the message was available.

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

                # Capture receive clocks immediately after
                # recv() returns and before any decoding/parsing.
                received_at = utc_now_fn()
                received_at_monotonic = (
                    time.monotonic()
                )

            except websocket.WebSocketTimeoutException:
                yield LiquidationWebSocketMessage(
                    received_at_monotonic=(
                        time.monotonic()
                    ),
                    kind="heartbeat",
                    payload=None,
                    received_at=utc_now_fn(),
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
                    received_at_monotonic
                ),
                kind="message",
                payload=message,
                received_at=received_at,
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
