from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cryptolab.sources.binance_liquidation_ws import (
    LiquidationWebSocketMessage,
    stream_liquidation_messages,
)


class FakeWebSocket:
    def __init__(
        self,
        message: str,
    ):
        self.message = message
        self.closed = False
        self.timeout = None
        self.calls = 0

    def settimeout(
        self,
        value,
    ):
        self.timeout = value

    def recv(
        self,
    ):
        self.calls += 1

        if self.calls == 1:
            return self.message

        return None

    def close(
        self,
    ):
        self.closed = True


def test_websocket_message_captures_exact_receive_time():
    raw = '{"e":"forceOrder"}'

    fake = FakeWebSocket(
        raw
    )

    expected = datetime(
        2026,
        8,
        11,
        9,
        0,
        0,
        123456,
        tzinfo=timezone.utc,
    )

    stream = stream_liquidation_messages(
        symbol="BTCUSDT",
        create_connection_fn=(
            lambda *args, **kwargs: fake
        ),
        utc_now_fn=lambda: expected,
    )

    item = next(
        stream
    )

    assert isinstance(
        item,
        LiquidationWebSocketMessage,
    )

    assert (
        item.kind
        == "message"
    )

    assert (
        item.payload
        == raw
    )

    assert (
        item.received_at
        == expected
    )

    assert (
        item.received_at_monotonic
        > 0
    )


def test_message_received_at_is_utc():
    raw = '{"e":"forceOrder"}'

    fake = FakeWebSocket(
        raw
    )

    expected = datetime.now(
        timezone.utc
    )

    stream = stream_liquidation_messages(
        create_connection_fn=(
            lambda *args, **kwargs: fake
        ),
        utc_now_fn=lambda: expected,
    )

    item = next(
        stream
    )

    assert item.received_at is not None

    timestamp = pd.Timestamp(
        item.received_at
    )

    assert timestamp.tzinfo is not None
