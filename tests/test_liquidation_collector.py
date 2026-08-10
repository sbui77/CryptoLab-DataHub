from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.liquidation_collector as module
from cryptolab.sources.binance_liquidation_ws import (
    BinanceLiquidationWebSocketError,
    LiquidationWebSocketMessage,
)


def make_message(
    payload: str = "{}",
) -> LiquidationWebSocketMessage:
    return LiquidationWebSocketMessage(
        received_at_monotonic=1.0,
        kind="message",
        payload=payload,
    )


def make_heartbeat() -> LiquidationWebSocketMessage:
    return LiquidationWebSocketMessage(
        received_at_monotonic=1.0,
        kind="heartbeat",
        payload=None,
    )


def make_liquidation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": ["binance"],
            "market": ["usdm_perpetual"],
            "symbol": ["BTCUSDT"],
            "event_time": pd.to_datetime(
                ["2026-08-10T12:00:00Z"],
                utc=True,
            ),
            "order_time": pd.to_datetime(
                ["2026-08-10T12:00:00Z"],
                utc=True,
            ),
            "side": ["SELL"],
            "liquidation_side": ["long"],
            "order_type": ["LIMIT"],
            "time_in_force": ["IOC"],
            "original_quantity": [1.0],
            "price": [65000.0],
            "average_price": [65000.0],
            "order_status": ["FILLED"],
            "last_filled_quantity": [1.0],
            "filled_quantity": [1.0],
            "liquidation_notional": [65000.0],
        }
    )


def install_heartbeat_mock(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "save_liquidation_heartbeat",
        lambda df: [],
    )


def test_collect_one_message(
    monkeypatch,
):
    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        yield make_message(
            '{"event":"test"}'
        )

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        lambda payload: (
            make_liquidation_frame()
        ),
    )

    saved = []

    monkeypatch.setattr(
        module,
        "save_liquidations",
        lambda df: saved.append(
            df.copy()
        ),
    )

    install_heartbeat_mock(
        monkeypatch
    )

    result = (
        module.run_liquidation_collector(
            symbol="BTCUSDT",
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )
    )

    assert result.success is True
    assert result.messages_received == 1
    assert result.liquidation_rows == 1
    assert result.rows_saved == 1
    assert len(saved) == 1


def test_control_heartbeat_not_counted_as_message(
    monkeypatch,
):
    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        yield make_heartbeat()
        yield make_message("good")

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        lambda payload: (
            make_liquidation_frame()
        ),
    )

    monkeypatch.setattr(
        module,
        "save_liquidations",
        lambda df: [],
    )

    install_heartbeat_mock(
        monkeypatch
    )

    result = (
        module.run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )
    )

    assert result.messages_received == 1
    assert result.liquidation_rows == 1


def test_parse_error_does_not_kill_collector(
    monkeypatch,
):
    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        yield make_message("bad")
        yield make_message("good")

    def parser(
        payload,
    ):
        if payload == "bad":
            raise ValueError(
                "bad payload"
            )

        return make_liquidation_frame()

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        parser,
    )

    monkeypatch.setattr(
        module,
        "save_liquidations",
        lambda df: [],
    )

    install_heartbeat_mock(
        monkeypatch
    )

    result = (
        module.run_liquidation_collector(
            max_messages=2,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )
    )

    assert result.messages_received == 2
    assert result.parse_error_count == 1
    assert result.liquidation_rows == 1


def test_reconnect_after_ws_failure(
    monkeypatch,
):
    calls = {
        "count": 0,
    }

    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        calls["count"] += 1

        if calls["count"] == 1:
            raise BinanceLiquidationWebSocketError(
                "disconnect"
            )

        yield make_message(
            "good"
        )

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        lambda payload: (
            make_liquidation_frame()
        ),
    )

    monkeypatch.setattr(
        module,
        "save_liquidations",
        lambda df: [],
    )

    install_heartbeat_mock(
        monkeypatch
    )

    sleeps = []

    result = (
        module.run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=1,
            reconnect_jitter_seconds=0,
            sleep_fn=sleeps.append,
        )
    )

    assert result.messages_received == 1
    assert result.reconnect_count == 1
    assert sleeps == [1]


def test_empty_parsed_frame_is_allowed(
    monkeypatch,
):
    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        yield make_message(
            "empty"
        )

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        lambda payload: (
            pd.DataFrame()
        ),
    )

    install_heartbeat_mock(
        monkeypatch
    )

    result = (
        module.run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )
    )

    assert result.success is True
    assert result.messages_received == 1
    assert result.liquidation_rows == 0
    assert result.rows_saved == 0


def test_storage_failure_is_fatal(
    monkeypatch,
):
    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        yield make_message(
            "good"
        )

    monkeypatch.setattr(
        module,
        "parse_liquidation_message",
        lambda payload: (
            make_liquidation_frame()
        ),
    )

    def fail_save(
        df,
    ):
        raise RuntimeError(
            "disk failure"
        )

    monkeypatch.setattr(
        module,
        "save_liquidations",
        fail_save,
    )

    install_heartbeat_mock(
        monkeypatch
    )

    try:
        module.run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )

    except module.LiquidationCollectorError:
        return

    raise AssertionError(
        "Expected LiquidationCollectorError"
    )


def test_backoff_caps():
    delay = module._backoff_seconds(
        reconnect_count=10,
        base_seconds=1,
        max_seconds=30,
        jitter_seconds=0,
        random_fn=lambda: 0,
    )

    assert delay == 30


def test_unsupported_exchange():
    try:
        module.run_liquidation_collector(
            exchange="other",
            max_messages=1,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
