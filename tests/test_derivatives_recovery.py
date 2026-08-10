from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import cryptolab.pipelines.derivatives_update as derivatives_update
import cryptolab.pipelines.liquidation_collector as liquidation_collector

from cryptolab.sources.binance_liquidation_ws import (
    BinanceLiquidationWebSocketError,
    LiquidationWebSocketMessage,
)


# ============================================================
# FAKE DERIVATIVES RESULTS
# ============================================================


@dataclass(frozen=True)
class FakeOIResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    start_time: str = "2026-08-10T10:00:00Z"
    end_time: str = "2026-08-10T11:00:00Z"
    rows_fetched: int = 12
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    bootstrap: bool = False
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeFundingResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    bootstrap: bool = False
    rows_fetched: int = 1
    pages_fetched: int = 1
    first_funding_time: str = "2026-08-10T08:00:00Z"
    last_funding_time: str = "2026-08-10T08:00:00Z"
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeBasisResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    contract_type: str = "PERPETUAL"
    bootstrap: bool = False
    rows_fetched: int = 12
    pages_fetched: int = 1
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    target_end_time: str = "2026-08-10T10:55:00Z"
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeTakerResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    bootstrap: bool = False
    rows_fetched: int = 12
    pages_fetched: int = 1
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    target_end_time: str = "2026-08-10T10:55:00Z"
    up_to_date: bool = True


def install_rest_success(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        derivatives_update,
        "update_open_interest",
        lambda **kwargs: FakeOIResult(),
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_funding_rate",
        lambda **kwargs: FakeFundingResult(),
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_basis",
        lambda **kwargs: FakeBasisResult(),
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_taker_flow",
        lambda **kwargs: FakeTakerResult(),
    )


# ============================================================
# REST FAILURE / RECOVERY
# ============================================================


def test_rest_partial_failure_does_not_abort_other_streams(
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(
        derivatives_update,
        "update_open_interest",
        lambda **kwargs: (
            calls.append("oi")
            or FakeOIResult()
        ),
    )

    def fail_funding(
        **kwargs,
    ):
        calls.append("funding")

        raise RuntimeError(
            "temporary funding failure"
        )

    monkeypatch.setattr(
        derivatives_update,
        "update_funding_rate",
        fail_funding,
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_basis",
        lambda **kwargs: (
            calls.append("basis")
            or FakeBasisResult()
        ),
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_taker_flow",
        lambda **kwargs: (
            calls.append("taker")
            or FakeTakerResult()
        ),
    )

    result = (
        derivatives_update.update_derivatives(
            continue_on_error=True,
        )
    )

    assert result.success is False

    assert result.failures == (
        "funding_rate",
    )

    assert calls == [
        "oi",
        "funding",
        "basis",
        "taker",
    ]

    assert result.open_interest.success is True
    assert result.funding_rate.success is False
    assert result.basis.success is True
    assert result.taker_flow.success is True


def test_next_run_recovers_previous_failed_stream(
    monkeypatch,
):
    state = {
        "funding_attempts": 0,
    }

    monkeypatch.setattr(
        derivatives_update,
        "update_open_interest",
        lambda **kwargs: FakeOIResult(),
    )

    def funding(
        **kwargs,
    ):
        state[
            "funding_attempts"
        ] += 1

        if (
            state[
                "funding_attempts"
            ]
            == 1
        ):
            raise RuntimeError(
                "temporary failure"
            )

        return FakeFundingResult()

    monkeypatch.setattr(
        derivatives_update,
        "update_funding_rate",
        funding,
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_basis",
        lambda **kwargs: FakeBasisResult(),
    )

    monkeypatch.setattr(
        derivatives_update,
        "update_taker_flow",
        lambda **kwargs: FakeTakerResult(),
    )

    first = (
        derivatives_update.update_derivatives(
            continue_on_error=True,
        )
    )

    second = (
        derivatives_update.update_derivatives(
            continue_on_error=True,
        )
    )

    assert first.success is False

    assert first.funding_rate.success is False

    assert second.success is True

    assert second.funding_rate.success is True

    assert state[
        "funding_attempts"
    ] == 2


def test_failed_run_does_not_poison_next_run(
    monkeypatch,
):
    install_rest_success(
        monkeypatch
    )

    first = (
        derivatives_update.update_derivatives()
    )

    second = (
        derivatives_update.update_derivatives()
    )

    assert first.success is True
    assert second.success is True

    assert first.failure_count == 0
    assert second.failure_count == 0


# ============================================================
# LIQUIDATION WEBSOCKET RECOVERY
# ============================================================


def make_ws_message(
    payload: str = "good",
) -> LiquidationWebSocketMessage:
    return LiquidationWebSocketMessage(
        received_at_monotonic=1.0,
        kind="message",
        payload=payload,
    )


def make_liquidation_frame(
    event_time: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance"
            ],

            "market": [
                "usdm_perpetual"
            ],

            "symbol": [
                "BTCUSDT"
            ],

            "event_time": pd.to_datetime(
                [
                    event_time
                ],
                utc=True,
            ),

            "order_time": pd.to_datetime(
                [
                    event_time
                ],
                utc=True,
            ),

            "side": [
                "SELL"
            ],

            "liquidation_side": [
                "long"
            ],

            "order_type": [
                "LIMIT"
            ],

            "time_in_force": [
                "IOC"
            ],

            "original_quantity": [
                1.0
            ],

            "price": [
                65000.0
            ],

            "average_price": [
                65000.0
            ],

            "order_status": [
                "FILLED"
            ],

            "last_filled_quantity": [
                1.0
            ],

            "filled_quantity": [
                1.0
            ],

            "liquidation_notional": [
                65000.0
            ],
        }
    )


def test_websocket_disconnect_reconnects_and_continues(
    monkeypatch,
):
    connection_attempts = {
        "count": 0,
    }

    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        connection_attempts[
            "count"
        ] += 1

        if (
            connection_attempts[
                "count"
            ]
            == 1
        ):
            raise (
                BinanceLiquidationWebSocketError(
                    "connection dropped"
                )
            )

        yield make_ws_message(
            "good"
        )

    monkeypatch.setattr(
        liquidation_collector,
        "parse_liquidation_message",
        lambda payload: (
            make_liquidation_frame(
                "2026-08-10T12:00:00Z"
            )
        ),
    )

    saved: list[
        pd.DataFrame
    ] = []

    monkeypatch.setattr(
        liquidation_collector,
        "save_liquidations",
        lambda df: saved.append(
            df.copy()
        ),
    )

    monkeypatch.setattr(
        liquidation_collector,
        "save_liquidation_heartbeat",
        lambda df: [],
    )

    sleeps: list[
        float
    ] = []

    result = (
        liquidation_collector
        .run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=1,
            reconnect_jitter_seconds=0,
            sleep_fn=sleeps.append,
        )
    )

    assert result.success is True

    assert result.reconnect_count == 1

    assert result.messages_received == 1

    assert result.rows_saved == 1

    assert connection_attempts[
        "count"
    ] == 2

    assert sleeps == [
        1
    ]

    assert len(
        saved
    ) == 1


def test_reconnect_preserves_previous_saved_event(
    monkeypatch,
):
    connection_attempts = {
        "count": 0,
    }

    def stream_factory(
        symbol,
        receive_timeout_seconds,
    ):
        connection_attempts[
            "count"
        ] += 1

        if (
            connection_attempts[
                "count"
            ]
            == 1
        ):
            yield make_ws_message(
                "event-1"
            )

            raise (
                BinanceLiquidationWebSocketError(
                    "disconnect"
                )
            )

        yield make_ws_message(
            "event-2"
        )

    def parser(
        payload,
    ):
        if payload == "event-1":
            return make_liquidation_frame(
                "2026-08-10T12:00:00Z"
            )

        return make_liquidation_frame(
            "2026-08-10T12:01:00Z"
        )

    monkeypatch.setattr(
        liquidation_collector,
        "parse_liquidation_message",
        parser,
    )

    saved: list[
        pd.DataFrame
    ] = []

    monkeypatch.setattr(
        liquidation_collector,
        "save_liquidations",
        lambda df: saved.append(
            df.copy()
        ),
    )

    monkeypatch.setattr(
        liquidation_collector,
        "save_liquidation_heartbeat",
        lambda df: [],
    )

    result = (
        liquidation_collector
        .run_liquidation_collector(
            max_messages=2,
            stream_factory=stream_factory,
            reconnect_base_seconds=0,
            reconnect_jitter_seconds=0,
        )
    )

    assert result.success is True

    assert result.messages_received == 2

    assert result.rows_saved == 2

    assert result.reconnect_count == 1

    assert len(
        saved
    ) == 2

    first_time = (
        saved[
            0
        ][
            "event_time"
        ].iloc[0]
    )

    second_time = (
        saved[
            1
        ][
            "event_time"
        ].iloc[0]
    )

    assert (
        first_time
        < second_time
    )


# ============================================================
# RESTART / IDEMPOTENT PERSISTENCE CONTRACT
# ============================================================


def test_restart_same_event_can_be_safely_persisted_twice(
    tmp_path,
    monkeypatch,
):
    """
    Simulate:
        process A saves event
        process dies
        process B sees same event and saves again

    Storage contract should ultimately deduplicate it.

    We emulate persistence locally so this test does not alter
    real project data.
    """

    store = (
        tmp_path
        / "liquidations.parquet"
    )

    def fake_save(
        df: pd.DataFrame,
    ):
        incoming = (
            df.copy()
        )

        if store.exists():
            existing = (
                pd.read_parquet(
                    store
                )
            )

            combined = pd.concat(
                [
                    existing,
                    incoming,
                ],
                ignore_index=True,
            )

        else:
            combined = incoming

        dedup_columns = [
            "event_time",
            "side",
            "price",
            "original_quantity",
            "filled_quantity",
        ]

        combined = (
            combined
            .drop_duplicates(
                subset=dedup_columns,
                keep="last",
            )
            .sort_values(
                "event_time"
            )
            .reset_index(drop=True)
        )

        combined.to_parquet(
            store,
            index=False,
        )

        return [
            store
        ]

    frame = make_liquidation_frame(
        "2026-08-10T12:00:00Z"
    )

    fake_save(
        frame
    )

    fake_save(
        frame
    )

    stored = pd.read_parquet(
        store
    )

    assert len(
        stored
    ) == 1
