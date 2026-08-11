from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

import cryptolab.pipelines.liquidation_collector as collector
from cryptolab.sources.binance_liquidation_ws import (
    LiquidationWebSocketMessage,
)


def _payload() -> str:
    return json.dumps(
        {
            "e": "forceOrder",
            "E": 1786438800000,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "o": "LIMIT",
                "f": "IOC",
                "q": "1.0",
                "p": "120000.0",
                "ap": "119990.0",
                "X": "FILLED",
                "l": "1.0",
                "z": "1.0",
                "T": 1786438800000,
            },
        }
    )


def test_collector_preserves_exact_ws_receive_time(
    monkeypatch,
):
    received_at = datetime(
        2026,
        8,
        11,
        9,
        0,
        0,
        100000,
        tzinfo=timezone.utc,
    )

    def stream_factory(
        **kwargs,
    ):
        yield LiquidationWebSocketMessage(
            received_at_monotonic=1.0,
            kind="message",
            payload=_payload(),
            received_at=received_at,
        )

    saved_frames = []

    def fake_save_liquidations(
        frame,
    ):
        saved_frames.append(
            frame.copy()
        )

        return []

    monkeypatch.setattr(
        collector,
        "save_liquidations",
        fake_save_liquidations,
    )

    monkeypatch.setattr(
        collector,
        "save_liquidation_heartbeat",
        lambda frame: [],
    )

    result = (
        collector.run_liquidation_collector(
            symbol="BTCUSDT",
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0.0,
            reconnect_max_seconds=0.0,
            reconnect_jitter_seconds=0.0,
        )
    )

    assert result.success
    assert result.messages_received == 1
    assert result.liquidation_rows == 1
    assert result.rows_saved == 1

    assert len(saved_frames) == 1

    frame = saved_frames[0]

    assert (
        frame.iloc[0][
            "available_at"
        ]
        == pd.Timestamp(
            received_at
        )
    )

    assert (
        frame.iloc[0][
            "available_at_quality"
        ]
        == "exact"
    )

    assert (
        frame.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        frame.iloc[0][
            "ingested_at"
        ]
        >= frame.iloc[0][
            "available_at"
        ]
    )


def test_legacy_custom_stream_does_not_fake_exact_receive_time(
    monkeypatch,
):
    def stream_factory(
        **kwargs,
    ):
        yield LiquidationWebSocketMessage(
            received_at_monotonic=1.0,
            kind="message",
            payload=_payload(),
        )

    saved_frames = []

    monkeypatch.setattr(
        collector,
        "save_liquidations",
        lambda frame: (
            saved_frames.append(
                frame.copy()
            )
            or []
        ),
    )

    monkeypatch.setattr(
        collector,
        "save_liquidation_heartbeat",
        lambda frame: [],
    )

    result = (
        collector.run_liquidation_collector(
            max_messages=1,
            stream_factory=stream_factory,
            reconnect_base_seconds=0.0,
            reconnect_max_seconds=0.0,
            reconnect_jitter_seconds=0.0,
        )
    )

    assert result.success

    frame = saved_frames[0]

    assert (
        frame.iloc[0][
            "available_at_quality"
        ]
        == "derived"
    )

    assert (
        frame.iloc[0][
            "available_at"
        ]
        == frame.iloc[0][
            "event_time"
        ]
    )

    assert (
        frame.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )
