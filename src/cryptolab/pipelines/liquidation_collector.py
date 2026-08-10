from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import random
import time
from typing import Callable, Iterator

import pandas as pd

from cryptolab.pipelines.liquidation_heartbeat import (
    save_liquidation_heartbeat,
)
from cryptolab.pipelines.liquidation_storage import (
    save_liquidations,
)
from cryptolab.sources.binance_liquidation_ws import (
    BinanceLiquidationWebSocketError,
    LiquidationWebSocketMessage,
    stream_liquidation_messages,
)
from cryptolab.sources.derivatives_liquidations import (
    BinanceLiquidationError,
    parse_liquidation_message,
)


class LiquidationCollectorError(RuntimeError):
    """Raised when liquidation collector cannot continue."""


@dataclass(frozen=True)
class LiquidationCollectorResult:
    exchange: str
    symbol: str

    started_at: datetime
    finished_at: datetime

    messages_received: int
    liquidation_rows: int
    rows_saved: int

    heartbeat_rows_saved: int

    reconnect_count: int
    parse_error_count: int
    storage_error_count: int

    stopped_by_limit: bool

    success: bool


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _backoff_seconds(
    reconnect_count: int,
    base_seconds: float,
    max_seconds: float,
    jitter_seconds: float,
    random_fn: Callable[[], float],
) -> float:
    delay = (
        base_seconds
        * (
            2
            ** max(
                reconnect_count - 1,
                0,
            )
        )
    )

    delay = min(
        delay,
        max_seconds,
    )

    jitter = (
        random_fn()
        * jitter_seconds
    )

    return (
        delay
        + jitter
    )


def _heartbeat_frame(
    exchange: str,
    symbol: str,
    connected: bool,
    messages_received: int,
    liquidation_rows: int,
    reconnect_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": exchange,
                "symbol": symbol,
                "heartbeat_time": (
                    pd.Timestamp.now(
                        tz="UTC"
                    )
                ),
                "connected": connected,
                "messages_received": (
                    messages_received
                ),
                "liquidation_rows": (
                    liquidation_rows
                ),
                "reconnect_count": (
                    reconnect_count
                ),
            }
        ]
    )


def _write_heartbeat(
    exchange: str,
    symbol: str,
    connected: bool,
    messages_received: int,
    liquidation_rows: int,
    reconnect_count: int,
) -> int:
    frame = _heartbeat_frame(
        exchange=exchange,
        symbol=symbol,
        connected=connected,
        messages_received=messages_received,
        liquidation_rows=liquidation_rows,
        reconnect_count=reconnect_count,
    )

    save_liquidation_heartbeat(
        frame
    )

    return len(
        frame
    )


def run_liquidation_collector(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    max_messages: int | None = None,
    max_runtime_seconds: float | None = None,
    heartbeat_interval_seconds: float = 30.0,
    reconnect_base_seconds: float = 1.0,
    reconnect_max_seconds: float = 60.0,
    reconnect_jitter_seconds: float = 0.25,
    max_consecutive_reconnects: int = 100,
    stream_factory: Callable[
        ...,
        Iterator[
            LiquidationWebSocketMessage
        ],
    ] = stream_liquidation_messages,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> LiquidationCollectorResult:
    """
    Persistent Binance liquidation collector.

    Heartbeat contract
    ------------------
    Heartbeat is independent of liquidation-event frequency.

    The WebSocket source emits timeout ticks while connected.
    These ticks allow:
        - periodic heartbeat persistence
        - max_runtime enforcement
        - proof that a zero-event period was actually observed

    connected=True:
        WebSocket connection is alive.

    connected=False:
        WebSocket ended/failed or collector stopped.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Liquidation collector currently "
            "supports Binance only"
        )

    if (
        max_messages is not None
        and max_messages <= 0
    ):
        raise ValueError(
            "max_messages must be positive"
        )

    if (
        max_runtime_seconds is not None
        and max_runtime_seconds <= 0
    ):
        raise ValueError(
            "max_runtime_seconds must be positive"
        )

    if heartbeat_interval_seconds <= 0:
        raise ValueError(
            "heartbeat_interval_seconds must be positive"
        )

    if reconnect_base_seconds < 0:
        raise ValueError(
            "reconnect_base_seconds cannot be negative"
        )

    if reconnect_max_seconds < 0:
        raise ValueError(
            "reconnect_max_seconds cannot be negative"
        )

    if reconnect_jitter_seconds < 0:
        raise ValueError(
            "reconnect_jitter_seconds cannot be negative"
        )

    if max_consecutive_reconnects <= 0:
        raise ValueError(
            "max_consecutive_reconnects must be positive"
        )

    started_at = _utc_now()

    start_monotonic = (
        time.monotonic()
    )

    messages_received = 0
    liquidation_rows = 0
    rows_saved = 0

    heartbeat_rows_saved = 0

    reconnect_count = 0
    consecutive_reconnects = 0

    parse_error_count = 0
    storage_error_count = 0

    stopped_by_limit = False

    last_heartbeat_monotonic: (
        float | None
    ) = None

    while True:
        now_monotonic = (
            time.monotonic()
        )

        if (
            max_runtime_seconds
            is not None
            and (
                now_monotonic
                - start_monotonic
            )
            >= max_runtime_seconds
        ):
            stopped_by_limit = True
            break

        try:
            stream = stream_factory(
                symbol=symbol,
                receive_timeout_seconds=min(
                    heartbeat_interval_seconds,
                    10.0,
                ),
            )

            connection_received_item = False

            heartbeat_rows_saved += (
                _write_heartbeat(
                    exchange=exchange,
                    symbol=symbol,
                    connected=True,
                    messages_received=(
                        messages_received
                    ),
                    liquidation_rows=(
                        liquidation_rows
                    ),
                    reconnect_count=(
                        reconnect_count
                    ),
                )
            )

            last_heartbeat_monotonic = (
                time.monotonic()
            )

            for item in stream:
                connection_received_item = True

                now_monotonic = (
                    time.monotonic()
                )

                # ============================================
                # HEARTBEAT
                # ============================================

                if (
                    last_heartbeat_monotonic
                    is None
                    or (
                        now_monotonic
                        - last_heartbeat_monotonic
                    )
                    >= heartbeat_interval_seconds
                ):
                    heartbeat_rows_saved += (
                        _write_heartbeat(
                            exchange=exchange,
                            symbol=symbol,
                            connected=True,
                            messages_received=(
                                messages_received
                            ),
                            liquidation_rows=(
                                liquidation_rows
                            ),
                            reconnect_count=(
                                reconnect_count
                            ),
                        )
                    )

                    last_heartbeat_monotonic = (
                        now_monotonic
                    )

                # ============================================
                # RUNTIME LIMIT
                # ============================================

                if (
                    max_runtime_seconds
                    is not None
                    and (
                        now_monotonic
                        - start_monotonic
                    )
                    >= max_runtime_seconds
                ):
                    stopped_by_limit = True
                    break

                # ============================================
                # CONTROL HEARTBEAT TICK
                # ============================================

                if item.kind == "heartbeat":
                    continue

                if item.kind != "message":
                    raise LiquidationCollectorError(
                        "Unexpected liquidation stream "
                        f"item kind={item.kind}"
                    )

                if item.payload is None:
                    parse_error_count += 1
                    continue

                # ============================================
                # REAL LIQUIDATION MESSAGE
                # ============================================

                messages_received += 1

                try:
                    frame = (
                        parse_liquidation_message(
                            item.payload
                        )
                    )

                except (
                    BinanceLiquidationError,
                    ValueError,
                    TypeError,
                    KeyError,
                ):
                    parse_error_count += 1
                    continue

                if not frame.empty:
                    liquidation_rows += len(
                        frame
                    )

                    try:
                        save_liquidations(
                            frame
                        )

                        rows_saved += len(
                            frame
                        )

                    except Exception as exc:
                        storage_error_count += 1

                        raise LiquidationCollectorError(
                            "Liquidation storage failed: "
                            f"{exc}"
                        ) from exc

                if (
                    max_messages is not None
                    and messages_received
                    >= max_messages
                ):
                    stopped_by_limit = True
                    break

            if stopped_by_limit:
                break

            reconnect_count += 1

            heartbeat_rows_saved += (
                _write_heartbeat(
                    exchange=exchange,
                    symbol=symbol,
                    connected=False,
                    messages_received=(
                        messages_received
                    ),
                    liquidation_rows=(
                        liquidation_rows
                    ),
                    reconnect_count=(
                        reconnect_count
                    ),
                )
            )

            if connection_received_item:
                consecutive_reconnects = 1
            else:
                consecutive_reconnects += 1

        except BinanceLiquidationWebSocketError:
            reconnect_count += 1
            consecutive_reconnects += 1

            heartbeat_rows_saved += (
                _write_heartbeat(
                    exchange=exchange,
                    symbol=symbol,
                    connected=False,
                    messages_received=(
                        messages_received
                    ),
                    liquidation_rows=(
                        liquidation_rows
                    ),
                    reconnect_count=(
                        reconnect_count
                    ),
                )
            )

        if (
            consecutive_reconnects
            > max_consecutive_reconnects
        ):
            raise LiquidationCollectorError(
                "Maximum consecutive liquidation "
                "WebSocket reconnects exceeded"
            )

        delay = _backoff_seconds(
            reconnect_count=(
                consecutive_reconnects
            ),
            base_seconds=(
                reconnect_base_seconds
            ),
            max_seconds=(
                reconnect_max_seconds
            ),
            jitter_seconds=(
                reconnect_jitter_seconds
            ),
            random_fn=random_fn,
        )

        if delay > 0:
            sleep_fn(
                delay
            )

    heartbeat_rows_saved += (
        _write_heartbeat(
            exchange=exchange,
            symbol=symbol,
            connected=False,
            messages_received=(
                messages_received
            ),
            liquidation_rows=(
                liquidation_rows
            ),
            reconnect_count=(
                reconnect_count
            ),
        )
    )

    finished_at = _utc_now()

    return LiquidationCollectorResult(
        exchange=exchange,
        symbol=symbol,
        started_at=started_at,
        finished_at=finished_at,
        messages_received=messages_received,
        liquidation_rows=liquidation_rows,
        rows_saved=rows_saved,
        heartbeat_rows_saved=(
            heartbeat_rows_saved
        ),
        reconnect_count=reconnect_count,
        parse_error_count=parse_error_count,
        storage_error_count=storage_error_count,
        stopped_by_limit=stopped_by_limit,
        success=(
            storage_error_count
            == 0
        ),
    )
