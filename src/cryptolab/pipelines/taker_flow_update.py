from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from cryptolab.pipelines.taker_flow_storage import (
    get_latest_taker_flow_timestamp,
    save_taker_flow,
)
from cryptolab.sources.derivatives_taker_flow import (
    MAX_TAKER_FLOW_LIMIT,
    fetch_taker_flow_history,
)


class TakerFlowUpdateError(RuntimeError):
    """Raised when futures taker-flow update fails."""


@dataclass(frozen=True)
class TakerFlowUpdateResult:
    exchange: str
    symbol: str
    period: str

    bootstrap: bool

    rows_fetched: int
    pages_fetched: int

    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    target_end_time: pd.Timestamp | None

    up_to_date: bool


PERIOD_TO_DELTA = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


PERIOD_TO_FLOOR = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
}


def _current_period_start(
    period: str,
) -> pd.Timestamp:
    """
    Return current UTC timestamp floored to period.
    """

    return (
        pd.Timestamp.now(
            tz="UTC"
        )
        .floor(
            PERIOD_TO_FLOOR[
                period
            ]
        )
    )


def update_taker_flow(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    period: str = "5m",
    page_limit: int = 500,
    max_pages: int = 100,
    pause_seconds: float = 0.25,
) -> TakerFlowUpdateResult:
    """
    Bootstrap or incrementally update futures taker flow.

    Bootstrap
    ---------
    Fetch latest available page.

    Incremental
    -----------
    Fetch bounded windows from:

        latest_timestamp + one period

    to a fixed target timestamp captured at run start.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Taker-flow updater supports Binance only"
        )

    if period not in PERIOD_TO_DELTA:
        raise ValueError(
            f"Unsupported period: {period}"
        )

    if page_limit <= 0:
        raise ValueError(
            "page_limit must be greater than zero"
        )

    if page_limit > MAX_TAKER_FLOW_LIMIT:
        raise ValueError(
            f"page_limit cannot exceed "
            f"{MAX_TAKER_FLOW_LIMIT}"
        )

    if max_pages <= 0:
        raise ValueError(
            "max_pages must be greater than zero"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    interval = PERIOD_TO_DELTA[
        period
    ]

    latest = (
        get_latest_taker_flow_timestamp(
            exchange=exchange,
            symbol=symbol,
            period=period,
        )
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    if latest is None:
        df = fetch_taker_flow_history(
            symbol=symbol,
            period=period,
            limit=page_limit,
        )

        if df.empty:
            raise TakerFlowUpdateError(
                "Taker-flow bootstrap returned no data"
            )

        save_taker_flow(
            df
        )

        return TakerFlowUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            bootstrap=True,
            rows_fetched=len(df),
            pages_fetched=1,
            first_timestamp=pd.Timestamp(
                df["timestamp"].iloc[0]
            ),
            last_timestamp=pd.Timestamp(
                df["timestamp"].iloc[-1]
            ),
            target_end_time=None,
            up_to_date=True,
        )

    # ========================================================
    # INCREMENTAL TARGET
    # ========================================================

    target_end = _current_period_start(
        period
    )

    next_start = (
        latest
        + interval
    )

    if next_start > target_end:
        return TakerFlowUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            bootstrap=False,
            rows_fetched=0,
            pages_fetched=0,
            first_timestamp=None,
            last_timestamp=None,
            target_end_time=target_end,
            up_to_date=True,
        )

    rows_fetched = 0
    pages_fetched = 0

    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None

    # ========================================================
    # FORWARD PAGINATION
    # ========================================================

    while next_start <= target_end:
        if pages_fetched >= max_pages:
            raise TakerFlowUpdateError(
                "Maximum taker-flow page count reached. "
                f"next_start={next_start} "
                f"target_end={target_end}"
            )

        page_end = min(
            target_end,
            next_start
            + interval
            * (
                page_limit
                - 1
            ),
        )

        page = fetch_taker_flow_history(
            symbol=symbol,
            period=period,
            start_time=next_start,
            end_time=page_end,
            limit=page_limit,
        )

        if not page.empty:
            page = (
                page
                .sort_values("timestamp")
                .drop_duplicates(
                    subset=["timestamp"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

            page = page[
                (
                    page["timestamp"]
                    >= next_start
                )
                & (
                    page["timestamp"]
                    <= page_end
                )
                & (
                    page["timestamp"]
                    > latest
                )
            ].copy()

        if not page.empty:
            save_taker_flow(
                page
            )

            page_first = pd.Timestamp(
                page["timestamp"].iloc[0]
            )

            page_last = pd.Timestamp(
                page["timestamp"].iloc[-1]
            )

            if first_timestamp is None:
                first_timestamp = page_first

            last_timestamp = page_last

            rows_fetched += len(page)

        pages_fetched += 1

        new_next_start = (
            page_end
            + interval
        )

        if new_next_start <= next_start:
            raise TakerFlowUpdateError(
                "Taker-flow pagination did not advance"
            )

        next_start = new_next_start

        if (
            next_start <= target_end
            and pause_seconds > 0
        ):
            time.sleep(
                pause_seconds
            )

    return TakerFlowUpdateResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        bootstrap=False,
        rows_fetched=rows_fetched,
        pages_fetched=pages_fetched,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        target_end_time=target_end,
        up_to_date=True,
    )
