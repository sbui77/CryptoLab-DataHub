from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from cryptolab.pipelines.aggtrades_storage import (
    save_aggtrades,
)
from cryptolab.sources.aggtrades import (
    fetch_aggtrades_by_time,
    fetch_aggtrades_from_id,
)


class AggTradesBackfillError(RuntimeError):
    """Raised when historical aggTrades backfill fails."""


@dataclass(frozen=True)
class AggTradesBackfillResult:
    exchange: str
    symbol: str

    requested_start_time: pd.Timestamp
    requested_end_time: pd.Timestamp

    seed_id: int

    rows_fetched: int
    rows_saved: int
    pages_fetched: int

    first_saved_id: int | None
    last_saved_id: int | None

    first_saved_time: pd.Timestamp | None
    last_saved_time: pd.Timestamp | None

    crossed_end_time: bool
    complete: bool


def _to_utc(
    value: pd.Timestamp | str,
) -> pd.Timestamp:
    result = pd.Timestamp(
        value
    )

    if result.tzinfo is None:
        return result.tz_localize(
            "UTC"
        )

    return result.tz_convert(
        "UTC"
    )


def _find_seed_trade(
    symbol: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    search_window: pd.Timedelta = pd.Timedelta(
        minutes=5
    ),
) -> pd.DataFrame:
    """
    Locate the first aggTrade at or after start_time.

    Binance time-based query is used only to discover
    a deterministic starting agg_trade_id.

    If a short window contains no trades, advance the window
    until either a trade is found or end_time is reached.
    """

    cursor = start_time

    while cursor <= end_time:
        window_end = min(
            cursor
            + search_window,
            end_time,
        )

        seed = fetch_aggtrades_by_time(
            symbol=symbol,
            start_time=cursor,
            end_time=window_end,
            limit=1000,
        )

        if not seed.empty:
            seed = (
                seed[
                    seed["trade_time"]
                    >= start_time
                ]
                .sort_values(
                    [
                        "trade_time",
                        "agg_trade_id",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            if not seed.empty:
                return seed

        cursor = (
            window_end
            + pd.Timedelta(
                milliseconds=1
            )
        )

    raise AggTradesBackfillError(
        "Could not locate an aggTrade "
        f"between {start_time} and {end_time}"
    )


def backfill_aggtrades(
    symbol: str,
    start_time: pd.Timestamp | str,
    end_time: pd.Timestamp | str,
    exchange: str = "binance",
    page_limit: int = 1000,
    max_pages: int = 100_000,
    pause_seconds: float = 0.20,
) -> AggTradesBackfillResult:
    """
    Backfill Binance Spot aggTrades for a fixed time range.

    Algorithm
    ---------
    1. Locate first aggTrade at/after start_time using time query.
    2. Capture its agg_trade_id.
    3. Paginate forward deterministically using fromId.
    4. Save only trades inside [start_time, end_time].
    5. Stop only after Binance returns a trade beyond end_time.

    Storage is idempotent because save_aggtrades() deduplicates
    by agg_trade_id.

    Re-running this function is therefore safe.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Historical aggTrades backfill currently "
            "supports Binance only"
        )

    start = _to_utc(
        start_time
    )

    end = _to_utc(
        end_time
    )

    if end <= start:
        raise ValueError(
            "end_time must be after start_time"
        )

    if page_limit <= 0:
        raise ValueError(
            "page_limit must be greater than zero"
        )

    if page_limit > 1000:
        raise ValueError(
            "page_limit cannot exceed 1000"
        )

    if max_pages <= 0:
        raise ValueError(
            "max_pages must be greater than zero"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    # ========================================================
    # FIND STARTING ID
    # ========================================================

    seed = _find_seed_trade(
        symbol=symbol,
        start_time=start,
        end_time=end,
    )

    seed_id = int(
        seed[
            "agg_trade_id"
        ].iloc[0]
    )

    next_id = seed_id

    rows_fetched = 0
    rows_saved = 0
    pages_fetched = 0

    first_saved_id: int | None = None
    last_saved_id: int | None = None

    first_saved_time: (
        pd.Timestamp | None
    ) = None

    last_saved_time: (
        pd.Timestamp | None
    ) = None

    crossed_end_time = False

    # ========================================================
    # FORWARD PAGINATION
    # ========================================================

    while True:

        if pages_fetched >= max_pages:
            raise AggTradesBackfillError(
                "Maximum page count reached before "
                "end_time was crossed. "
                f"next_id={next_id} "
                f"end_time={end}"
            )

        page = fetch_aggtrades_from_id(
            symbol=symbol,
            from_id=next_id,
            limit=page_limit,
        )

        if page.empty:
            raise AggTradesBackfillError(
                "Binance returned an empty page before "
                "historical target was completed. "
                f"fromId={next_id}"
            )

        page = (
            page
            .sort_values(
                "agg_trade_id"
            )
            .reset_index(
                drop=True
            )
        )

        page_first_id = int(
            page[
                "agg_trade_id"
            ].iloc[0]
        )

        page_last_id = int(
            page[
                "agg_trade_id"
            ].iloc[-1]
        )

        if page_first_id != next_id:
            raise AggTradesBackfillError(
                "Unexpected pagination start. "
                f"requested={next_id}, "
                f"returned={page_first_id}"
            )

        if (
            page[
                "agg_trade_id"
            ]
            .duplicated()
            .any()
        ):
            raise AggTradesBackfillError(
                "Duplicate agg_trade_id inside API page"
            )

        if not (
            page[
                "agg_trade_id"
            ]
            .is_monotonic_increasing
        ):
            raise AggTradesBackfillError(
                "agg_trade_id page is not monotonic"
            )

        rows_fetched += len(
            page
        )

        pages_fetched += 1

        page_last_time = pd.Timestamp(
            page[
                "trade_time"
            ].iloc[-1]
        )

        # ====================================================
        # FILTER REQUESTED TIME RANGE
        # ====================================================

        inside = page[
            (
                page["trade_time"]
                >= start
            )
            & (
                page["trade_time"]
                <= end
            )
        ].copy()

        if not inside.empty:
            save_aggtrades(
                inside
            )

            rows_saved += len(
                inside
            )

            inside_first_id = int(
                inside[
                    "agg_trade_id"
                ].iloc[0]
            )

            inside_last_id = int(
                inside[
                    "agg_trade_id"
                ].iloc[-1]
            )

            inside_first_time = pd.Timestamp(
                inside[
                    "trade_time"
                ].iloc[0]
            )

            inside_last_time = pd.Timestamp(
                inside[
                    "trade_time"
                ].iloc[-1]
            )

            if first_saved_id is None:
                first_saved_id = (
                    inside_first_id
                )

                first_saved_time = (
                    inside_first_time
                )

            last_saved_id = (
                inside_last_id
            )

            last_saved_time = (
                inside_last_time
            )

        # ====================================================
        # END CONDITION
        #
        # We deliberately require a returned trade AFTER
        # end_time. That proves pagination crossed the target.
        # ====================================================

        if (
            page[
                "trade_time"
            ]
            > end
        ).any():
            crossed_end_time = True
            break

        new_next_id = (
            page_last_id
            + 1
        )

        if new_next_id <= next_id:
            raise AggTradesBackfillError(
                "Pagination did not advance"
            )

        next_id = (
            new_next_id
        )

        if pause_seconds > 0:
            time.sleep(
                pause_seconds
            )

    complete = (
        crossed_end_time
        and first_saved_time is not None
        and last_saved_time is not None
        and first_saved_time >= start
        and last_saved_time <= end
    )

    return AggTradesBackfillResult(
        exchange=exchange,
        symbol=symbol,

        requested_start_time=start,
        requested_end_time=end,

        seed_id=seed_id,

        rows_fetched=rows_fetched,
        rows_saved=rows_saved,
        pages_fetched=pages_fetched,

        first_saved_id=first_saved_id,
        last_saved_id=last_saved_id,

        first_saved_time=first_saved_time,
        last_saved_time=last_saved_time,

        crossed_end_time=crossed_end_time,
        complete=complete,
    )
