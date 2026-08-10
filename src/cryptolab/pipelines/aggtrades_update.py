from __future__ import annotations

from dataclasses import dataclass
import time

from cryptolab.pipelines.aggtrades_storage import (
    get_latest_agg_trade_id,
    save_aggtrades,
)
from cryptolab.sources.aggtrades import (
    fetch_aggtrades_from_id,
    fetch_recent_aggtrades,
)


class AggTradesUpdateError(RuntimeError):
    """Raised when incremental aggTrades update fails."""


@dataclass(frozen=True)
class AggTradesUpdateResult:
    """
    Summary of one incremental aggTrades update.
    """

    exchange: str
    symbol: str

    start_id: int
    target_id: int

    rows_fetched: int
    pages_fetched: int

    first_fetched_id: int | None
    last_fetched_id: int | None

    up_to_date: bool


def _get_remote_latest_id(
    symbol: str,
) -> int:
    """
    Snapshot the latest aggTrade ID currently available
    from Binance.

    This fixed target prevents an updater from endlessly
    chasing newly arriving trades.
    """

    recent = fetch_recent_aggtrades(
        symbol=symbol,
        limit=1,
    )

    if recent.empty:
        raise AggTradesUpdateError(
            "Binance returned no recent aggTrades"
        )

    return int(
        recent[
            "agg_trade_id"
        ].iloc[-1]
    )


def update_aggtrades(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    page_limit: int = 1000,
    max_pages: int = 100_000,
    pause_seconds: float = 0.05,
) -> AggTradesUpdateResult:
    """
    Incrementally update stored Binance aggTrades.

    Preconditions
    -------------
    Local aggTrades storage must already contain at least
    one trade. Historical bootstrap is handled separately.

    Algorithm
    ---------
    1. Read latest stored agg_trade_id.
    2. next_id = latest_id + 1.
    3. Snapshot Binance's latest current agg_trade_id.
    4. Fetch forward using fromId.
    5. Persist each page immediately.
    6. Stop after reaching the original target ID.

    Safety
    ------
    - Page size is capped at 1000.
    - Detects non-advancing pagination.
    - max_pages prevents accidental infinite loops.
    - Data beyond target_id is discarded from this run.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Incremental aggTrades updater currently "
            "supports Binance only"
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

    latest_local_id = get_latest_agg_trade_id(
        exchange=exchange,
        symbol=symbol,
    )

    if latest_local_id is None:
        raise AggTradesUpdateError(
            "Local aggTrades storage is empty. "
            "Bootstrap the dataset before incremental update."
        )

    start_id = (
        int(latest_local_id)
        + 1
    )

    target_id = _get_remote_latest_id(
        symbol=symbol
    )

    # --------------------------------------------------------
    # Already current.
    # --------------------------------------------------------

    if start_id > target_id:
        return AggTradesUpdateResult(
            exchange=exchange,
            symbol=symbol,
            start_id=start_id,
            target_id=target_id,
            rows_fetched=0,
            pages_fetched=0,
            first_fetched_id=None,
            last_fetched_id=None,
            up_to_date=True,
        )

    next_id = start_id

    rows_fetched = 0
    pages_fetched = 0

    first_fetched_id: int | None = None
    last_fetched_id: int | None = None

    # ========================================================
    # FORWARD PAGINATION
    # ========================================================

    while next_id <= target_id:

        if pages_fetched >= max_pages:
            raise AggTradesUpdateError(
                "Maximum page count reached before "
                f"target ID {target_id}. "
                f"Current next ID: {next_id}"
            )

        remaining = (
            target_id
            - next_id
            + 1
        )

        request_limit = min(
            page_limit,
            remaining,
        )

        page = fetch_aggtrades_from_id(
            symbol=symbol,
            from_id=next_id,
            limit=request_limit,
        )

        if page.empty:
            raise AggTradesUpdateError(
                "Binance returned an empty page "
                f"before target ID was reached. "
                f"Requested fromId={next_id}, "
                f"target_id={target_id}"
            )

        page = (
            page[
                page[
                    "agg_trade_id"
                ]
                <= target_id
            ]
            .sort_values(
                "agg_trade_id"
            )
            .reset_index(drop=True)
        )

        if page.empty:
            raise AggTradesUpdateError(
                "Returned page contained no rows "
                "within the update target"
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

        # ----------------------------------------------------
        # fromId is expected to be inclusive.
        #
        # Require the requested trade to be the first row.
        # This catches unexpected API/pagination behavior.
        # ----------------------------------------------------

        if page_first_id != next_id:
            raise AggTradesUpdateError(
                "Unexpected aggTrades pagination start. "
                f"Requested fromId={next_id}, "
                f"returned first ID={page_first_id}"
            )

        if page_last_id < page_first_id:
            raise AggTradesUpdateError(
                "Invalid aggTrade ID ordering in page"
            )

        # ----------------------------------------------------
        # Require unique, strictly increasing IDs in page.
        # ----------------------------------------------------

        if (
            page[
                "agg_trade_id"
            ]
            .duplicated()
            .any()
        ):
            raise AggTradesUpdateError(
                "Duplicate agg_trade_id detected "
                "inside Binance response page"
            )

        if not (
            page[
                "agg_trade_id"
            ]
            .is_monotonic_increasing
        ):
            raise AggTradesUpdateError(
                "agg_trade_id values are not "
                "monotonically increasing"
            )

        # ----------------------------------------------------
        # Persist immediately.
        # ----------------------------------------------------

        save_aggtrades(
            page
        )

        pages_fetched += 1
        rows_fetched += len(page)

        if first_fetched_id is None:
            first_fetched_id = (
                page_first_id
            )

        last_fetched_id = (
            page_last_id
        )

        new_next_id = (
            page_last_id
            + 1
        )

        if new_next_id <= next_id:
            raise AggTradesUpdateError(
                "Pagination did not advance"
            )

        next_id = new_next_id

        if (
            next_id <= target_id
            and pause_seconds > 0
        ):
            time.sleep(
                pause_seconds
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    up_to_date = (
        last_fetched_id is not None
        and last_fetched_id >= target_id
    )

    return AggTradesUpdateResult(
        exchange=exchange,
        symbol=symbol,
        start_id=start_id,
        target_id=target_id,
        rows_fetched=rows_fetched,
        pages_fetched=pages_fetched,
        first_fetched_id=first_fetched_id,
        last_fetched_id=last_fetched_id,
        up_to_date=up_to_date,
    )
