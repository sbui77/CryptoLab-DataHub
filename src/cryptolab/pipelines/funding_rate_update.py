from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from cryptolab.pipelines.funding_rate_storage import (
    get_latest_funding_time,
    save_funding_rate,
)
from cryptolab.sources.derivatives_funding import (
    MAX_FUNDING_LIMIT,
    fetch_funding_rate_history,
)


class FundingRateUpdateError(RuntimeError):
    """Raised when funding-rate update fails."""


@dataclass(frozen=True)
class FundingRateUpdateResult:
    exchange: str
    symbol: str

    bootstrap: bool

    rows_fetched: int
    pages_fetched: int

    first_funding_time: pd.Timestamp | None
    last_funding_time: pd.Timestamp | None

    up_to_date: bool


def update_funding_rate(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    bootstrap_start_time: pd.Timestamp | None = None,
    page_limit: int = 1000,
    max_pages: int = 10_000,
    pause_seconds: float = 0.05,
) -> FundingRateUpdateResult:
    """
    Bootstrap or incrementally update Binance funding history.

    Bootstrap behavior
    ------------------
    If local storage is empty:

    - bootstrap_start_time is None:
        fetch the most recent page only.

    - bootstrap_start_time is supplied:
        paginate forward from that time.

    Incremental behavior
    --------------------
    Continue from latest local funding_time + 1 millisecond.

    Since Binance returns funding rows in ascending order,
    pagination is deterministic.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Funding updater supports Binance only"
        )

    if page_limit <= 0:
        raise ValueError(
            "page_limit must be greater than zero"
        )

    if page_limit > MAX_FUNDING_LIMIT:
        raise ValueError(
            f"page_limit cannot exceed "
            f"{MAX_FUNDING_LIMIT}"
        )

    if max_pages <= 0:
        raise ValueError(
            "max_pages must be greater than zero"
        )

    if pause_seconds < 0:
        raise ValueError(
            "pause_seconds cannot be negative"
        )

    latest_local = get_latest_funding_time(
        exchange=exchange,
        symbol=symbol,
    )

    bootstrap = (
        latest_local is None
    )

    # ========================================================
    # SIMPLE BOOTSTRAP — MOST RECENT PAGE
    # ========================================================

    if (
        bootstrap
        and bootstrap_start_time is None
    ):
        df = fetch_funding_rate_history(
            symbol=symbol,
            limit=page_limit,
        )

        if df.empty:
            raise FundingRateUpdateError(
                "Funding bootstrap returned no data"
            )

        save_funding_rate(
            df
        )

        return FundingRateUpdateResult(
            exchange=exchange,
            symbol=symbol,
            bootstrap=True,
            rows_fetched=len(df),
            pages_fetched=1,
            first_funding_time=pd.Timestamp(
                df["funding_time"].iloc[0]
            ),
            last_funding_time=pd.Timestamp(
                df["funding_time"].iloc[-1]
            ),
            up_to_date=True,
        )

    # ========================================================
    # FORWARD PAGINATION
    # ========================================================

    if bootstrap:
        next_start = pd.Timestamp(
            bootstrap_start_time
        )

        if next_start.tzinfo is None:
            next_start = (
                next_start
                .tz_localize("UTC")
            )
        else:
            next_start = (
                next_start
                .tz_convert("UTC")
            )

    else:
        next_start = (
            latest_local
            + pd.Timedelta(
                milliseconds=1
            )
        )

    rows_fetched = 0
    pages_fetched = 0

    first_funding_time: (
        pd.Timestamp | None
    ) = None

    last_funding_time: (
        pd.Timestamp | None
    ) = None

    while True:
        if pages_fetched >= max_pages:
            raise FundingRateUpdateError(
                "Maximum funding page count reached"
            )

        page = fetch_funding_rate_history(
            symbol=symbol,
            start_time=next_start,
            limit=page_limit,
        )

        if page.empty:
            break

        page = (
            page
            .sort_values("funding_time")
            .drop_duplicates(
                subset=["funding_time"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        if latest_local is not None:
            page = page[
                page["funding_time"]
                > latest_local
            ].copy()

        if page.empty:
            break

        page_first = pd.Timestamp(
            page[
                "funding_time"
            ].iloc[0]
        )

        page_last = pd.Timestamp(
            page[
                "funding_time"
            ].iloc[-1]
        )

        save_funding_rate(
            page
        )

        if first_funding_time is None:
            first_funding_time = (
                page_first
            )

        last_funding_time = (
            page_last
        )

        rows_fetched += len(page)
        pages_fetched += 1

        if len(page) < page_limit:
            break

        new_start = (
            page_last
            + pd.Timedelta(
                milliseconds=1
            )
        )

        if new_start <= next_start:
            raise FundingRateUpdateError(
                "Funding pagination did not advance"
            )

        next_start = new_start

        if pause_seconds > 0:
            time.sleep(
                pause_seconds
            )

    return FundingRateUpdateResult(
        exchange=exchange,
        symbol=symbol,
        bootstrap=bootstrap,
        rows_fetched=rows_fetched,
        pages_fetched=pages_fetched,
        first_funding_time=first_funding_time,
        last_funding_time=last_funding_time,
        up_to_date=True,
    )
