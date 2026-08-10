from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd

from cryptolab.pipelines.basis_storage import (
    get_latest_basis_timestamp,
    save_basis,
)
from cryptolab.sources.derivatives_basis import (
    MAX_BASIS_LIMIT,
    fetch_basis_history,
)


class BasisUpdateError(RuntimeError):
    """Raised when basis update fails."""


@dataclass(frozen=True)
class BasisUpdateResult:
    exchange: str
    symbol: str
    period: str
    contract_type: str

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


def _floor_utc_now(
    period: str,
) -> pd.Timestamp:
    """
    Return current UTC time floored to the canonical period.

    Basis timestamps represent the start of each period.
    """

    now = pd.Timestamp.now(
        tz="UTC"
    )

    aliases = {
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

    if period not in aliases:
        raise ValueError(
            f"Unsupported period: {period}"
        )

    return now.floor(
        aliases[period]
    )


def update_basis(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    contract_type: str = "PERPETUAL",
    period: str = "5m",
    page_limit: int = 500,
    max_pages: int = 100,
    pause_seconds: float = 0.25,
) -> BasisUpdateResult:
    """
    Bootstrap or incrementally update Binance basis history.

    Bootstrap
    ---------
    When local storage is empty, fetch the latest available page.

    Incremental
    -----------
    Start at:

        latest_local_timestamp + one period

    and paginate forward to a fixed target time captured at
    the beginning of the run.

    Important
    ---------
    Binance basis requests are sent with BOTH startTime and
    endTime during incremental updates.

    Although the public documentation describes these fields
    as optional, the live USDⓈ-M endpoint may reject a request
    containing startTime without endTime.

    Each request is bounded to at most page_limit periods,
    avoiding ambiguous pagination behavior.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()
    contract_type = (
        contract_type.upper()
    )

    if exchange != "binance":
        raise ValueError(
            "Basis updater supports Binance only"
        )

    if period not in PERIOD_TO_DELTA:
        raise ValueError(
            f"Unsupported period: {period}"
        )

    if page_limit <= 0:
        raise ValueError(
            "page_limit must be greater than zero"
        )

    if page_limit > MAX_BASIS_LIMIT:
        raise ValueError(
            f"page_limit cannot exceed "
            f"{MAX_BASIS_LIMIT}"
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
        get_latest_basis_timestamp(
            exchange=exchange,
            symbol=symbol,
            period=period,
        )
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    if latest is None:
        df = fetch_basis_history(
            pair=symbol,
            contract_type=contract_type,
            period=period,
            limit=page_limit,
        )

        if df.empty:
            raise BasisUpdateError(
                "Basis bootstrap returned no data"
            )

        save_basis(
            df
        )

        return BasisUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            contract_type=contract_type,
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
    # FIXED UPDATE TARGET
    # ========================================================

    target_end = _floor_utc_now(
        period
    )

    next_start = (
        latest
        + interval
    )

    # Nothing to request yet.
    if next_start > target_end:
        return BasisUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            contract_type=contract_type,
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

    first_timestamp: (
        pd.Timestamp | None
    ) = None

    last_timestamp: (
        pd.Timestamp | None
    ) = None

    # ========================================================
    # FORWARD PAGINATION
    # ========================================================

    while next_start <= target_end:

        if pages_fetched >= max_pages:
            raise BasisUpdateError(
                "Maximum basis page count reached "
                "before update target was completed. "
                f"next_start={next_start} "
                f"target_end={target_end}"
            )

        # A page can contain at most page_limit observations.
        #
        # Example for 5m / limit=500:
        #
        # start + 499 * 5m
        #
        # gives a window containing at most 500 timestamps.
        page_end = min(
            target_end,
            next_start
            + interval
            * (
                page_limit
                - 1
            ),
        )

        page = fetch_basis_history(
            pair=symbol,
            contract_type=contract_type,
            period=period,
            start_time=next_start,
            end_time=page_end,
            limit=page_limit,
        )

        # ----------------------------------------------------
        # No data in this bounded window.
        # Advance beyond the window rather than retry forever.
        # ----------------------------------------------------

        if page.empty:
            next_start = (
                page_end
                + interval
            )

            pages_fetched += 1

            if (
                next_start <= target_end
                and pause_seconds > 0
            ):
                time.sleep(
                    pause_seconds
                )

            continue

        page = (
            page
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # Defensive filtering in case Binance returns records
        # outside the requested window.
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
            page_first = pd.Timestamp(
                page[
                    "timestamp"
                ].iloc[0]
            )

            page_last = pd.Timestamp(
                page[
                    "timestamp"
                ].iloc[-1]
            )

            save_basis(
                page
            )

            if first_timestamp is None:
                first_timestamp = (
                    page_first
                )

            last_timestamp = (
                page_last
            )

            rows_fetched += len(page)

        pages_fetched += 1

        # Advance by the explicit request window rather than
        # relying on number of returned records.
        new_next_start = (
            page_end
            + interval
        )

        if new_next_start <= next_start:
            raise BasisUpdateError(
                "Basis pagination did not advance"
            )

        next_start = (
            new_next_start
        )

        if (
            next_start <= target_end
            and pause_seconds > 0
        ):
            time.sleep(
                pause_seconds
            )

    return BasisUpdateResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        contract_type=contract_type,
        bootstrap=False,
        rows_fetched=rows_fetched,
        pages_fetched=pages_fetched,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        target_end_time=target_end,
        up_to_date=True,
    )