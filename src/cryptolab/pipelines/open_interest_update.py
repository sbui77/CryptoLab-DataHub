from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cryptolab.pipelines.open_interest_storage import (
    get_latest_open_interest_timestamp,
    save_open_interest,
)
from cryptolab.sources.derivatives_open_interest import (
    fetch_open_interest_history,
)


class OpenInterestUpdateError(RuntimeError):
    """Raised when OI update fails."""


@dataclass(frozen=True)
class OpenInterestUpdateResult:
    exchange: str
    symbol: str
    period: str

    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None

    rows_fetched: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    bootstrap: bool
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


def update_open_interest(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    period: str = "5m",
) -> OpenInterestUpdateResult:
    """
    Bootstrap or incrementally update OI history.

    Bootstrap:
        fetch latest available 500 rows.

    Incremental:
        start from latest local timestamp + one period.

    Note:
        Binance historical endpoint only exposes the
        latest limited history window, so DataHub should
        run this regularly to preserve its own history.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "OI updater supports Binance only"
        )

    if period not in PERIOD_TO_DELTA:
        raise ValueError(
            f"Unsupported period: {period}"
        )

    latest = (
        get_latest_open_interest_timestamp(
            exchange=exchange,
            symbol=symbol,
            period=period,
        )
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    if latest is None:
        df = fetch_open_interest_history(
            symbol=symbol,
            period=period,
            limit=500,
        )

        if df.empty:
            raise OpenInterestUpdateError(
                "Bootstrap returned no OI data"
            )

        save_open_interest(
            df
        )

        return OpenInterestUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            start_time=None,
            end_time=None,
            rows_fetched=len(df),
            first_timestamp=pd.Timestamp(
                df["timestamp"].iloc[0]
            ),
            last_timestamp=pd.Timestamp(
                df["timestamp"].iloc[-1]
            ),
            bootstrap=True,
            up_to_date=True,
        )

    # ========================================================
    # INCREMENTAL
    # ========================================================

    start_time = (
        latest
        + PERIOD_TO_DELTA[
            period
        ]
    )

    df = fetch_open_interest_history(
        symbol=symbol,
        period=period,
        start_time=start_time,
        limit=500,
    )

    if df.empty:
        return OpenInterestUpdateResult(
            exchange=exchange,
            symbol=symbol,
            period=period,
            start_time=start_time,
            end_time=None,
            rows_fetched=0,
            first_timestamp=None,
            last_timestamp=None,
            bootstrap=False,
            up_to_date=True,
        )

    df = df[
        df["timestamp"]
        > latest
    ].copy()

    if not df.empty:
        save_open_interest(
            df
        )

    return OpenInterestUpdateResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        start_time=start_time,
        end_time=None,
        rows_fetched=len(df),
        first_timestamp=(
            pd.Timestamp(
                df["timestamp"].iloc[0]
            )
            if not df.empty
            else None
        ),
        last_timestamp=(
            pd.Timestamp(
                df["timestamp"].iloc[-1]
            )
            if not df.empty
            else None
        ),
        bootstrap=False,
        up_to_date=True,
    )
