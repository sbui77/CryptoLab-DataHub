from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class OpenInterestStorageError(RuntimeError):
    """Raised when OI storage operations fail."""


OPEN_INTEREST_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "open_interest_base",
    "open_interest_quote",
]


def _get_raw_root() -> Path:
    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.raw",
            "data/raw",
        )
    )


def _open_interest_root(
    exchange: str,
    symbol: str,
    period: str,
) -> Path:
    return (
        _get_raw_root()
        / "derivatives"
        / "open_interest"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"period={period}"
    )


def normalize_open_interest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize OI history to canonical schema.
    """

    if df.empty:
        return pd.DataFrame(
            columns=OPEN_INTEREST_COLUMNS
        )

    required = [
        "exchange",
        "market",
        "symbol",
        "period",
        "timestamp",
        "open_interest_base",
        "open_interest_quote",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise OpenInterestStorageError(
            f"Missing OI columns: {missing}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise OpenInterestStorageError(
            "Invalid OI timestamp"
        )

    for column in [
        "open_interest_base",
        "open_interest_quote",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("float64")

    result["exchange"] = (
        result["exchange"]
        .astype("string")
        .str.lower()
    )

    result["market"] = (
        result["market"]
        .astype("string")
    )

    result["symbol"] = (
        result["symbol"]
        .astype("string")
        .str.upper()
    )

    result["period"] = (
        result["period"]
        .astype("string")
    )

    return (
        result[
            OPEN_INTEREST_COLUMNS
        ]
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_open_interest(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save OI history into daily Parquet partitions.

    Existing daily partitions are merged and deduplicated.
    """

    if df.empty:
        return []

    work = normalize_open_interest(
        df
    )

    exchanges = (
        work["exchange"]
        .unique()
        .tolist()
    )

    symbols = (
        work["symbol"]
        .unique()
        .tolist()
    )

    periods = (
        work["period"]
        .unique()
        .tolist()
    )

    if (
        len(exchanges) != 1
        or len(symbols) != 1
        or len(periods) != 1
    ):
        raise OpenInterestStorageError(
            "save_open_interest requires one "
            "exchange/symbol/period"
        )

    exchange = str(
        exchanges[0]
    )

    symbol = str(
        symbols[0]
    )

    period = str(
        periods[0]
    )

    root = _open_interest_root(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    work["year"] = (
        work["timestamp"]
        .dt.year
    )

    work["month"] = (
        work["timestamp"]
        .dt.month
    )

    work["day"] = (
        work["timestamp"]
        .dt.day
    )

    saved_files: list[Path] = []

    grouped = work.groupby(
        [
            "year",
            "month",
            "day",
        ],
        sort=True,
    )

    for (
        year,
        month,
        day,
    ), partition in grouped:

        directory = (
            root
            / f"year={int(year)}"
            / f"month={int(month):02d}"
            / f"day={int(day):02d}"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            directory
            / "data.parquet"
        )

        partition = (
            partition
            .drop(
                columns=[
                    "year",
                    "month",
                    "day",
                ]
            )
            .reset_index(drop=True)
        )

        if output.exists():
            existing = pd.read_parquet(
                output
            )

            existing = normalize_open_interest(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = normalize_open_interest(
            partition
        )

        partition.to_parquet(
            output,
            index=False,
            compression="snappy",
        )

        saved_files.append(
            output
        )

    return saved_files


def list_open_interest_files(
    exchange: str,
    symbol: str,
    period: str,
) -> list[Path]:
    root = _open_interest_root(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not root.exists():
        return []

    return sorted(
        root.glob(
            "year=*/month=*/day=*/data.parquet"
        )
    )


def read_open_interest(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.DataFrame:
    files = list_open_interest_files(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not files:
        return pd.DataFrame(
            columns=OPEN_INTEREST_COLUMNS
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    return normalize_open_interest(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def get_latest_open_interest_timestamp(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.Timestamp | None:
    files = list_open_interest_files(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not files:
        return None

    df = pd.read_parquet(
        files[-1]
    )

    if df.empty:
        return None

    df = normalize_open_interest(
        df
    )

    return pd.Timestamp(
        df[
            "timestamp"
        ].iloc[-1]
    )
