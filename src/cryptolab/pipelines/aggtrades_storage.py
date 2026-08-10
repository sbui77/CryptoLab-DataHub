from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.schemas.aggtrades import (
    AGGTRADE_COLUMNS,
    normalize_aggtrades,
)


class AggTradesStorageError(RuntimeError):
    """Raised when aggTrades storage operations fail."""


def _get_raw_root() -> Path:
    """
    Resolve raw data root from CryptoLab config.
    """

    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.raw",
            "data/raw",
        )
    )


def _aggtrades_root(
    exchange: str,
    symbol: str,
) -> Path:
    """
    Return canonical aggTrades storage root.
    """

    return (
        _get_raw_root()
        / "trade_flow"
        / "aggtrades"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
    )


def save_aggtrades(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save canonical aggTrades into daily Parquet partitions.

    Partitioning:
        year
        month
        day

    Existing daily files are merged with incoming data and
    deduplicated by agg_trade_id.

    Raw trades are never fabricated or interpolated.
    """

    if df.empty:
        return []

    work = normalize_aggtrades(
        df
    )

    exchanges = (
        work["exchange"]
        .dropna()
        .unique()
        .tolist()
    )

    symbols = (
        work["symbol"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(exchanges) != 1:
        raise AggTradesStorageError(
            "save_aggtrades requires exactly one exchange"
        )

    if len(symbols) != 1:
        raise AggTradesStorageError(
            "save_aggtrades requires exactly one symbol"
        )

    exchange = str(
        exchanges[0]
    )

    symbol = str(
        symbols[0]
    )

    work = (
        work
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(drop=True)
    )

    work["year"] = (
        work["trade_time"]
        .dt.year
    )

    work["month"] = (
        work["trade_time"]
        .dt.month
    )

    work["day"] = (
        work["trade_time"]
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

    root = _aggtrades_root(
        exchange=exchange,
        symbol=symbol,
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

            existing = normalize_aggtrades(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = (
            partition
            .drop_duplicates(
                subset=[
                    "agg_trade_id",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "trade_time",
                    "agg_trade_id",
                ]
            )
            .reset_index(drop=True)
        )

        partition = normalize_aggtrades(
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


def list_aggtrades_files(
    exchange: str,
    symbol: str,
) -> list[Path]:
    """
    List all stored aggTrades Parquet partitions.
    """

    root = _aggtrades_root(
        exchange=exchange,
        symbol=symbol,
    )

    if not root.exists():
        return []

    return sorted(
        root.glob(
            "year=*/month=*/day=*/data.parquet"
        )
    )


def read_aggtrades(
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    """
    Read all stored aggTrades for an exchange/symbol.
    """

    files = list_aggtrades_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return pd.DataFrame(
            columns=AGGTRADE_COLUMNS
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = normalize_aggtrades(
        result
    )

    result = (
        result
        .drop_duplicates(
            subset=[
                "agg_trade_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def read_latest_aggtrade(
    exchange: str,
    symbol: str,
) -> pd.Series | None:
    """
    Return the latest stored aggregate trade.

    Returns None when no data exists.
    """

    files = list_aggtrades_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return None

    latest_file = files[-1]

    df = pd.read_parquet(
        latest_file
    )

    if df.empty:
        return None

    df = normalize_aggtrades(
        df
    )

    df = (
        df
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df.iloc[-1]


def get_latest_agg_trade_id(
    exchange: str,
    symbol: str,
) -> int | None:
    """
    Return latest stored agg_trade_id.
    """

    latest = read_latest_aggtrade(
        exchange=exchange,
        symbol=symbol,
    )

    if latest is None:
        return None

    return int(
        latest[
            "agg_trade_id"
        ]
    )
