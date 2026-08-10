from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class LiquidationStorageError(RuntimeError):
    """Raised when liquidation storage fails."""


LIQUIDATION_COLUMNS = [
    "exchange",
    "market",
    "symbol",

    "event_time",
    "order_time",

    "side",
    "liquidation_side",

    "order_type",
    "time_in_force",

    "original_quantity",
    "price",
    "average_price",

    "order_status",

    "last_filled_quantity",
    "filled_quantity",

    "liquidation_notional",
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


def _liquidation_root(
    exchange: str,
    symbol: str,
) -> Path:
    return (
        _get_raw_root()
        / "derivatives"
        / "liquidations"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
    )


def normalize_liquidations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize canonical liquidation events.
    """

    if df.empty:
        return pd.DataFrame(
            columns=LIQUIDATION_COLUMNS
        )

    missing = [
        column
        for column in LIQUIDATION_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise LiquidationStorageError(
            f"Missing liquidation columns: {missing}"
        )

    result = df.copy()

    for column in [
        "event_time",
        "order_time",
    ]:
        result[column] = pd.to_datetime(
            result[column],
            utc=True,
            errors="coerce",
        )

        if result[column].isna().any():
            raise LiquidationStorageError(
                f"Invalid {column}"
            )

    numeric_columns = [
        "original_quantity",
        "price",
        "average_price",
        "last_filled_quantity",
        "filled_quantity",
        "liquidation_notional",
    ]

    for column in numeric_columns:
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

    result["side"] = (
        result["side"]
        .astype("string")
        .str.upper()
    )

    result["liquidation_side"] = (
        result["liquidation_side"]
        .astype("string")
        .str.lower()
    )

    # Event stream has no globally documented liquidation ID.
    # Use a compound event key to make repeated writes
    # idempotent.
    dedup_columns = [
        "event_time",
        "order_time",
        "side",
        "price",
        "original_quantity",
        "filled_quantity",
    ]

    return (
        result[
            LIQUIDATION_COLUMNS
        ]
        .sort_values(
            [
                "event_time",
                "order_time",
            ]
        )
        .drop_duplicates(
            subset=dedup_columns,
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_liquidations(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save liquidation events into daily Parquet partitions.

    Existing partitions are merged and deduplicated.
    """

    if df.empty:
        return []

    work = normalize_liquidations(
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

    if (
        len(exchanges) != 1
        or len(symbols) != 1
    ):
        raise LiquidationStorageError(
            "save_liquidations requires "
            "one exchange/symbol"
        )

    exchange = str(
        exchanges[0]
    )

    symbol = str(
        symbols[0]
    )

    root = _liquidation_root(
        exchange=exchange,
        symbol=symbol,
    )

    work["year"] = (
        work["event_time"]
        .dt.year
    )

    work["month"] = (
        work["event_time"]
        .dt.month
    )

    work["day"] = (
        work["event_time"]
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

            existing = normalize_liquidations(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = normalize_liquidations(
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


def list_liquidation_files(
    exchange: str,
    symbol: str,
) -> list[Path]:
    root = _liquidation_root(
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


def read_liquidations(
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    files = list_liquidation_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return pd.DataFrame(
            columns=LIQUIDATION_COLUMNS
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    return normalize_liquidations(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )
