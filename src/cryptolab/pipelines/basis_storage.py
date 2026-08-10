from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class BasisStorageError(RuntimeError):
    """Raised when basis storage fails."""


BASIS_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "contract_type",
    "period",
    "timestamp",
    "index_price",
    "futures_price",
    "basis",
    "basis_rate",
    "annualized_basis_rate",
]


def _get_raw_root() -> Path:
    """
    Resolve raw-data root.
    """

    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.raw",
            "data/raw",
        )
    )


def _basis_root(
    exchange: str,
    symbol: str,
    period: str,
) -> Path:
    """
    Canonical basis storage root.
    """

    return (
        _get_raw_root()
        / "derivatives"
        / "premium_basis"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"period={period}"
    )


def normalize_basis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize basis history to canonical schema.
    """

    if df.empty:
        return pd.DataFrame(
            columns=BASIS_COLUMNS
        )

    missing = [
        column
        for column in BASIS_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise BasisStorageError(
            f"Missing basis columns: {missing}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise BasisStorageError(
            "Invalid basis timestamp"
        )

    required_numeric = [
        "index_price",
        "futures_price",
        "basis",
        "basis_rate",
    ]

    for column in required_numeric:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype("float64")

    result[
        "annualized_basis_rate"
    ] = pd.to_numeric(
        result[
            "annualized_basis_rate"
        ],
        errors="coerce",
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

    result["contract_type"] = (
        result["contract_type"]
        .astype("string")
        .str.upper()
    )

    result["period"] = (
        result["period"]
        .astype("string")
    )

    return (
        result[
            BASIS_COLUMNS
        ]
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_basis(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save basis history into daily Parquet partitions.

    Existing daily files are merged and deduplicated.
    """

    if df.empty:
        return []

    work = normalize_basis(
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

    periods = (
        work["period"]
        .dropna()
        .unique()
        .tolist()
    )

    contract_types = (
        work["contract_type"]
        .dropna()
        .unique()
        .tolist()
    )

    if (
        len(exchanges) != 1
        or len(symbols) != 1
        or len(periods) != 1
        or len(contract_types) != 1
    ):
        raise BasisStorageError(
            "save_basis requires one "
            "exchange/symbol/period/contract_type"
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

    root = _basis_root(
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

            existing = normalize_basis(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = normalize_basis(
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


def list_basis_files(
    exchange: str,
    symbol: str,
    period: str,
) -> list[Path]:
    """
    List all stored basis partitions.
    """

    root = _basis_root(
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


def read_basis(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.DataFrame:
    """
    Read all locally stored basis history.
    """

    files = list_basis_files(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not files:
        return pd.DataFrame(
            columns=BASIS_COLUMNS
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    return normalize_basis(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def get_latest_basis_timestamp(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.Timestamp | None:
    """
    Return latest locally stored basis timestamp.
    """

    files = list_basis_files(
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

    df = normalize_basis(
        df
    )

    return pd.Timestamp(
        df["timestamp"].iloc[-1]
    )
