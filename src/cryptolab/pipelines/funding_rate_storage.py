from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class FundingRateStorageError(RuntimeError):
    """Raised when funding-rate storage fails."""


FUNDING_RATE_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "funding_time",
    "funding_rate",
    "mark_price",
    "rate_type",
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


def _funding_rate_root(
    exchange: str,
    symbol: str,
) -> Path:
    """
    Canonical funding-rate storage root.
    """

    return (
        _get_raw_root()
        / "derivatives"
        / "funding_rate"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
    )


def normalize_funding_rate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize funding-rate rows to canonical schema.
    """

    if df.empty:
        return pd.DataFrame(
            columns=FUNDING_RATE_COLUMNS
        )

    required = [
        "exchange",
        "market",
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price",
        "rate_type",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise FundingRateStorageError(
            f"Missing funding columns: {missing}"
        )

    result = df.copy()

    result["funding_time"] = pd.to_datetime(
        result["funding_time"],
        utc=True,
        errors="coerce",
    )

    if result["funding_time"].isna().any():
        raise FundingRateStorageError(
            "Invalid funding_time"
        )

    result["funding_rate"] = pd.to_numeric(
        result["funding_rate"],
        errors="raise",
    ).astype("float64")

    result["mark_price"] = pd.to_numeric(
        result["mark_price"],
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

    result["rate_type"] = (
        result["rate_type"]
        .astype("string")
    )

    return (
        result[
            FUNDING_RATE_COLUMNS
        ]
        .sort_values("funding_time")
        .drop_duplicates(
            subset=[
                "funding_time",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_funding_rate(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save funding events into monthly Parquet partitions.

    Existing partitions are merged and deduplicated.
    """

    if df.empty:
        return []

    work = normalize_funding_rate(
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
        raise FundingRateStorageError(
            "save_funding_rate requires one "
            "exchange/symbol"
        )

    exchange = str(
        exchanges[0]
    )

    symbol = str(
        symbols[0]
    )

    root = _funding_rate_root(
        exchange=exchange,
        symbol=symbol,
    )

    work["year"] = (
        work["funding_time"]
        .dt.year
    )

    work["month"] = (
        work["funding_time"]
        .dt.month
    )

    saved_files: list[Path] = []

    grouped = work.groupby(
        [
            "year",
            "month",
        ],
        sort=True,
    )

    for (
        year,
        month,
    ), partition in grouped:

        directory = (
            root
            / f"year={int(year)}"
            / f"month={int(month):02d}"
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
                ]
            )
            .reset_index(drop=True)
        )

        if output.exists():
            existing = pd.read_parquet(
                output
            )

            existing = normalize_funding_rate(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = normalize_funding_rate(
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


def list_funding_rate_files(
    exchange: str,
    symbol: str,
) -> list[Path]:
    """
    List funding-rate partitions.
    """

    root = _funding_rate_root(
        exchange=exchange,
        symbol=symbol,
    )

    if not root.exists():
        return []

    return sorted(
        root.glob(
            "year=*/month=*/data.parquet"
        )
    )


def read_funding_rate(
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    """
    Read all locally stored funding history.
    """

    files = list_funding_rate_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return pd.DataFrame(
            columns=FUNDING_RATE_COLUMNS
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    return normalize_funding_rate(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def get_latest_funding_time(
    exchange: str,
    symbol: str,
) -> pd.Timestamp | None:
    """
    Return latest locally stored funding event time.
    """

    files = list_funding_rate_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return None

    df = pd.read_parquet(
        files[-1]
    )

    if df.empty:
        return None

    df = normalize_funding_rate(
        df
    )

    return pd.Timestamp(
        df[
            "funding_time"
        ].iloc[-1]
    )
