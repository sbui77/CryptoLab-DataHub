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


OPEN_INTEREST_BASE_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "open_interest_base",
    "open_interest_quote",
]


OPEN_INTEREST_TIME_COLUMNS = [
    "available_at",
    "available_at_quality",
    "ingested_at",
    "ingested_at_quality",
]


OPEN_INTEREST_COLUMNS = (
    OPEN_INTEREST_BASE_COLUMNS
    + OPEN_INTEREST_TIME_COLUMNS
)


VALID_TIME_QUALITIES = {
    "exact",
    "derived",
    "unknown",
}


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


def _empty_open_interest() -> pd.DataFrame:
    """
    Return empty canonical OI DataFrame including
    point-in-time metadata.
    """

    result = pd.DataFrame(
        columns=OPEN_INTEREST_COLUMNS
    )

    result["timestamp"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["available_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["ingested_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    return result


def _normalize_time_metadata(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize OI point-in-time metadata.

    Legacy partition
    ----------------
    If all four PIT columns are absent:

        available_at = timestamp
        available_at_quality = derived
        ingested_at = NaT
        ingested_at_quality = unknown

    PIT-aware partition
    -------------------
    If any PIT column exists, all four must exist.

    Each Parquet partition must be normalized before
    cross-partition concatenation.
    """

    metadata_columns = set(
        OPEN_INTEREST_TIME_COLUMNS
    )

    present = {
        column
        for column in metadata_columns
        if column in result.columns
    }

    if not present:
        result["available_at"] = (
            result["timestamp"]
        )

        result[
            "available_at_quality"
        ] = "derived"

        result["ingested_at"] = pd.Series(
            pd.NaT,
            index=result.index,
            dtype="datetime64[ns, UTC]",
        )

        result[
            "ingested_at_quality"
        ] = "unknown"

        return result

    if present != metadata_columns:
        missing = sorted(
            metadata_columns - present
        )

        raise OpenInterestStorageError(
            "Partial OI point-in-time metadata "
            "detected. Missing columns: "
            f"{missing}"
        )

    result["available_at"] = pd.to_datetime(
        result["available_at"],
        utc=True,
        errors="coerce",
    )

    result["ingested_at"] = pd.to_datetime(
        result["ingested_at"],
        utc=True,
        errors="coerce",
    )

    result[
        "available_at_quality"
    ] = (
        result[
            "available_at_quality"
        ]
        .astype("string")
        .str.lower()
    )

    result[
        "ingested_at_quality"
    ] = (
        result[
            "ingested_at_quality"
        ]
        .astype("string")
        .str.lower()
    )

    invalid_available_quality = (
        result[
            "available_at_quality"
        ]
        .isna()
        | ~result[
            "available_at_quality"
        ].isin(
            VALID_TIME_QUALITIES
        )
    )

    if invalid_available_quality.any():
        raise OpenInterestStorageError(
            "Invalid OI available_at_quality"
        )

    invalid_ingested_quality = (
        result[
            "ingested_at_quality"
        ]
        .isna()
        | ~result[
            "ingested_at_quality"
        ].isin(
            VALID_TIME_QUALITIES
        )
    )

    if invalid_ingested_quality.any():
        raise OpenInterestStorageError(
            "Invalid OI ingested_at_quality"
        )

    known_available = (
        result[
            "available_at_quality"
        ]
        != "unknown"
    )

    if (
        result.loc[
            known_available,
            "available_at",
        ]
        .isna()
        .any()
    ):
        raise OpenInterestStorageError(
            "Known OI available_at cannot be null"
        )

    unknown_available = (
        result[
            "available_at_quality"
        ]
        == "unknown"
    )

    if (
        result.loc[
            unknown_available,
            "available_at",
        ]
        .notna()
        .any()
    ):
        raise OpenInterestStorageError(
            "OI available_at_quality='unknown' "
            "requires available_at=NaT"
        )

    known_ingested = (
        result[
            "ingested_at_quality"
        ]
        != "unknown"
    )

    if (
        result.loc[
            known_ingested,
            "ingested_at",
        ]
        .isna()
        .any()
    ):
        raise OpenInterestStorageError(
            "Known OI ingested_at cannot be null"
        )

    unknown_ingested = (
        result[
            "ingested_at_quality"
        ]
        == "unknown"
    )

    if (
        result.loc[
            unknown_ingested,
            "ingested_at",
        ]
        .notna()
        .any()
    ):
        raise OpenInterestStorageError(
            "OI ingested_at_quality='unknown' "
            "requires ingested_at=NaT"
        )

    return result


def normalize_open_interest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize OI history to canonical schema.

    Supports both legacy and PIT-aware observations.
    """

    if df.empty:
        return _empty_open_interest()

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
        ).astype(
            "float64"
        )

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

    result = _normalize_time_metadata(
        result
    )

    result[
        "available_at_quality"
    ] = (
        result[
            "available_at_quality"
        ]
        .astype("string")
    )

    result[
        "ingested_at_quality"
    ] = (
        result[
            "ingested_at_quality"
        ]
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
        .reset_index(
            drop=True
        )
    )


def save_open_interest(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save OI history into daily Parquet partitions.

    Existing daily partitions are merged and deduplicated.

    Existing legacy partitions are normalized before
    merging with PIT-aware runtime observations.
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
    """
    Read all locally stored OI history.

    Critical mixed-schema rule:
    each partition is normalized independently BEFORE
    cross-partition concatenation.
    """

    files = list_open_interest_files(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not files:
        return _empty_open_interest()

    frames: list[pd.DataFrame] = []

    for file in files:
        raw = pd.read_parquet(
            file
        )

        normalized = normalize_open_interest(
            raw
        )

        frames.append(
            normalized
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return normalize_open_interest(
        result
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
