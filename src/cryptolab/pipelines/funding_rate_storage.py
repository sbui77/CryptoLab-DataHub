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


FUNDING_RATE_BASE_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "funding_time",
    "funding_rate",
    "mark_price",
    "rate_type",
]


FUNDING_RATE_TIME_COLUMNS = [
    "available_at",
    "available_at_quality",
    "ingested_at",
    "ingested_at_quality",
]


FUNDING_RATE_COLUMNS = (
    FUNDING_RATE_BASE_COLUMNS
    + FUNDING_RATE_TIME_COLUMNS
)


VALID_TIME_QUALITIES = {
    "exact",
    "derived",
    "unknown",
}


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


def _empty_funding_rate() -> pd.DataFrame:
    """
    Return empty canonical funding DataFrame.
    """

    result = pd.DataFrame(
        columns=FUNDING_RATE_COLUMNS
    )

    result["funding_time"] = pd.Series(
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
    Normalize funding point-in-time metadata.

    Legacy partition
    ----------------
    If all PIT columns are absent:

        available_at
            = funding_time

        available_at_quality
            = derived

        ingested_at
            = NaT

        ingested_at_quality
            = unknown

    PIT-aware partition
    -------------------
    If PIT metadata is present, all four columns
    must exist and their semantics must be valid.

    Important
    ---------
    Different Parquet partitions may have different
    schema vintages.

    Each partition must therefore be normalized
    independently BEFORE cross-partition concatenation.
    """

    metadata_columns = set(
        FUNDING_RATE_TIME_COLUMNS
    )

    present = {
        column
        for column in metadata_columns
        if column in result.columns
    }

    # --------------------------------------------------------
    # Pure legacy schema
    # --------------------------------------------------------

    if not present:
        result["available_at"] = (
            result["funding_time"]
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

    # --------------------------------------------------------
    # Partial schema is unsafe
    # --------------------------------------------------------

    if present != metadata_columns:
        missing = sorted(
            metadata_columns
            - present
        )

        raise FundingRateStorageError(
            "Partial funding point-in-time metadata "
            "detected. Missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Timestamp normalization
    # --------------------------------------------------------

    result[
        "available_at"
    ] = pd.to_datetime(
        result[
            "available_at"
        ],
        utc=True,
        errors="coerce",
    )

    result[
        "ingested_at"
    ] = pd.to_datetime(
        result[
            "ingested_at"
        ],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Quality normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Quality validation
    # --------------------------------------------------------

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
        raise FundingRateStorageError(
            "Invalid funding "
            "available_at_quality"
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
        raise FundingRateStorageError(
            "Invalid funding "
            "ingested_at_quality"
        )

    # --------------------------------------------------------
    # available_at consistency
    # --------------------------------------------------------

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
        raise FundingRateStorageError(
            "Known funding available_at "
            "cannot be null"
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
        raise FundingRateStorageError(
            "Funding "
            "available_at_quality='unknown' "
            "requires available_at=NaT"
        )

    # --------------------------------------------------------
    # ingested_at consistency
    # --------------------------------------------------------

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
        raise FundingRateStorageError(
            "Known funding ingested_at "
            "cannot be null"
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
        raise FundingRateStorageError(
            "Funding "
            "ingested_at_quality='unknown' "
            "requires ingested_at=NaT"
        )

    return result


def normalize_funding_rate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize funding-rate rows to canonical schema.

    Supports both:
        legacy 7-column Funding data
        PIT-aware 11-column Funding data
    """

    if df.empty:
        return _empty_funding_rate()

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

    # --------------------------------------------------------
    # Event timestamp
    # --------------------------------------------------------

    result[
        "funding_time"
    ] = pd.to_datetime(
        result[
            "funding_time"
        ],
        utc=True,
        errors="coerce",
    )

    if (
        result[
            "funding_time"
        ]
        .isna()
        .any()
    ):
        raise FundingRateStorageError(
            "Invalid funding_time"
        )

    # --------------------------------------------------------
    # Numeric data
    # --------------------------------------------------------

    result[
        "funding_rate"
    ] = pd.to_numeric(
        result[
            "funding_rate"
        ],
        errors="raise",
    ).astype(
        "float64"
    )

    result[
        "mark_price"
    ] = pd.to_numeric(
        result[
            "mark_price"
        ],
        errors="coerce",
    ).astype(
        "float64"
    )

    # --------------------------------------------------------
    # Text metadata
    # --------------------------------------------------------

    result[
        "exchange"
    ] = (
        result[
            "exchange"
        ]
        .astype("string")
        .str.lower()
    )

    result[
        "market"
    ] = (
        result[
            "market"
        ]
        .astype("string")
    )

    result[
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype("string")
        .str.upper()
    )

    result[
        "rate_type"
    ] = (
        result[
            "rate_type"
        ]
        .astype("string")
    )

    # --------------------------------------------------------
    # PIT metadata
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Canonical ordering / deduplication
    # --------------------------------------------------------

    return (
        result[
            FUNDING_RATE_COLUMNS
        ]
        .sort_values(
            "funding_time"
        )
        .drop_duplicates(
            subset=[
                "funding_time",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


def save_funding_rate(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save funding events into monthly Parquet partitions.

    Existing partitions are merged and deduplicated.

    Legacy partitions are normalized before merging
    with runtime PIT-aware data.
    """

    if df.empty:
        return []

    work = normalize_funding_rate(
        df
    )

    exchanges = (
        work[
            "exchange"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    symbols = (
        work[
            "symbol"
        ]
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

    work[
        "year"
    ] = (
        work[
            "funding_time"
        ]
        .dt.year
    )

    work[
        "month"
    ] = (
        work[
            "funding_time"
        ]
        .dt.month
    )

    saved_files: list[
        Path
    ] = []

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
            .reset_index(
                drop=True
            )
        )

        if output.exists():
            existing = pd.read_parquet(
                output
            )

            # IMPORTANT:
            # normalize existing partition BEFORE concat.
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
    Read all locally stored Funding history.

    Critical mixed-schema rule
    --------------------------
    Each Parquet partition is normalized independently
    BEFORE concatenation.

    This allows legacy partitions and PIT-aware partitions
    to coexist safely during migration.
    """

    files = list_funding_rate_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return _empty_funding_rate()

    frames = []

    for file in files:
        raw = pd.read_parquet(
            file
        )

        normalized = normalize_funding_rate(
            raw
        )

        frames.append(
            normalized
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return normalize_funding_rate(
        result
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
