from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class TakerFlowStorageError(RuntimeError):
    """Raised when futures taker-flow storage fails."""


TAKER_FLOW_BASE_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "buy_volume",
    "sell_volume",
    "buy_sell_ratio",
]


TAKER_FLOW_TIME_COLUMNS = [
    "available_at",
    "available_at_quality",
    "ingested_at",
    "ingested_at_quality",
]


TAKER_FLOW_COLUMNS = (
    TAKER_FLOW_BASE_COLUMNS
    + TAKER_FLOW_TIME_COLUMNS
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


def _taker_flow_root(
    exchange: str,
    symbol: str,
    period: str,
) -> Path:
    return (
        _get_raw_root()
        / "derivatives"
        / "taker_flow"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"period={period}"
    )


def _empty_taker_flow() -> pd.DataFrame:
    """
    Return empty canonical Taker Flow DataFrame.
    """

    result = pd.DataFrame(
        columns=TAKER_FLOW_COLUMNS
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
    Normalize Taker Flow point-in-time metadata.

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

    Cross-partition rule
    --------------------
    Every Parquet partition must be normalized
    independently before concatenation.
    """

    metadata_columns = set(
        TAKER_FLOW_TIME_COLUMNS
    )

    present = {
        column
        for column in metadata_columns
        if column in result.columns
    }

    if not present:
        result[
            "available_at"
        ] = result[
            "timestamp"
        ]

        result[
            "available_at_quality"
        ] = "derived"

        result[
            "ingested_at"
        ] = pd.Series(
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
            metadata_columns
            - present
        )

        raise TakerFlowStorageError(
            "Partial Taker Flow point-in-time metadata "
            "detected. Missing columns: "
            f"{missing}"
        )

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
        raise TakerFlowStorageError(
            "Invalid Taker Flow "
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
        raise TakerFlowStorageError(
            "Invalid Taker Flow "
            "ingested_at_quality"
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
        raise TakerFlowStorageError(
            "Known Taker Flow available_at "
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
        raise TakerFlowStorageError(
            "Taker Flow "
            "available_at_quality='unknown' "
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
        raise TakerFlowStorageError(
            "Known Taker Flow ingested_at "
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
        raise TakerFlowStorageError(
            "Taker Flow "
            "ingested_at_quality='unknown' "
            "requires ingested_at=NaT"
        )

    return result


def normalize_taker_flow(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize futures taker-flow DataFrame.

    Supports both legacy and PIT-aware observations.
    """

    if df.empty:
        return _empty_taker_flow()

    missing = [
        column
        for column in TAKER_FLOW_BASE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise TakerFlowStorageError(
            f"Missing taker-flow columns: {missing}"
        )

    result = df.copy()

    result[
        "timestamp"
    ] = pd.to_datetime(
        result[
            "timestamp"
        ],
        utc=True,
        errors="coerce",
    )

    if (
        result[
            "timestamp"
        ]
        .isna()
        .any()
    ):
        raise TakerFlowStorageError(
            "Invalid taker-flow timestamp"
        )

    for column in [
        "buy_volume",
        "sell_volume",
        "buy_sell_ratio",
    ]:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="raise",
        ).astype(
            "float64"
        )

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
        "period"
    ] = (
        result[
            "period"
        ]
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
            TAKER_FLOW_COLUMNS
        ]
        .sort_values(
            "timestamp"
        )
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


def save_taker_flow(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save futures Taker Flow into daily Parquet partitions.

    Existing legacy partitions are normalized before
    merging with PIT-aware runtime observations.
    """

    if df.empty:
        return []

    work = normalize_taker_flow(
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

    periods = (
        work[
            "period"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if (
        len(exchanges) != 1
        or len(symbols) != 1
        or len(periods) != 1
    ):
        raise TakerFlowStorageError(
            "save_taker_flow requires one "
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

    root = _taker_flow_root(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    work[
        "year"
    ] = (
        work[
            "timestamp"
        ]
        .dt.year
    )

    work[
        "month"
    ] = (
        work[
            "timestamp"
        ]
        .dt.month
    )

    work[
        "day"
    ] = (
        work[
            "timestamp"
        ]
        .dt.day
    )

    saved_files: list[
        Path
    ] = []

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
            .reset_index(
                drop=True
            )
        )

        if output.exists():
            existing = pd.read_parquet(
                output
            )

            existing = normalize_taker_flow(
                existing
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = normalize_taker_flow(
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


def list_taker_flow_files(
    exchange: str,
    symbol: str,
    period: str,
) -> list[Path]:
    root = _taker_flow_root(
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


def read_taker_flow(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.DataFrame:
    """
    Read all locally stored Taker Flow history.

    Critical mixed-schema rule:
    each partition is normalized independently BEFORE
    cross-partition concatenation.
    """

    files = list_taker_flow_files(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if not files:
        return _empty_taker_flow()

    frames: list[
        pd.DataFrame
    ] = []

    for file in files:
        raw = pd.read_parquet(
            file
        )

        normalized = normalize_taker_flow(
            raw
        )

        frames.append(
            normalized
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return normalize_taker_flow(
        result
    )


def get_latest_taker_flow_timestamp(
    exchange: str,
    symbol: str,
    period: str,
) -> pd.Timestamp | None:
    files = list_taker_flow_files(
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

    df = normalize_taker_flow(
        df
    )

    return pd.Timestamp(
        df[
            "timestamp"
        ].iloc[-1]
    )
