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


LIQUIDATION_BASE_COLUMNS = [
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


LIQUIDATION_TIME_COLUMNS = [
    "available_at",
    "available_at_quality",
    "ingested_at",
    "ingested_at_quality",
]


LIQUIDATION_COLUMNS = (
    LIQUIDATION_BASE_COLUMNS
    + LIQUIDATION_TIME_COLUMNS
)


VALID_TIME_QUALITIES = {
    "exact",
    "derived",
    "unknown",
}


LIQUIDATION_DEDUP_COLUMNS = [
    "event_time",
    "order_time",
    "side",
    "price",
    "original_quantity",
    "filled_quantity",
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


def _empty_liquidations() -> pd.DataFrame:
    """
    Return empty canonical liquidation DataFrame.
    """

    result = pd.DataFrame(
        columns=LIQUIDATION_COLUMNS
    )

    for column in [
        "event_time",
        "order_time",
        "available_at",
        "ingested_at",
    ]:
        result[column] = pd.Series(
            dtype="datetime64[ns, UTC]"
        )

    return result


def _normalize_time_metadata(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize liquidation PIT metadata.

    Legacy rows
    -----------
    If all four PIT columns are absent:

        available_at = event_time
        available_at_quality = derived
        ingested_at = NaT
        ingested_at_quality = unknown

    Runtime WebSocket rows
    ----------------------
        available_at = local WS receive time
        available_at_quality = exact
        ingested_at = local persistence time
        ingested_at_quality = exact

    Partial PIT schemas are rejected.
    """

    metadata_columns = set(
        LIQUIDATION_TIME_COLUMNS
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
            "event_time"
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

        raise LiquidationStorageError(
            "Partial liquidation point-in-time metadata "
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
        raise LiquidationStorageError(
            "Invalid liquidation "
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
        raise LiquidationStorageError(
            "Invalid liquidation "
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
        raise LiquidationStorageError(
            "Known liquidation available_at "
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
        raise LiquidationStorageError(
            "Liquidation "
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
        raise LiquidationStorageError(
            "Known liquidation ingested_at "
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
        raise LiquidationStorageError(
            "Liquidation "
            "ingested_at_quality='unknown' "
            "requires ingested_at=NaT"
        )

    # For exact runtime ingestion, persistence cannot
    # precede local receive time.
    exact_runtime = (
        result[
            "available_at_quality"
        ].eq("exact")
        & result[
            "ingested_at_quality"
        ].eq("exact")
    )

    if (
        result.loc[
            exact_runtime,
            "ingested_at",
        ]
        <
        result.loc[
            exact_runtime,
            "available_at",
        ]
    ).any():
        raise LiquidationStorageError(
            "Liquidation ingested_at cannot precede "
            "exact available_at"
        )

    return result


def normalize_liquidations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize canonical liquidation events.

    Supports both legacy liquidation partitions and
    PIT-aware WebSocket runtime observations.
    """

    if df.empty:
        return _empty_liquidations()

    missing = [
        column
        for column in LIQUIDATION_BASE_COLUMNS
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

    result["order_type"] = (
        result["order_type"]
        .astype("string")
    )

    result["time_in_force"] = (
        result["time_in_force"]
        .astype("string")
    )

    result["order_status"] = (
        result["order_status"]
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

    # Event stream has no globally documented liquidation ID.
    # Continue using the established compound event key.
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
            subset=LIQUIDATION_DEDUP_COLUMNS,
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_liquidations(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save liquidation events into daily Parquet partitions.

    Existing partitions are normalized BEFORE being merged
    with incoming PIT-aware runtime observations.

    This allows legacy and PIT-aware partitions to coexist
    during lazy schema migration.
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
    """
    Read all liquidation history.

    Every partition is normalized independently BEFORE
    cross-partition concatenation, preventing mixed-schema
    PIT metadata from becoming artificial nulls.
    """

    files = list_liquidation_files(
        exchange=exchange,
        symbol=symbol,
    )

    if not files:
        return _empty_liquidations()

    frames: list[pd.DataFrame] = []

    for file in files:
        raw = pd.read_parquet(
            file
        )

        normalized = normalize_liquidations(
            raw
        )

        frames.append(
            normalized
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return normalize_liquidations(
        result
    )
