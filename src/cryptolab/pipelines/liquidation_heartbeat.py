from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


class LiquidationHeartbeatError(RuntimeError):
    """Raised when liquidation heartbeat storage fails."""


HEARTBEAT_COLUMNS = [
    "exchange",
    "symbol",
    "heartbeat_time",
    "connected",
    "messages_received",
    "liquidation_rows",
    "reconnect_count",
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


def _heartbeat_root(
    exchange: str,
    symbol: str,
) -> Path:
    return (
        _get_raw_root()
        / "derivatives"
        / "liquidation_heartbeat"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
    )


def normalize_liquidation_heartbeat(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=HEARTBEAT_COLUMNS
        )

    missing = [
        column
        for column in HEARTBEAT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise LiquidationHeartbeatError(
            "Missing heartbeat columns: "
            f"{missing}"
        )

    result = df[
        HEARTBEAT_COLUMNS
    ].copy()

    result[
        "heartbeat_time"
    ] = pd.to_datetime(
        result[
            "heartbeat_time"
        ],
        utc=True,
        errors="coerce",
    )

    if (
        result[
            "heartbeat_time"
        ]
        .isna()
        .any()
    ):
        raise LiquidationHeartbeatError(
            "Invalid heartbeat_time"
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
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype("string")
        .str.upper()
    )

    result[
        "connected"
    ] = (
        result[
            "connected"
        ]
        .fillna(False)
        .astype(bool)
    )

    for column in [
        "messages_received",
        "liquidation_rows",
        "reconnect_count",
    ]:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="raise",
        ).astype("int64")

        if (
            result[
                column
            ]
            < 0
        ).any():
            raise LiquidationHeartbeatError(
                f"{column} cannot be negative"
            )

    return (
        result
        .sort_values(
            "heartbeat_time"
        )
        .drop_duplicates(
            subset=[
                "heartbeat_time",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


def save_liquidation_heartbeat(
    df: pd.DataFrame,
) -> list[Path]:
    if df.empty:
        return []

    work = normalize_liquidation_heartbeat(
        df
    )

    exchanges = (
        work[
            "exchange"
        ]
        .unique()
        .tolist()
    )

    symbols = (
        work[
            "symbol"
        ]
        .unique()
        .tolist()
    )

    if (
        len(exchanges) != 1
        or len(symbols) != 1
    ):
        raise LiquidationHeartbeatError(
            "Heartbeat save requires exactly "
            "one exchange/symbol"
        )

    exchange = str(
        exchanges[0]
    )

    symbol = str(
        symbols[0]
    )

    work["year"] = (
        work[
            "heartbeat_time"
        ].dt.year
    )

    work["month"] = (
        work[
            "heartbeat_time"
        ].dt.month
    )

    work["day"] = (
        work[
            "heartbeat_time"
        ].dt.day
    )

    root = _heartbeat_root(
        exchange=exchange,
        symbol=symbol,
    )

    saved: list[Path] = []

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

            existing = (
                normalize_liquidation_heartbeat(
                    existing
                )
            )

            partition = pd.concat(
                [
                    existing,
                    partition,
                ],
                ignore_index=True,
            )

        partition = (
            normalize_liquidation_heartbeat(
                partition
            )
        )

        partition.to_parquet(
            output,
            index=False,
            compression="snappy",
        )

        saved.append(
            output
        )

    return saved


def list_liquidation_heartbeat_files(
    exchange: str,
    symbol: str,
) -> list[Path]:
    root = _heartbeat_root(
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


def read_liquidation_heartbeat(
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    files = (
        list_liquidation_heartbeat_files(
            exchange=exchange,
            symbol=symbol,
        )
    )

    if not files:
        return pd.DataFrame(
            columns=HEARTBEAT_COLUMNS
        )

    result = pd.concat(
        [
            pd.read_parquet(
                file
            )
            for file in files
        ],
        ignore_index=True,
    )

    return normalize_liquidation_heartbeat(
        result
    )


def build_hourly_liquidation_coverage(
    heartbeat_df: pd.DataFrame,
    minimum_coverage_ratio: float = 0.80,
    heartbeat_interval_seconds: float = 30.0,
) -> pd.DataFrame:
    """
    Build hourly collector coverage.

    A 1h bucket is considered covered when the number of
    connected heartbeats reaches the configured fraction of
    expected heartbeats.

    Example
    -------
    heartbeat every 30s:
        expected = 120/hour

    minimum_coverage_ratio = 0.80:
        require >= 96 connected heartbeats.
    """

    if not (
        0.0
        <= minimum_coverage_ratio
        <= 1.0
    ):
        raise ValueError(
            "minimum_coverage_ratio must be "
            "between 0 and 1"
        )

    if heartbeat_interval_seconds <= 0:
        raise ValueError(
            "heartbeat_interval_seconds must be positive"
        )

    columns = [
        "open_time",
        "as_of_time",
        "heartbeat_count",
        "connected_heartbeat_count",
        "expected_heartbeat_count",
        "collector_coverage_ratio",
        "liquidation_collector_covered",
    ]

    if heartbeat_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    work = (
        normalize_liquidation_heartbeat(
            heartbeat_df
        )
    )

    work[
        "open_time"
    ] = (
        work[
            "heartbeat_time"
        ]
        .dt.floor("1h")
    )

    expected = max(
        1,
        int(
            3600
            / heartbeat_interval_seconds
        ),
    )

    result = (
        work
        .groupby(
            "open_time",
            as_index=False,
        )
        .agg(
            heartbeat_count=(
                "heartbeat_time",
                "size",
            ),
            connected_heartbeat_count=(
                "connected",
                "sum",
            ),
        )
    )

    result[
        "as_of_time"
    ] = (
        result[
            "open_time"
        ]
        + pd.Timedelta(hours=1)
    )

    result[
        "expected_heartbeat_count"
    ] = expected

    result[
        "collector_coverage_ratio"
    ] = (
        result[
            "connected_heartbeat_count"
        ]
        / expected
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    result[
        "liquidation_collector_covered"
    ] = (
        result[
            "collector_coverage_ratio"
        ]
        >= minimum_coverage_ratio
    )

    return result[
        columns
    ].sort_values(
        "open_time"
    ).reset_index(
        drop=True
    )
