from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.quality.ohlcv import (
    validate_ohlcv,
)


def save_curated_ohlcv(
    df: pd.DataFrame,
) -> list[Path]:
    """
    Save resampled OHLCV to curated storage.
    """

    if df.empty:
        return []

    validate_ohlcv(df)

    config = load_config()

    curated_root = resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )

    work = df.copy()

    work["year"] = (
        work["open_time"].dt.year
    )

    work["month"] = (
        work["open_time"].dt.month
    )

    saved_files: list[Path] = []

    grouping = [
        "exchange",
        "symbol",
        "timeframe",
        "year",
        "month",
    ]

    for keys, partition in work.groupby(
        grouping,
        sort=True,
    ):
        (
            exchange,
            symbol,
            timeframe,
            year,
            month,
        ) = keys

        directory = (
            curated_root
            / "price"
            / "ohlcv_resampled"
            / f"exchange={exchange}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year={year}"
            / f"month={month:02d}"
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
            .sort_values("open_time")
        )

        if output.exists():
            existing = pd.read_parquet(
                output
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
                        "exchange",
                        "symbol",
                        "timeframe",
                        "open_time",
                    ],
                    keep="last",
                )
                .sort_values("open_time")
                .reset_index(drop=True)
            )

        validate_ohlcv(
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
