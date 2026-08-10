from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


def get_ohlcv_root() -> Path:
    config = load_config()

    raw_root = resolve_project_path(
        get_config_value(
            config,
            "paths.raw",
            "data/raw",
        )
    )

    return raw_root / "price" / "ohlcv"


def get_dataset_path(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    return (
        get_ohlcv_root()
        / f"exchange={exchange}"
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
    )


def list_ohlcv_files(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    root = get_dataset_path(
        exchange,
        symbol,
        timeframe,
    )

    if not root.exists():
        return []

    return sorted(
        root.glob(
            "year=*/month=*/data.parquet"
        )
    )


def read_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    files = list_ohlcv_files(
        exchange,
        symbol,
        timeframe,
    )

    if not files:
        return pd.DataFrame()

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        df
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


def get_latest_open_time(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.Timestamp | None:
    files = list_ohlcv_files(
        exchange,
        symbol,
        timeframe,
    )

    if not files:
        return None

    latest_file = files[-1]

    df = pd.read_parquet(
        latest_file,
        columns=["open_time"],
    )

    if df.empty:
        return None

    return df["open_time"].max()
