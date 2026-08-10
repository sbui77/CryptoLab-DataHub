from __future__ import annotations
from pathlib import Path
import pandas as pd
from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.snapshot import (
    PriceSnapshot,
    build_price_snapshot,
    price_snapshot_to_dataframe,
)
def read_price_features(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Read the complete curated Price Layer.
    """
    config = load_config()
    curated_root = resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )
    root = (
        curated_root
        / "price"
        / "features"
        / f"exchange={exchange}"
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
    )
    files = sorted(
        root.glob(
            "year=*/month=*/data.parquet"
        )
    )
    if not files:
        raise RuntimeError(
            "No curated price feature files found"
        )
    frames = [
        pd.read_parquet(file)
        for file in files
    ]
    result = pd.concat(
        frames,
        ignore_index=True,
    )
    return (
        result
        .sort_values("open_time")
        .reset_index(drop=True)
    )
def save_price_snapshot(
    snapshot: PriceSnapshot,
) -> Path:
    """
    Save latest Price Layer snapshot.
    """
    config = load_config()
    curated_root = resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )
    directory = (
        curated_root
        / "price"
        / "snapshot"
        / f"exchange={snapshot.exchange}"
        / f"symbol={snapshot.symbol}"
        / f"timeframe={snapshot.timeframe}"
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = (
        directory
        / "latest.parquet"
    )
    df = price_snapshot_to_dataframe(
        snapshot
    )
    df.to_parquet(
        output,
        index=False,
        compression="snappy",
    )
    return output
def build_and_save_price_snapshot(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
) -> PriceSnapshot:
    """
    Read curated Price Layer, build latest snapshot,
    and persist it.
    """
    features = read_price_features(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    snapshot = build_price_snapshot(
        features
    )
    save_price_snapshot(
        snapshot
    )
    return snapshot
