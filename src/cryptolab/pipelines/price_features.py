from __future__ import annotations
from pathlib import Path
import pandas as pd
from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.market_structure import (
    add_market_structure_features,
)
from cryptolab.features.price import (
    build_price_features,
)
from cryptolab.features.quality_masks import (
    add_price_quality_masks,
)
from cryptolab.features.regime import (
    add_price_regime_features,
)
from cryptolab.features.structural_levels import (
    add_structural_level_features,
)
from cryptolab.features.structural_state import (
    add_structural_state_features,
)
from cryptolab.pipelines.ohlcv_storage import (
    read_ohlcv,
)
def save_price_features(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    """
    Save curated Price Layer features to monthly
    Parquet partitions.
    Existing monthly files are overwritten so schema
    changes propagate consistently through history.
    """
    if df.empty:
        return []
    config = load_config()
    curated_root = resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )
    work = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    work["year"] = (
        work["open_time"]
        .dt.year
    )
    work["month"] = (
        work["open_time"]
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
            curated_root
            / "price"
            / "features"
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
            .sort_values(
                "open_time"
            )
            .reset_index(
                drop=True
            )
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
def build_price_feature_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Build the complete curated Price Layer.
    Processing order:
        Raw OHLCV
            ↓
        Price Features
            ↓
        Data Quality Masks
            ↓
        Confirmed Market Structure
            ↓
        BOS / CHOCH / Structural State
            ↓
        Structural Levels / Range State
            ↓
        Price Regime Foundation
            ↓
        Final Curated Dataset
    """
    if timeframe != "1h":
        raise ValueError(
            "Price feature pipeline currently supports "
            "canonical 1h input only"
        )
    source = read_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    if source.empty:
        raise RuntimeError(
            "Source OHLCV dataset is empty"
        )
    source = (
        source
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    features = build_price_features(
        source
    )
    features = add_price_quality_masks(
        features,
        expected_interval_hours=1,
    )
    features = add_market_structure_features(
        features,
        left_bars=3,
        right_bars=3,
        expected_interval_hours=1,
    )
    features = add_structural_state_features(
        features
    )
    features = add_structural_level_features(
        features
    )
    features = add_price_regime_features(
        features
    )
    return features
def build_and_save_price_features(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Build and persist the complete curated Price Layer.
    """
    features = build_price_feature_dataset(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    save_price_features(
        features,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    return features