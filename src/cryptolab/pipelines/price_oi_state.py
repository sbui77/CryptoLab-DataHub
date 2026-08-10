from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.price_oi_state import (
    build_price_oi_state,
)
from cryptolab.pipelines.derivatives_features import (
    build_derivatives_feature_dataset,
)
from cryptolab.pipelines.price_features import (
    build_price_feature_dataset,
)


class PriceOIStatePipelineError(
    RuntimeError
):
    """Raised when Price × OI pipeline fails."""


def build_price_oi_state_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    price_threshold: float = 0.001,
    oi_threshold: float = 0.002,
) -> pd.DataFrame:
    """
    Build Price × OI state dataset.

    Price:
        existing Price Layer 1h dataset

    OI:
        Derivatives Layer 5m dataset,
        resampled internally to 1h.
    """

    if price_timeframe != "1h":
        raise PriceOIStatePipelineError(
            "Price × OI v1 currently requires "
            "price_timeframe=1h"
        )

    if derivatives_period != "5m":
        raise PriceOIStatePipelineError(
            "Price × OI v1 currently requires "
            "derivatives_period=5m"
        )

    price = build_price_feature_dataset(
        exchange=exchange,
        symbol=symbol,
        timeframe=price_timeframe,
    )

    derivatives = (
        build_derivatives_feature_dataset(
            exchange=exchange,
            symbol=symbol,
            period=derivatives_period,
        )
    )

    return build_price_oi_state(
        price_df=price,
        derivatives_df=derivatives,
        price_threshold=price_threshold,
        oi_threshold=oi_threshold,
    )


def _get_curated_root() -> Path:
    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )


def save_price_oi_state(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Save Price × OI state dataset.

    Production partitioning will be finalized in Step 8.12.
    """

    if df.empty:
        raise PriceOIStatePipelineError(
            "Cannot save empty Price × OI dataset"
        )

    directory = (
        _get_curated_root()
        / "derivatives"
        / "price_oi_state"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"timeframe={timeframe}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        directory
        / "price_oi_state.parquet"
    )

    (
        df
        .sort_values("open_time")
        .reset_index(drop=True)
        .to_parquet(
            output,
            index=False,
            compression="snappy",
        )
    )

    return output


def build_and_save_price_oi_state(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    price_threshold: float = 0.001,
    oi_threshold: float = 0.002,
) -> tuple[
    pd.DataFrame,
    Path,
]:
    """
    Build and persist Price × OI states.
    """

    df = build_price_oi_state_dataset(
        exchange=exchange,
        symbol=symbol,
        price_timeframe=price_timeframe,
        derivatives_period=derivatives_period,
        price_threshold=price_threshold,
        oi_threshold=oi_threshold,
    )

    output = save_price_oi_state(
        df,
        exchange=exchange,
        symbol=symbol,
        timeframe=price_timeframe,
    )

    return (
        df,
        output,
    )
