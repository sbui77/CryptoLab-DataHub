from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.spot_perp_flow import (
    build_spot_perp_flow_state,
)
from cryptolab.pipelines.derivatives_features import (
    build_derivatives_feature_dataset,
)
from cryptolab.pipelines.trade_flow_storage import (
    read_trade_flow_features,
)


class SpotPerpFlowPipelineError(
    RuntimeError
):
    """Raised when Spot × Perp flow pipeline fails."""


def build_spot_perp_flow_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    spot_timeframe: str = "1m",
    derivatives_period: str = "5m",
    directional_threshold: float = 0.10,
    leadership_ratio: float = 1.50,
) -> pd.DataFrame:
    """
    Build Spot × Perp interaction dataset.

    Spot:
        curated Trade Flow Layer 1m.

    Perp:
        Derivatives Feature Layer 5m.
    """

    if spot_timeframe != "1m":
        raise SpotPerpFlowPipelineError(
            "Spot × Perp v1 currently requires "
            "spot_timeframe=1m"
        )

    if derivatives_period != "5m":
        raise SpotPerpFlowPipelineError(
            "Spot × Perp v1 currently requires "
            "derivatives_period=5m"
        )

    spot = read_trade_flow_features(
        exchange=exchange,
        symbol=symbol,
        timeframe=spot_timeframe,
    )

    if spot.empty:
        raise SpotPerpFlowPipelineError(
            "Spot Trade Flow dataset is empty"
        )

    derivatives = (
        build_derivatives_feature_dataset(
            exchange=exchange,
            symbol=symbol,
            period=derivatives_period,
        )
    )

    return build_spot_perp_flow_state(
        spot_flow=spot,
        derivatives_df=derivatives,
        directional_threshold=(
            directional_threshold
        ),
        leadership_ratio=(
            leadership_ratio
        ),
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


def save_spot_perp_flow(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    period: str,
) -> Path:
    """
    Save Spot × Perp interaction dataset.

    Final production partitioning is deferred to Step 8.12.
    """

    if df.empty:
        raise SpotPerpFlowPipelineError(
            "Cannot save empty Spot × Perp dataset"
        )

    directory = (
        _get_curated_root()
        / "derivatives"
        / "spot_perp_flow"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"period={period}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        directory
        / "spot_perp_flow.parquet"
    )

    (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .to_parquet(
            output,
            index=False,
            compression="snappy",
        )
    )

    return output


def build_and_save_spot_perp_flow(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    spot_timeframe: str = "1m",
    derivatives_period: str = "5m",
    directional_threshold: float = 0.10,
    leadership_ratio: float = 1.50,
) -> tuple[
    pd.DataFrame,
    Path,
]:
    df = build_spot_perp_flow_dataset(
        exchange=exchange,
        symbol=symbol,
        spot_timeframe=spot_timeframe,
        derivatives_period=derivatives_period,
        directional_threshold=(
            directional_threshold
        ),
        leadership_ratio=(
            leadership_ratio
        ),
    )

    output = save_spot_perp_flow(
        df,
        exchange=exchange,
        symbol=symbol,
        period=derivatives_period,
    )

    return (
        df,
        output,
    )
