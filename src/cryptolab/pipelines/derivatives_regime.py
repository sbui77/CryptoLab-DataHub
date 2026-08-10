from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.derivatives_regime import (
    build_derivatives_regime,
)
from cryptolab.pipelines.derivatives_features import (
    build_derivatives_feature_dataset,
)
from cryptolab.pipelines.price_oi_state import (
    build_price_oi_state_dataset,
)
from cryptolab.pipelines.spot_perp_flow import (
    build_spot_perp_flow_dataset,
)


class DerivativesRegimePipelineError(
    RuntimeError
):
    """Raised when derivatives regime pipeline fails."""


def build_derivatives_regime_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    flow_threshold: float = 0.10,
    liquidation_imbalance_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Build complete 1h derivatives regime dataset.
    """

    price_oi = (
        build_price_oi_state_dataset(
            exchange=exchange,
            symbol=symbol,
            price_timeframe=price_timeframe,
            derivatives_period=derivatives_period,
        )
    )

    spot_perp = (
        build_spot_perp_flow_dataset(
            exchange=exchange,
            symbol=symbol,
            spot_timeframe="1m",
            derivatives_period=derivatives_period,
        )
    )

    derivatives = (
        build_derivatives_feature_dataset(
            exchange=exchange,
            symbol=symbol,
            period=derivatives_period,
        )
    )

    return build_derivatives_regime(
        price_oi_df=price_oi,
        spot_perp_df=spot_perp,
        derivatives_df=derivatives,
        flow_threshold=flow_threshold,
        liquidation_imbalance_threshold=(
            liquidation_imbalance_threshold
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


def save_derivatives_regime(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Save current derivatives regime dataset.

    Final production partitioning is handled in Step 8.12.
    """

    if df.empty:
        raise DerivativesRegimePipelineError(
            "Cannot save empty derivatives regime"
        )

    directory = (
        _get_curated_root()
        / "derivatives"
        / "regime"
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
        / "derivatives_regime.parquet"
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


def build_and_save_derivatives_regime(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    flow_threshold: float = 0.10,
    liquidation_imbalance_threshold: float = 0.50,
) -> tuple[
    pd.DataFrame,
    Path,
]:
    df = build_derivatives_regime_dataset(
        exchange=exchange,
        symbol=symbol,
        price_timeframe=price_timeframe,
        derivatives_period=derivatives_period,
        flow_threshold=flow_threshold,
        liquidation_imbalance_threshold=(
            liquidation_imbalance_threshold
        ),
    )

    output = save_derivatives_regime(
        df,
        exchange=exchange,
        symbol=symbol,
        timeframe=price_timeframe,
    )

    return (
        df,
        output,
    )
