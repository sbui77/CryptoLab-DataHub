from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.derivatives_quality import (
    add_derivatives_quality_masks,
)
from cryptolab.pipelines.derivatives_regime import (
    build_derivatives_regime_dataset,
)


class DerivativesQualityPipelineError(
    RuntimeError
):
    """Raised when derivatives quality pipeline fails."""


def build_derivatives_quality_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    funding_max_age_hours: float = 12.0,
    spot_perp_min_valid_share: float = 0.80,
) -> pd.DataFrame:
    """
    Build derivatives regime dataset with evidence-level
    validity masks.
    """

    regime = (
        build_derivatives_regime_dataset(
            exchange=exchange,
            symbol=symbol,
            price_timeframe=price_timeframe,
            derivatives_period=derivatives_period,
        )
    )

    return add_derivatives_quality_masks(
        regime,
        funding_max_age_hours=(
            funding_max_age_hours
        ),
        spot_perp_min_valid_share=(
            spot_perp_min_valid_share
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


def save_derivatives_quality(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Persist quality-enriched derivatives dataset.
    """

    if df.empty:
        raise DerivativesQualityPipelineError(
            "Cannot save empty derivatives quality dataset"
        )

    directory = (
        _get_curated_root()
        / "derivatives"
        / "quality"
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
        / "derivatives_quality.parquet"
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


def build_and_save_derivatives_quality(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    funding_max_age_hours: float = 12.0,
    spot_perp_min_valid_share: float = 0.80,
) -> tuple[
    pd.DataFrame,
    Path,
]:
    df = build_derivatives_quality_dataset(
        exchange=exchange,
        symbol=symbol,
        price_timeframe=price_timeframe,
        derivatives_period=derivatives_period,
        funding_max_age_hours=(
            funding_max_age_hours
        ),
        spot_perp_min_valid_share=(
            spot_perp_min_valid_share
        ),
    )

    output = save_derivatives_quality(
        df,
        exchange=exchange,
        symbol=symbol,
        timeframe=price_timeframe,
    )

    return (
        df,
        output,
    )
