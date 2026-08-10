from __future__ import annotations

import pandas as pd

from cryptolab.features.derivatives_liquidation_coverage import (
    apply_liquidation_coverage,
)
from cryptolab.pipelines.derivatives_heartbeat import (
    build_derivatives_liquidation_coverage,
)
from cryptolab.pipelines.derivatives_quality import (
    build_derivatives_quality_dataset,
)


def build_covered_derivatives_quality_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
    minimum_liquidation_coverage_ratio: float = 0.80,
    heartbeat_interval_seconds: float = 30.0,
) -> pd.DataFrame:
    """
    Build derivatives quality dataset with explicit
    liquidation collector coverage.
    """

    quality = build_derivatives_quality_dataset(
        exchange=exchange,
        symbol=symbol,
        price_timeframe=price_timeframe,
        derivatives_period=derivatives_period,
    )

    coverage = build_derivatives_liquidation_coverage(
        exchange=exchange,
        symbol=symbol,
        minimum_coverage_ratio=(
            minimum_liquidation_coverage_ratio
        ),
        heartbeat_interval_seconds=(
            heartbeat_interval_seconds
        ),
    )

    return apply_liquidation_coverage(
        quality_df=quality,
        coverage_df=coverage,
    )
