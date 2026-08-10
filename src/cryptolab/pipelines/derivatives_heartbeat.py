from __future__ import annotations

import pandas as pd

from cryptolab.pipelines.liquidation_heartbeat import (
    build_hourly_liquidation_coverage,
    read_liquidation_heartbeat,
)


def build_derivatives_liquidation_coverage(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    minimum_coverage_ratio: float = 0.80,
    heartbeat_interval_seconds: float = 30.0,
) -> pd.DataFrame:
    """
    Build hourly liquidation collector coverage aligned
    with the 1h derivatives-regime clock.
    """

    heartbeat = read_liquidation_heartbeat(
        exchange=exchange,
        symbol=symbol,
    )

    return build_hourly_liquidation_coverage(
        heartbeat_df=heartbeat,
        minimum_coverage_ratio=(
            minimum_coverage_ratio
        ),
        heartbeat_interval_seconds=(
            heartbeat_interval_seconds
        ),
    )
