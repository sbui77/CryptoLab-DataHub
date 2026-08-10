import numpy as np
import pandas as pd
from cryptolab.features.snapshot import (
    build_price_snapshot,
    price_snapshot_to_dataframe,
)
def make_df() -> pd.DataFrame:
    rows = 5
    return pd.DataFrame(
        {
            "exchange": [
                "binance"
            ] * rows,
            "symbol": [
                "BTCUSDT"
            ] * rows,
            "timeframe": [
                "1h"
            ] * rows,
            "open_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="1h",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
            "simple_return": [
                np.nan,
                0.01,
                0.01,
                0.01,
                0.01,
            ],
            "return_24h": [
                np.nan
            ] * 4
            + [
                0.04
            ],
            "return_7d": [
                0.10
            ] * rows,
            "return_30d": [
                0.20
            ] * rows,
            "atr_14_pct": [
                0.02
            ] * rows,
            "rv_24h": [
                0.40
            ] * rows,
            "rv_7d": [
                0.45
            ] * rows,
            "rv_30d": [
                0.50
            ] * rows,
            "rv_90d": [
                0.55
            ] * rows,
            "rv_30d_percentile_1y": [
                0.70
            ] * rows,
            "rv_30d_zscore_1y": [
                0.80
            ] * rows,
            "daily_vwap": [
                101.0
            ] * rows,
            "weekly_vwap": [
                100.0
            ] * rows,
            "monthly_vwap": [
                99.0
            ] * rows,
            "distance_to_daily_vwap_pct": [
                0.01
            ] * rows,
            "distance_to_weekly_vwap_pct": [
                0.02
            ] * rows,
            "distance_to_monthly_vwap_pct": [
                0.03
            ] * rows,
            "structure_state": [
                "bullish"
            ] * rows,
            "last_swing_high": [
                110.0
            ] * rows,
            "last_swing_low": [
                95.0
            ] * rows,
            "active_resistance": [
                110.0
            ] * rows,
            "active_support": [
                95.0
            ] * rows,
            "structural_range": [
                15.0
            ] * rows,
            "structural_range_pct": [
                15.0 / 95.0
            ] * rows,
            "structural_range_position": [
                0.60
            ] * rows,
            "range_state": [
                "inside_range"
            ] * rows,
            "nearest_structure_side": [
                "resistance"
            ] * rows,
            "nearest_structure_distance_pct": [
                0.05
            ] * rows,
            "price_regime": [
                "trending_up"
            ] * rows,
            "volatility_regime": [
                "high"
            ] * rows,
            "location_regime": [
                "upper_range"
            ] * rows,
            "regime_confidence": [
                1.0
            ] * rows,
            "price_regime_valid": [
                True
            ] * rows,
            "quality_24h": [
                True
            ] * rows,
            "quality_7d": [
                True
            ] * rows,
            "quality_30d": [
                True
            ] * rows,
            "quality_90d": [
                True
            ] * rows,
            "bos": [
                False,
                True,
                False,
                False,
                False,
            ],
            "bos_direction": [
                pd.NA,
                "bullish",
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "choch": [
                False,
                False,
                False,
                True,
                False,
            ],
            "choch_direction": [
                pd.NA,
                pd.NA,
                pd.NA,
                "bullish",
                pd.NA,
            ],
        }
    )
def test_snapshot_builds():
    snapshot = build_price_snapshot(
        make_df()
    )
    assert (
        snapshot.symbol
        == "BTCUSDT"
    )
    assert (
        snapshot.close
        == 104.0
    )
def test_latest_bos():
    snapshot = build_price_snapshot(
        make_df()
    )
    assert (
        snapshot.latest_bos_direction
        == "bullish"
    )
    assert (
        snapshot.latest_bos_time
        == pd.Timestamp(
            "2026-01-01T01:00:00Z"
        )
    )
def test_latest_choch():
    snapshot = build_price_snapshot(
        make_df()
    )
    assert (
        snapshot.latest_choch_direction
        == "bullish"
    )
    assert (
        snapshot.latest_choch_time
        == pd.Timestamp(
            "2026-01-01T03:00:00Z"
        )
    )
def test_snapshot_dataframe():
    snapshot = build_price_snapshot(
        make_df()
    )
    result = (
        price_snapshot_to_dataframe(
            snapshot
        )
    )
    assert len(result) == 1
    assert (
        "price_regime"
        in result.columns
    )
    assert (
        result.loc[
            0,
            "price_regime",
        ]
        == "trending_up"
    )
