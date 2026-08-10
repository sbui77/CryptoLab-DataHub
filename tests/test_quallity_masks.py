import numpy as np
import pandas as pd
from cryptolab.features.price import (
    build_price_features,
)
from cryptolab.features.quality_masks import (
    add_price_quality_masks,
)
def make_continuous_df(
    rows: int = 10_000,
) -> pd.DataFrame:
    """
    Create continuous hourly OHLCV test data.
    """
    open_time = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=rows,
        freq="1h",
    )
    index = np.arange(
        rows,
        dtype="float64",
    )
    close = (
        40_000.0
        + index
        + np.sin(
            index / 20.0
        ) * 100.0
    )
    open_price = close - 10.0
    high = close + 50.0
    low = open_price - 50.0
    base_volume = (
        10.0
        + np.sin(
            index / 10.0
        )
        + 2.0
    )
    quote_volume = (
        base_volume
        * close
    )
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": open_time,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "base_volume": base_volume,
            "quote_volume": quote_volume,
            "close_time": (
                open_time
                + pd.Timedelta(
                    hours=1
                )
                - pd.Timedelta(
                    milliseconds=1
                )
            ),
            "trade_count": np.full(
                rows,
                1000,
                dtype="int64",
            ),
            "taker_buy_base_volume": (
                base_volume
                * 0.55
            ),
            "taker_buy_quote_volume": (
                quote_volume
                * 0.55
            ),
            "ingested_at": (
                open_time
                + pd.Timedelta(
                    hours=2
                )
            ),
        }
    )
def make_gap_df() -> pd.DataFrame:
    """
    Create hourly data with one missing candle.
    Original:
        00
        01
        02
        03
        04
        ...
    Remove:
        hour 50
    Therefore the candle after the gap should show
    a two-hour interval.
    """
    df = make_continuous_df(
        rows=200
    )
    df = (
        df
        .drop(
            index=50
        )
        .reset_index(
            drop=True
        )
    )
    return df
def test_continuous_source():
    df = (
        build_price_features(
            make_continuous_df(
                rows=1000
            )
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    assert (
        result[
            "source_gap_before"
        ]
        .sum()
        == 0
    )
    assert (
        result[
            "missing_hours_before"
        ]
        .sum()
        == 0
    )
def test_gap_detection():
    df = (
        build_price_features(
            make_gap_df()
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    gaps = result[
        result[
            "source_gap_before"
        ]
    ]
    assert (
        len(gaps)
        == 1
    )
    gap_row = (
        gaps.iloc[0]
    )
    assert np.isclose(
        gap_row[
            "source_interval_hours"
        ],
        2.0,
    )
    assert (
        gap_row[
            "missing_hours_before"
        ]
        == 1
    )
def test_quality_24h_affected_by_gap():
    df = (
        build_price_features(
            make_gap_df()
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    gap_index = (
        result.index[
            result[
                "source_gap_before"
            ]
        ][0]
    )
    assert not bool(
        result.loc[
            gap_index,
            "quality_24h",
        ]
    )
    later_index = (
        gap_index + 30
    )
    assert bool(
        result.loc[
            later_index,
            "quality_24h",
        ]
    )
def test_return_quality_mask():
    df = (
        build_price_features(
            make_gap_df()
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    assert (
        "return_24h_valid"
        in result.columns
    )
    gap_index = (
        result.index[
            result[
                "source_gap_before"
            ]
        ][0]
    )
    assert not bool(
        result.loc[
            gap_index,
            "return_24h_valid",
        ]
    )
def test_rv_quality_masks_exist():
    df = (
        build_price_features(
            make_continuous_df(
                rows=3000
            )
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    expected_columns = [
        "rv_24h_valid",
        "rv_7d_valid",
        "rv_30d_valid",
        "rv_90d_valid",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
def test_vwap_quality_masks_exist():
    df = (
        build_price_features(
            make_continuous_df(
                rows=1000
            )
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    expected_columns = [
        "daily_vwap_valid",
        "weekly_vwap_valid",
        "monthly_vwap_valid",
        "distance_to_daily_vwap_pct_valid",
        "distance_to_weekly_vwap_pct_valid",
        "distance_to_monthly_vwap_pct_valid",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
def test_quality_horizon_columns():
    df = (
        build_price_features(
            make_continuous_df(
                rows=3000
            )
        )
    )
    result = (
        add_price_quality_masks(
            df
        )
    )
    expected_columns = [
        "gap_count_24h",
        "elapsed_hours_24h",
        "quality_24h",
        "gap_count_7d",
        "elapsed_hours_7d",
        "quality_7d",
        "gap_count_30d",
        "elapsed_hours_30d",
        "quality_30d",
        "gap_count_90d",
        "elapsed_hours_90d",
        "quality_90d",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
