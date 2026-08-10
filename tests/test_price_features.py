import numpy as np
import pandas as pd
from cryptolab.features.price import (
    add_range_features,
    add_returns,
    add_rolling_range_features,
    add_rolling_returns,
    add_volatility_features,
    add_vwap_features,
    build_price_features,
)
def make_df(
    rows: int = 10_000,
) -> pd.DataFrame:
    """
    Create deterministic hourly test data.
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
    base_price = (
        40_000.0
        + index * 0.5
        + np.sin(index / 20.0) * 100.0
    )
    open_price = base_price
    close_price = (
        base_price
        + np.sin(index / 5.0) * 20.0
    )
    high_price = (
        np.maximum(
            open_price,
            close_price,
        )
        + 50.0
    )
    low_price = (
        np.minimum(
            open_price,
            close_price,
        )
        - 50.0
    )
    base_volume = (
        10.0
        + np.sin(index / 10.0)
        + 2.0
    )
    typical_price = (
        high_price
        + low_price
        + close_price
    ) / 3.0
    quote_volume = (
        base_volume
        * typical_price
    )
    close_time = (
        open_time
        + pd.Timedelta(hours=1)
        - pd.Timedelta(milliseconds=1)
    )
    ingested_at = (
        open_time
        + pd.Timedelta(hours=2)
    )
    return pd.DataFrame(
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": open_time,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "base_volume": base_volume,
            "quote_volume": quote_volume,
            "close_time": close_time,
            "trade_count": np.full(
                rows,
                1000,
                dtype="int64",
            ),
            "taker_buy_base_volume": (
                base_volume * 0.55
            ),
            "taker_buy_quote_volume": (
                quote_volume * 0.55
            ),
            "ingested_at": ingested_at,
        }
    )
def test_returns():
    result = add_returns(
        make_df(
            rows=100
        )
    )
    assert (
        "simple_return"
        in result.columns
    )
    assert (
        "log_return"
        in result.columns
    )
def test_rolling_returns():
    result = add_rolling_returns(
        make_df(
            rows=100
        ),
        horizons=(
            2,
            24,
        ),
    )
    assert (
        "return_2"
        in result.columns
    )
    assert (
        "return_24h"
        in result.columns
    )
def test_range_features():
    result = add_range_features(
        make_df(
            rows=100
        ),
        atr_window=14,
    )
    expected_columns = [
        "high_low_range",
        "range_pct",
        "true_range",
        "atr_14",
        "atr_14_pct",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
def test_rolling_range_features():
    result = add_rolling_range_features(
        make_df(
            rows=100
        ),
        windows=(24,),
    )
    expected_columns = [
        "rolling_high_24",
        "rolling_low_24",
        "rolling_range_24",
        "range_position_24",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
    position = (
        result[
            "range_position_24"
        ]
        .dropna()
    )
    assert (
        position >= 0
    ).all()
    assert (
        position <= 1
    ).all()
def test_volatility_features():
    df = add_returns(
        make_df(
            rows=10_000
        )
    )
    result = add_volatility_features(
        df
    )
    expected_columns = [
        "rv_24h",
        "rv_7d",
        "rv_30d",
        "rv_90d",
        "rv_30d_percentile_1y",
        "rv_30d_zscore_1y",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
def test_vwap_features():
    result = add_vwap_features(
        make_df(
            rows=1000
        )
    )
    expected_columns = [
        "daily_vwap",
        "weekly_vwap",
        "monthly_vwap",
        "distance_to_daily_vwap",
        "distance_to_weekly_vwap",
        "distance_to_monthly_vwap",
        "distance_to_daily_vwap_pct",
        "distance_to_weekly_vwap_pct",
        "distance_to_monthly_vwap_pct",
    ]
    for column in (
        expected_columns
    ):
        assert (
            column
            in result.columns
        )
    assert (
        result["daily_vwap"]
        .dropna()
        > 0
    ).all()
    assert (
        result["weekly_vwap"]
        .dropna()
        > 0
    ).all()
    assert (
        result["monthly_vwap"]
        .dropna()
        > 0
    ).all()
def test_daily_vwap_resets():
    df = make_df(
        rows=48
    )
    result = add_vwap_features(
        df
    )
    first_day_vwap = (
        result.loc[
            0,
            "daily_vwap",
        ]
    )
    second_day_vwap = (
        result.loc[
            24,
            "daily_vwap",
        ]
    )
    expected_first = (
        result.loc[
            0,
            "quote_volume",
        ]
        / result.loc[
            0,
            "base_volume",
        ]
    )
    expected_second = (
        result.loc[
            24,
            "quote_volume",
        ]
        / result.loc[
            24,
            "base_volume",
        ]
    )
    assert np.isclose(
        first_day_vwap,
        expected_first,
    )
    assert np.isclose(
        second_day_vwap,
        expected_second,
    )
def test_build_price_features():
    result = build_price_features(
        make_df(
            rows=10_000
        )
    )
    required_features = [
        "simple_return",
        "log_return",
        "return_24h",
        "return_7d",
        "return_30d",
        "high_low_range",
        "range_pct",
        "true_range",
        "atr_14",
        "atr_14_pct",
        "rolling_high_24",
        "rolling_low_24",
        "rolling_range_24",
        "range_position_24",
        "rolling_high_168",
        "rolling_low_168",
        "rolling_range_168",
        "range_position_168",
        "rolling_high_720",
        "rolling_low_720",
        "rolling_range_720",
        "range_position_720",
        "rv_24h",
        "rv_7d",
        "rv_30d",
        "rv_90d",
        "rv_30d_percentile_1y",
        "rv_30d_zscore_1y",
        "daily_vwap",
        "weekly_vwap",
        "monthly_vwap",
        "distance_to_daily_vwap_pct",
        "distance_to_weekly_vwap_pct",
        "distance_to_monthly_vwap_pct",
    ]
    for column in (
        required_features
    ):
        assert (
            column
            in result.columns
        )