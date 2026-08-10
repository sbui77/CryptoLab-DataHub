import pandas as pd
from cryptolab.features.regime import (
    add_price_regime_features,
)
def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=7,
                freq="1h",
            ),
            "close": [
                100.0,
                110.0,
                120.0,
                90.0,
                100.0,
                100.0,
                100.0,
            ],
            "structure_state": [
                "neutral",
                "bullish",
                "bullish",
                "bearish",
                "bullish",
                "bearish",
                "neutral",
            ],
            "range_state": [
                "incomplete",
                "inside_range",
                "above_resistance",
                "below_support",
                "inside_range",
                "inside_range",
                "inside_range",
            ],
            "rv_30d_percentile_1y": [
                None,
                0.50,
                0.90,
                0.90,
                0.30,
                0.90,
                0.50,
            ],
            "range_position_720": [
                None,
                0.80,
                0.90,
                0.10,
                0.50,
                0.50,
                0.50,
            ],
            "structural_range_position": [
                None,
                0.80,
                1.10,
                -0.10,
                0.50,
                0.50,
                0.50,
            ],
        }
    )
def test_regime_columns():
    result = add_price_regime_features(
        make_df()
    )
    expected = [
        "volatility_regime",
        "location_regime",
        "price_regime",
        "regime_confidence",
        "price_regime_valid",
    ]
    for column in expected:
        assert column in result.columns
def test_bullish_expansion():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result.loc[
            2,
            "price_regime",
        ]
        == "bullish_expansion"
    )
def test_bearish_expansion():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result.loc[
            3,
            "price_regime",
        ]
        == "bearish_expansion"
    )
def test_trending_up():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result.loc[
            1,
            "price_regime",
        ]
        == "trending_up"
    )
def test_range():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result.loc[
            4,
            "price_regime",
        ]
        == "range"
    )
def test_volatile_range():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result.loc[
            5,
            "price_regime",
        ]
        == "volatile_range"
    )
def test_confidence_range():
    result = add_price_regime_features(
        make_df()
    )
    assert (
        result[
            "regime_confidence"
        ]
        >= 0
    ).all()
    assert (
        result[
            "regime_confidence"
        ]
        <= 1
    ).all()
