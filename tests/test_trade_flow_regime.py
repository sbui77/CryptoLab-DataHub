from __future__ import annotations

import pandas as pd

from cryptolab.features.trade_flow_regime import (
    add_trade_flow_regime_features,
)


def make_flow_df() -> pd.DataFrame:
    rows = 70

    df = pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="1min",
            ),

            "quote_volume": [
                1000.0
            ] * rows,

            "quote_delta": [
                200.0
            ] * rows,

            "quote_delta_pct": [
                0.20
            ] * rows,

            "trade_count_imbalance": [
                0.20
            ] * rows,
        }
    )

    return df


def make_large_flow_df() -> pd.DataFrame:
    rows = 70

    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="1min",
            ),

            "large_trade_quote": [
                300.0
            ] * rows,

            "large_quote_delta": [
                150.0
            ] * rows,

            "large_trade_quote_share": [
                0.30
            ] * rows,
        }
    )


def test_regime_columns():
    result = add_trade_flow_regime_features(
        make_flow_df(),
        make_large_flow_df(),
        rolling_window=10,
    )

    expected = [
        "rolling_quote_delta_60",
        "rolling_quote_volume_60",
        "rolling_quote_imbalance_60",
        "large_quote_delta_pct",
        "trade_flow_regime",
        "flow_regime_confidence",
        "trade_flow_regime_valid",
    ]

    for column in expected:
        assert column in result.columns


def test_large_buy_pressure():
    result = add_trade_flow_regime_features(
        make_flow_df(),
        make_large_flow_df(),
        rolling_window=10,
    )

    valid = result.iloc[-1]

    assert (
        valid["trade_flow_regime"]
        == "large_buy_pressure"
    )


def test_buy_dominant_without_large_flow():
    result = add_trade_flow_regime_features(
        make_flow_df(),
        large_flow_df=None,
        rolling_window=10,
    )

    assert (
        result.iloc[-1][
            "trade_flow_regime"
        ]
        == "buy_dominant"
    )


def test_sell_dominant():
    df = make_flow_df()

    df["quote_delta"] = -200.0
    df["quote_delta_pct"] = -0.20
    df[
        "trade_count_imbalance"
    ] = -0.20

    result = add_trade_flow_regime_features(
        df,
        large_flow_df=None,
        rolling_window=10,
    )

    assert (
        result.iloc[-1][
            "trade_flow_regime"
        ]
        == "sell_dominant"
    )


def test_balanced():
    df = make_flow_df()

    df["quote_delta"] = 10.0
    df["quote_delta_pct"] = 0.01
    df[
        "trade_count_imbalance"
    ] = 0.01

    result = add_trade_flow_regime_features(
        df,
        large_flow_df=None,
        rolling_window=10,
    )

    assert (
        result.iloc[-1][
            "trade_flow_regime"
        ]
        == "balanced"
    )


def test_confidence_range():
    result = add_trade_flow_regime_features(
        make_flow_df(),
        make_large_flow_df(),
        rolling_window=10,
    )

    assert (
        result[
            "flow_regime_confidence"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )
