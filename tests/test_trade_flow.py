from __future__ import annotations
import numpy as np
import pandas as pd
from cryptolab.features.trade_flow import (
    add_cvd_features,
    aggregate_aggtrades,
)
def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
                "binance",
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
                "BTCUSDT",
                "BTCUSDT",
            ],
            "agg_trade_id": [
                100,
                101,
                102,
                103,
            ],
            "price": [
                100.0,
                110.0,
                105.0,
                120.0,
            ],
            "quantity": [
                1.0,
                2.0,
                1.5,
                0.5,
            ],
            "quote_quantity": [
                100.0,
                220.0,
                157.5,
                60.0,
            ],
            "trade_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:10Z",
                    "2026-01-01T00:00:20Z",
                    "2026-01-01T00:00:30Z",
                    "2026-01-01T00:01:10Z",
                ],
                utc=True,
            ),
            "taker_side": [
                "buy",
                "sell",
                "buy",
                "sell",
            ],
        }
    )
def make_cvd_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-01-01T23:58:00Z",
                    "2026-01-01T23:59:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T00:01:00Z",
                ],
                utc=True,
            ),
            "base_delta": [
                1.0,
                -0.5,
                2.0,
                -1.0,
            ],
            "quote_delta": [
                100.0,
                -50.0,
                200.0,
                -100.0,
            ],
        }
    )
def test_aggregate_1m_rows():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    assert len(result) == 2
def test_ohlc():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    first = result.iloc[0]
    assert first["open"] == 100.0
    assert first["high"] == 110.0
    assert first["low"] == 100.0
    assert first["close"] == 105.0
def test_trade_counts():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    first = result.iloc[0]
    assert first["trade_count"] == 3
    assert first["buy_trade_count"] == 2
    assert first["sell_trade_count"] == 1
def test_volume_identity():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    assert np.allclose(
        result["base_volume"],
        result["buy_base_volume"]
        + result["sell_base_volume"],
    )
    assert np.allclose(
        result["quote_volume"],
        result["buy_quote_volume"]
        + result["sell_quote_volume"],
    )
def test_delta():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    first = result.iloc[0]
    assert np.isclose(
        first["base_delta"],
        0.5,
    )
    expected_quote_delta = (
        100.0
        + 157.5
        - 220.0
    )
    assert np.isclose(
        first["quote_delta"],
        expected_quote_delta,
    )
def test_vwap():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    first = result.iloc[0]
    expected = (
        100.0
        + 220.0
        + 157.5
    ) / (
        1.0
        + 2.0
        + 1.5
    )
    assert np.isclose(
        first["vwap"],
        expected,
    )
def test_buy_sell_ratios_sum_to_one():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    assert np.allclose(
        result["buy_base_ratio"]
        + result["sell_base_ratio"],
        1.0,
    )
    assert np.allclose(
        result["buy_quote_ratio"]
        + result["sell_quote_ratio"],
        1.0,
    )
    assert np.allclose(
        result["buy_trade_ratio"]
        + result["sell_trade_ratio"],
        1.0,
    )
def test_delta_pct_range():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    assert (
        result["base_delta_pct"]
        .between(
            -1.0,
            1.0,
        )
        .all()
    )
    assert (
        result["quote_delta_pct"]
        .between(
            -1.0,
            1.0,
        )
        .all()
    )
    assert (
        result["trade_count_imbalance"]
        .between(
            -1.0,
            1.0,
        )
        .all()
    )
def test_average_trade_sizes():
    result = aggregate_aggtrades(
        make_df(),
        timeframe="1m",
    )
    first = result.iloc[0]
    assert np.isclose(
        first["avg_buy_trade_base"],
        2.5 / 2,
    )
    assert np.isclose(
        first["avg_sell_trade_base"],
        2.0,
    )
    assert np.isclose(
        first["avg_buy_trade_quote"],
        (
            100.0
            + 157.5
        )
        / 2,
    )
    assert np.isclose(
        first["avg_sell_trade_quote"],
        220.0,
    )
def test_global_cvd():
    result = add_cvd_features(
        make_cvd_df()
    )
    assert np.allclose(
        result["base_cvd"],
        [
            1.0,
            0.5,
            2.5,
            1.5,
        ],
    )
    assert np.allclose(
        result["quote_cvd"],
        [
            100.0,
            50.0,
            250.0,
            150.0,
        ],
    )
def test_daily_cvd_resets():
    result = add_cvd_features(
        make_cvd_df()
    )
    assert np.allclose(
        result["daily_base_cvd"],
        [
            1.0,
            0.5,
            2.0,
            1.0,
        ],
    )
    assert np.allclose(
        result["daily_quote_cvd"],
        [
            100.0,
            50.0,
            200.0,
            100.0,
        ],
    )
def test_rolling_cvd_columns():
    result = add_cvd_features(
        make_cvd_df()
    )
    expected = [
        "rolling_base_cvd_60",
        "rolling_quote_cvd_60",
        "rolling_base_cvd_1440",
        "rolling_quote_cvd_1440",
    ]
    for column in expected:
        assert column in result.columns
def test_cvd_terminal_identity():
    df = make_cvd_df()
    result = add_cvd_features(
        df
    )
    assert np.isclose(
        result["base_cvd"].iloc[-1],
        df["base_delta"].sum(),
    )
    assert np.isclose(
        result["quote_cvd"].iloc[-1],
        df["quote_delta"].sum(),
    )