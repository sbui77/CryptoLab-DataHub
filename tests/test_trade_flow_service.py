from __future__ import annotations
import pandas as pd
from cryptolab.pipelines.trade_flow_service import (
    build_trade_flow_features,
)
def make_trades(
    rows: int = 200,
) -> pd.DataFrame:
    trade_time = pd.date_range(
        "2026-08-10T00:00:00Z",
        periods=rows,
        freq="5s",
    )
    price = [
        60000.0
        + float(index)
        for index in range(rows)
    ]
    quantity = [
        0.01
        + (
            index % 10
        )
        * 0.001
        for index in range(rows)
    ]
    quote_quantity = [
        p * q
        for p, q in zip(
            price,
            quantity,
            strict=True,
        )
    ]
    taker_side = [
        "buy"
        if index % 2 == 0
        else "sell"
        for index in range(rows)
    ]
    buyer_is_maker = [
        side == "sell"
        for side in taker_side
    ]
    return pd.DataFrame(
        {
            "exchange": [
                "binance"
            ] * rows,
            "symbol": [
                "BTCUSDT"
            ] * rows,
            "agg_trade_id": list(
                range(
                    1000,
                    1000 + rows,
                )
            ),
            "price": price,
            "quantity": quantity,
            "quote_quantity": (
                quote_quantity
            ),
            "first_trade_id": list(
                range(
                    2000,
                    2000 + rows,
                )
            ),
            "last_trade_id": list(
                range(
                    2000,
                    2000 + rows,
                )
            ),
            "trade_time": trade_time,
            "buyer_is_maker": (
                buyer_is_maker
            ),
            "taker_side": taker_side,
        }
    )
def test_build_trade_flow_features():
    result = build_trade_flow_features(
        make_trades(),
        timeframe="1m",
    )
    assert not result.empty
def test_required_production_columns():
    result = build_trade_flow_features(
        make_trades(),
        timeframe="1m",
    )
    expected = [
        "quote_delta_pct",
        "trade_count_imbalance",
        "base_cvd",
        "quote_cvd",
        "daily_base_cvd",
        "daily_quote_cvd",
        "rolling_base_cvd_60",
        "rolling_quote_cvd_60",
        "large_trade_quote_share",
        "large_quote_delta",
        "large_quote_delta_pct",
        "trade_flow_regime",
        "flow_regime_confidence",
        "trade_flow_regime_valid",
    ]
    for column in expected:
        assert column in result.columns
def test_flow_time_order():
    result = build_trade_flow_features(
        make_trades(),
        timeframe="1m",
    )
    assert (
        result["open_time"]
        .is_monotonic_increasing
    )
def test_no_duplicate_flow_bars():
    result = build_trade_flow_features(
        make_trades(),
        timeframe="1m",
    )
    assert (
        result["open_time"]
        .duplicated()
        .sum()
        == 0
    )
