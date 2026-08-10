from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.large_trades import (
    add_large_trade_flags,
    aggregate_large_trade_features,
)


def make_df() -> pd.DataFrame:
    rows = 20

    quote_quantity = [
        float(value)
        for value in range(
            100,
            2100,
            100,
        )
    ]

    return pd.DataFrame(
        {
            "agg_trade_id": list(
                range(
                    1,
                    rows + 1,
                )
            ),

            "trade_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="10s",
            ),

            "quote_quantity": (
                quote_quantity
            ),

            "taker_side": [
                "buy"
                if index % 2 == 0
                else "sell"
                for index in range(rows)
            ],
        }
    )


def test_large_trade_columns():
    result = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    expected = [
        "trade_notional",
        "large_trade_threshold",
        "very_large_trade_threshold",
        "large_trade",
        "very_large_trade",
        "large_buy",
        "large_sell",
        "very_large_buy",
        "very_large_sell",
    ]

    for column in expected:
        assert column in result.columns


def test_threshold_is_causal():
    df = make_df()

    result = add_large_trade_flags(
        df,
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    # At index 5 the threshold must be derived from
    # rows 0..4 only, because current trade is shifted out.
    expected = np.quantile(
        df.loc[
            0:4,
            "quote_quantity",
        ],
        0.80,
    )

    assert np.isclose(
        result.loc[
            5,
            "large_trade_threshold",
        ],
        expected,
    )


def test_large_trade_flags_exist():
    result = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    assert (
        result["large_trade"]
        .sum()
        > 0
    )

    assert (
        result["very_large_trade"]
        .sum()
        > 0
    )


def test_large_side_mapping():
    result = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    assert (
        (
            result["large_buy"]
            & result["large_sell"]
        )
        .sum()
        == 0
    )


def test_aggregate_large_trades():
    flagged = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    result = aggregate_large_trade_features(
        flagged,
        timeframe="1m",
    )

    assert not result.empty

    expected = [
        "large_trade_count",
        "large_buy_count",
        "large_sell_count",
        "large_buy_quote",
        "large_sell_quote",
        "large_quote_delta",
        "large_trade_share",
        "large_trade_quote_share",
    ]

    for column in expected:
        assert column in result.columns


def test_large_count_identity():
    flagged = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    result = aggregate_large_trade_features(
        flagged,
        timeframe="1m",
    )

    assert (
        result["large_trade_count"]
        == (
            result["large_buy_count"]
            + result["large_sell_count"]
        )
    ).all()


def test_large_quote_delta_identity():
    flagged = add_large_trade_flags(
        make_df(),
        rolling_window=10,
        min_periods=5,
        large_quantile=0.80,
        very_large_quantile=0.95,
    )

    result = aggregate_large_trade_features(
        flagged,
        timeframe="1m",
    )

    assert np.allclose(
        result["large_quote_delta"],
        result["large_buy_quote"]
        - result["large_sell_quote"],
    )
