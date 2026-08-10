from __future__ import annotations

import numpy as np
import pandas as pd


class LargeTradeFeatureError(ValueError):
    """Raised when large-trade feature generation fails."""


def add_large_trade_flags(
    df: pd.DataFrame,
    rolling_window: int = 10_000,
    large_quantile: float = 0.95,
    very_large_quantile: float = 0.99,
    min_periods: int = 1_000,
) -> pd.DataFrame:
    """
    Add rolling trade-size percentile and large-trade flags.

    The reference variable is quote_quantity because it
    represents trade notional and is more comparable across
    periods with different BTC prices.

    Parameters
    ----------
    rolling_window:
        Number of aggTrades in the rolling reference window.

    large_quantile:
        Threshold for large_trade.

    very_large_quantile:
        Threshold for very_large_trade.

    min_periods:
        Minimum observations required before percentile-based
        classification becomes valid.

    Output
    ------
    trade_notional
    large_trade_threshold
    very_large_trade_threshold

    large_trade
    very_large_trade

    large_buy
    large_sell

    very_large_buy
    very_large_sell
    """

    if df.empty:
        return df.copy()

    required_columns = [
        "agg_trade_id",
        "trade_time",
        "quote_quantity",
        "taker_side",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise LargeTradeFeatureError(
            f"Missing columns: {missing_columns}"
        )

    if rolling_window <= 0:
        raise LargeTradeFeatureError(
            "rolling_window must be greater than zero"
        )

    if min_periods <= 0:
        raise LargeTradeFeatureError(
            "min_periods must be greater than zero"
        )

    if min_periods > rolling_window:
        raise LargeTradeFeatureError(
            "min_periods cannot exceed rolling_window"
        )

    if not (
        0 < large_quantile < 1
    ):
        raise LargeTradeFeatureError(
            "large_quantile must be between 0 and 1"
        )

    if not (
        0 < very_large_quantile < 1
    ):
        raise LargeTradeFeatureError(
            "very_large_quantile must be between 0 and 1"
        )

    if (
        very_large_quantile
        <= large_quantile
    ):
        raise LargeTradeFeatureError(
            "very_large_quantile must exceed large_quantile"
        )

    result = (
        df.copy()
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(drop=True)
    )

    result["trade_notional"] = (
        result["quote_quantity"]
        .astype("float64")
    )

    # Shift one row before rolling quantile so the current
    # trade does not influence the threshold used to classify
    # itself. This keeps the feature causal/backtest-safe.
    historical_notional = (
        result["trade_notional"]
        .shift(1)
    )

    result[
        "large_trade_threshold"
    ] = (
        historical_notional
        .rolling(
            window=rolling_window,
            min_periods=min_periods,
        )
        .quantile(
            large_quantile
        )
    )

    result[
        "very_large_trade_threshold"
    ] = (
        historical_notional
        .rolling(
            window=rolling_window,
            min_periods=min_periods,
        )
        .quantile(
            very_large_quantile
        )
    )

    result["large_trade"] = (
        result[
            "large_trade_threshold"
        ].notna()
        & (
            result["trade_notional"]
            >= result[
                "large_trade_threshold"
            ]
        )
    )

    result["very_large_trade"] = (
        result[
            "very_large_trade_threshold"
        ].notna()
        & (
            result["trade_notional"]
            >= result[
                "very_large_trade_threshold"
            ]
        )
    )

    is_buy = (
        result["taker_side"]
        .eq("buy")
    )

    is_sell = (
        result["taker_side"]
        .eq("sell")
    )

    result["large_buy"] = (
        result["large_trade"]
        & is_buy
    )

    result["large_sell"] = (
        result["large_trade"]
        & is_sell
    )

    result["very_large_buy"] = (
        result["very_large_trade"]
        & is_buy
    )

    result["very_large_sell"] = (
        result["very_large_trade"]
        & is_sell
    )

    return result


def aggregate_large_trade_features(
    df: pd.DataFrame,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """
    Aggregate trade-level large-trade flags into time buckets.

    Requires add_large_trade_flags() output.

    Supported timeframe aliases:
        1m
        5m
        1h
    """

    if df.empty:
        return pd.DataFrame()

    rules = {
        "1m": "1min",
        "5m": "5min",
        "1h": "1h",
    }

    if timeframe not in rules:
        raise LargeTradeFeatureError(
            f"Unsupported timeframe: {timeframe}"
        )

    required_columns = [
        "trade_time",
        "quote_quantity",
        "large_trade",
        "large_buy",
        "large_sell",
        "very_large_trade",
        "very_large_buy",
        "very_large_sell",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise LargeTradeFeatureError(
            f"Missing columns: {missing_columns}"
        )

    work = (
        df.copy()
        .sort_values("trade_time")
        .reset_index(drop=True)
    )

    work["large_trade_count_i"] = (
        work["large_trade"]
        .astype("int64")
    )

    work["large_buy_count_i"] = (
        work["large_buy"]
        .astype("int64")
    )

    work["large_sell_count_i"] = (
        work["large_sell"]
        .astype("int64")
    )

    work["very_large_trade_count_i"] = (
        work["very_large_trade"]
        .astype("int64")
    )

    work["very_large_buy_count_i"] = (
        work["very_large_buy"]
        .astype("int64")
    )

    work["very_large_sell_count_i"] = (
        work["very_large_sell"]
        .astype("int64")
    )

    work["large_buy_quote_i"] = np.where(
        work["large_buy"],
        work["quote_quantity"],
        0.0,
    )

    work["large_sell_quote_i"] = np.where(
        work["large_sell"],
        work["quote_quantity"],
        0.0,
    )

    work["large_trade_quote_i"] = np.where(
        work["large_trade"],
        work["quote_quantity"],
        0.0,
    )

    work["very_large_buy_quote_i"] = np.where(
        work["very_large_buy"],
        work["quote_quantity"],
        0.0,
    )

    work["very_large_sell_quote_i"] = np.where(
        work["very_large_sell"],
        work["quote_quantity"],
        0.0,
    )

    work["very_large_trade_quote_i"] = np.where(
        work["very_large_trade"],
        work["quote_quantity"],
        0.0,
    )

    work = (
        work
        .set_index("trade_time")
    )

    result = (
        work
        .resample(
            rules[timeframe],
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            trade_count=(
                "quote_quantity",
                "count",
            ),

            quote_volume=(
                "quote_quantity",
                "sum",
            ),

            large_trade_count=(
                "large_trade_count_i",
                "sum",
            ),

            large_buy_count=(
                "large_buy_count_i",
                "sum",
            ),

            large_sell_count=(
                "large_sell_count_i",
                "sum",
            ),

            very_large_trade_count=(
                "very_large_trade_count_i",
                "sum",
            ),

            very_large_buy_count=(
                "very_large_buy_count_i",
                "sum",
            ),

            very_large_sell_count=(
                "very_large_sell_count_i",
                "sum",
            ),

            large_buy_quote=(
                "large_buy_quote_i",
                "sum",
            ),

            large_sell_quote=(
                "large_sell_quote_i",
                "sum",
            ),

            large_trade_quote=(
                "large_trade_quote_i",
                "sum",
            ),

            very_large_buy_quote=(
                "very_large_buy_quote_i",
                "sum",
            ),

            very_large_sell_quote=(
                "very_large_sell_quote_i",
                "sum",
            ),

            very_large_trade_quote=(
                "very_large_trade_quote_i",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "trade_time": "open_time",
            }
        )
    )

    result["large_quote_delta"] = (
        result["large_buy_quote"]
        - result["large_sell_quote"]
    )

    result[
        "very_large_quote_delta"
    ] = (
        result["very_large_buy_quote"]
        - result["very_large_sell_quote"]
    )

    result["large_trade_share"] = np.where(
        result["trade_count"] > 0,
        result["large_trade_count"]
        / result["trade_count"],
        np.nan,
    )

    result["very_large_trade_share"] = np.where(
        result["trade_count"] > 0,
        result["very_large_trade_count"]
        / result["trade_count"],
        np.nan,
    )

    result[
        "large_trade_quote_share"
    ] = np.where(
        result["quote_volume"] > 0,
        result["large_trade_quote"]
        / result["quote_volume"],
        np.nan,
    )

    result[
        "very_large_trade_quote_share"
    ] = np.where(
        result["quote_volume"] > 0,
        result[
            "very_large_trade_quote"
        ]
        / result["quote_volume"],
        np.nan,
    )

    return (
        result
        .sort_values("open_time")
        .reset_index(drop=True)
    )
