from __future__ import annotations
import numpy as np
import pandas as pd
class TradeFlowFeatureError(ValueError):
    """Raised when trade-flow feature generation fails."""
SUPPORTED_BUCKETS = {
    "1m": "1min",
    "5m": "5min",
    "1h": "1h",
}
def aggregate_aggtrades(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Aggregate canonical aggTrades into time buckets.
    Supported:
        1m
        5m
        1h
    Output includes:
        OHLC
        trade counts
        aggressor-side volume
        delta
        normalized imbalance
        average trade size
        VWAP
    """
    if df.empty:
        return pd.DataFrame()
    if timeframe not in SUPPORTED_BUCKETS:
        raise TradeFlowFeatureError(
            f"Unsupported timeframe: {timeframe}"
        )
    required_columns = [
        "exchange",
        "symbol",
        "agg_trade_id",
        "price",
        "quantity",
        "quote_quantity",
        "trade_time",
        "taker_side",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise TradeFlowFeatureError(
            f"Missing columns: {missing_columns}"
        )
    if df["exchange"].nunique() != 1:
        raise TradeFlowFeatureError(
            "Input contains multiple exchanges"
        )
    if df["symbol"].nunique() != 1:
        raise TradeFlowFeatureError(
            "Input contains multiple symbols"
        )
    exchange = str(
        df["exchange"].iloc[0]
    )
    symbol = str(
        df["symbol"].iloc[0]
    )
    rule = SUPPORTED_BUCKETS[
        timeframe
    ]
    work = (
        df.copy()
        .sort_values(
            [
                "trade_time",
                "agg_trade_id",
            ]
        )
        .reset_index(drop=True)
    )
    is_buy = (
        work["taker_side"]
        .eq("buy")
    )
    is_sell = (
        work["taker_side"]
        .eq("sell")
    )
    work["buy_base"] = np.where(
        is_buy,
        work["quantity"],
        0.0,
    )
    work["sell_base"] = np.where(
        is_sell,
        work["quantity"],
        0.0,
    )
    work["buy_quote"] = np.where(
        is_buy,
        work["quote_quantity"],
        0.0,
    )
    work["sell_quote"] = np.where(
        is_sell,
        work["quote_quantity"],
        0.0,
    )
    work["buy_trade"] = (
        is_buy.astype("int64")
    )
    work["sell_trade"] = (
        is_sell.astype("int64")
    )
    work = (
        work
        .set_index("trade_time")
    )
    grouped = work.resample(
        rule,
        label="left",
        closed="left",
        origin="epoch",
    )
    result = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        trade_count=("agg_trade_id", "count"),
        buy_trade_count=("buy_trade", "sum"),
        sell_trade_count=("sell_trade", "sum"),
        base_volume=("quantity", "sum"),
        buy_base_volume=("buy_base", "sum"),
        sell_base_volume=("sell_base", "sum"),
        quote_volume=("quote_quantity", "sum"),
        buy_quote_volume=("buy_quote", "sum"),
        sell_quote_volume=("sell_quote", "sum"),
    )
    result = (
        result
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "trade_time": "open_time",
            }
        )
    )
    # ========================================================
    # RAW DELTA
    # ========================================================
    result["base_delta"] = (
        result["buy_base_volume"]
        - result["sell_base_volume"]
    )
    result["quote_delta"] = (
        result["buy_quote_volume"]
        - result["sell_quote_volume"]
    )
    # ========================================================
    # AVERAGE TRADE SIZE
    # ========================================================
    result["avg_trade_base"] = np.where(
        result["trade_count"] > 0,
        result["base_volume"]
        / result["trade_count"],
        np.nan,
    )
    result["avg_trade_quote"] = np.where(
        result["trade_count"] > 0,
        result["quote_volume"]
        / result["trade_count"],
        np.nan,
    )
    result["avg_buy_trade_base"] = np.where(
        result["buy_trade_count"] > 0,
        result["buy_base_volume"]
        / result["buy_trade_count"],
        np.nan,
    )
    result["avg_sell_trade_base"] = np.where(
        result["sell_trade_count"] > 0,
        result["sell_base_volume"]
        / result["sell_trade_count"],
        np.nan,
    )
    result["avg_buy_trade_quote"] = np.where(
        result["buy_trade_count"] > 0,
        result["buy_quote_volume"]
        / result["buy_trade_count"],
        np.nan,
    )
    result["avg_sell_trade_quote"] = np.where(
        result["sell_trade_count"] > 0,
        result["sell_quote_volume"]
        / result["sell_trade_count"],
        np.nan,
    )
    # ========================================================
    # VWAP
    # ========================================================
    result["vwap"] = np.where(
        result["base_volume"] > 0,
        result["quote_volume"]
        / result["base_volume"],
        np.nan,
    )
    # ========================================================
    # NORMALIZED VOLUME RATIOS
    # ========================================================
    result["buy_base_ratio"] = np.where(
        result["base_volume"] > 0,
        result["buy_base_volume"]
        / result["base_volume"],
        np.nan,
    )
    result["sell_base_ratio"] = np.where(
        result["base_volume"] > 0,
        result["sell_base_volume"]
        / result["base_volume"],
        np.nan,
    )
    result["buy_quote_ratio"] = np.where(
        result["quote_volume"] > 0,
        result["buy_quote_volume"]
        / result["quote_volume"],
        np.nan,
    )
    result["sell_quote_ratio"] = np.where(
        result["quote_volume"] > 0,
        result["sell_quote_volume"]
        / result["quote_volume"],
        np.nan,
    )
    # ========================================================
    # NORMALIZED DELTA
    # ========================================================
    result["base_delta_pct"] = np.where(
        result["base_volume"] > 0,
        result["base_delta"]
        / result["base_volume"],
        np.nan,
    )
    result["quote_delta_pct"] = np.where(
        result["quote_volume"] > 0,
        result["quote_delta"]
        / result["quote_volume"],
        np.nan,
    )
    # ========================================================
    # TRADE-COUNT IMBALANCE
    # ========================================================
    result["buy_trade_ratio"] = np.where(
        result["trade_count"] > 0,
        result["buy_trade_count"]
        / result["trade_count"],
        np.nan,
    )
    result["sell_trade_ratio"] = np.where(
        result["trade_count"] > 0,
        result["sell_trade_count"]
        / result["trade_count"],
        np.nan,
    )
    result["trade_count_imbalance"] = np.where(
        result["trade_count"] > 0,
        (
            result["buy_trade_count"]
            - result["sell_trade_count"]
        )
        / result["trade_count"],
        np.nan,
    )
    # ========================================================
    # METADATA
    # ========================================================
    result["exchange"] = exchange
    result["symbol"] = symbol
    result["timeframe"] = timeframe
    bucket_delta = pd.to_timedelta(
        rule
    )
    result["close_time"] = (
        result["open_time"]
        + bucket_delta
    )
    integer_columns = [
        "trade_count",
        "buy_trade_count",
        "sell_trade_count",
    ]
    for column in integer_columns:
        result[column] = (
            result[column]
            .astype("int64")
        )
    columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "trade_count",
        "buy_trade_count",
        "sell_trade_count",
        "buy_trade_ratio",
        "sell_trade_ratio",
        "trade_count_imbalance",
        "base_volume",
        "buy_base_volume",
        "sell_base_volume",
        "buy_base_ratio",
        "sell_base_ratio",
        "quote_volume",
        "buy_quote_volume",
        "sell_quote_volume",
        "buy_quote_ratio",
        "sell_quote_ratio",
        "base_delta",
        "quote_delta",
        "base_delta_pct",
        "quote_delta_pct",
        "avg_trade_base",
        "avg_trade_quote",
        "avg_buy_trade_base",
        "avg_sell_trade_base",
        "avg_buy_trade_quote",
        "avg_sell_trade_quote",
        "vwap",
    ]
    return (
        result[columns]
        .sort_values("open_time")
        .reset_index(drop=True)
    )
def add_cvd_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cumulative-volume-delta features to trade-flow bars.
    Requires aggregated flow bars from aggregate_aggtrades().
    CVD definitions
    ---------------
    base_cvd:
        cumulative sum of base_delta from the beginning
        of the available dataset.
    quote_cvd:
        cumulative sum of quote_delta from the beginning
        of the available dataset.
    daily_base_cvd / daily_quote_cvd:
        reset at 00:00 UTC each calendar day.
    rolling_base_cvd_60 / rolling_quote_cvd_60:
        rolling sum over the last 60 observations.
    rolling_base_cvd_1440 / rolling_quote_cvd_1440:
        rolling sum over the last 1440 observations.
    Notes
    -----
    For 1-minute bars:
        60 observations   ≈ 1 hour
        1440 observations ≈ 24 hours
    For other timeframes these are observation counts,
    not fixed wall-clock durations.
    Therefore long-history analysis should always retain
    the timeframe column alongside these features.
    """
    if df.empty:
        return df.copy()
    required_columns = [
        "open_time",
        "base_delta",
        "quote_delta",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise TradeFlowFeatureError(
            f"Missing columns for CVD: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    # ========================================================
    # GLOBAL CVD
    # ========================================================
    result["base_cvd"] = (
        result["base_delta"]
        .cumsum()
    )
    result["quote_cvd"] = (
        result["quote_delta"]
        .cumsum()
    )
    # ========================================================
    # DAILY RESET CVD
    # ========================================================
    result["_session_day"] = (
        result["open_time"]
        .dt.floor("D")
    )
    result["daily_base_cvd"] = (
        result
        .groupby(
            "_session_day",
            sort=False,
        )["base_delta"]
        .cumsum()
    )
    result["daily_quote_cvd"] = (
        result
        .groupby(
            "_session_day",
            sort=False,
        )["quote_delta"]
        .cumsum()
    )
    # ========================================================
    # ROLLING CVD
    # ========================================================
    for window in (
        60,
        1440,
    ):
        result[
            f"rolling_base_cvd_{window}"
        ] = (
            result["base_delta"]
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
        )
        result[
            f"rolling_quote_cvd_{window}"
        ] = (
            result["quote_delta"]
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
        )
    result = result.drop(
        columns=[
            "_session_day",
        ]
    )
    return result