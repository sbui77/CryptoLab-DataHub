from __future__ import annotations
import numpy as np
import pandas as pd
class PriceFeatureError(ValueError):
    """Raised when price feature generation fails."""
# ============================================================
# RESAMPLING CONFIGURATION
# ============================================================
_RESAMPLE_RULES = {
    "4h": "4h",
    "1d": "1D",
    "1w": "7D",
}
_EXPECTED_SOURCE_COUNT = {
    "4h": 4,
    "1d": 24,
    "1w": 168,
}
# ============================================================
# OHLCV RESAMPLING
# ============================================================
def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
) -> pd.DataFrame:
    """
    Resample canonical 1h OHLCV into higher timeframes.
    Supported:
        4h
        1d
        1w
    Weekly candles are anchored to Monday 00:00 UTC.
    Only complete higher-timeframe candles are retained.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input OHLCV dataframe is empty"
        )
    if target_timeframe not in _RESAMPLE_RULES:
        raise PriceFeatureError(
            f"Unsupported target timeframe: "
            f"{target_timeframe}"
        )
    required_columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "close_time",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ingested_at",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise PriceFeatureError(
            f"Missing columns: {missing_columns}"
        )
    if df["timeframe"].nunique() != 1:
        raise PriceFeatureError(
            "Input dataframe contains multiple timeframes"
        )
    source_timeframe = df["timeframe"].iloc[0]
    if source_timeframe != "1h":
        raise PriceFeatureError(
            "Canonical resampling currently "
            "requires 1h input"
        )
    if df["exchange"].nunique() != 1:
        raise PriceFeatureError(
            "Input dataframe contains multiple exchanges"
        )
    if df["symbol"].nunique() != 1:
        raise PriceFeatureError(
            "Input dataframe contains multiple symbols"
        )
    exchange = df["exchange"].iloc[0]
    symbol = df["symbol"].iloc[0]
    work = (
        df.copy()
        .sort_values("open_time")
        .set_index("open_time")
    )
    work["_source_count"] = 1
    rule = _RESAMPLE_RULES[
        target_timeframe
    ]
    expected_source_count = (
        _EXPECTED_SOURCE_COUNT[
            target_timeframe
        ]
    )
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "base_volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "close_time": "last",
        "ingested_at": "max",
        "_source_count": "sum",
    }
    if target_timeframe == "1w":
        origin = pd.Timestamp(
            "1970-01-05T00:00:00Z"
        )
    else:
        origin = "epoch"
    result = (
        work
        .resample(
            rule,
            label="left",
            closed="left",
            origin=origin,
        )
        .agg(aggregation)
        .reset_index()
    )
    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )
    result = result[
        result["_source_count"]
        == expected_source_count
    ].copy()
    result = result.drop(
        columns=["_source_count"]
    )
    result["exchange"] = exchange
    result["symbol"] = symbol
    result["timeframe"] = target_timeframe
    result["trade_count"] = (
        result["trade_count"]
        .astype("int64")
    )
    columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "close_time",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ingested_at",
    ]
    return (
        result[columns]
        .sort_values("open_time")
        .reset_index(drop=True)
    )
# ============================================================
# BASIC RETURNS
# ============================================================
def add_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add one-period simple and logarithmic returns.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    result["simple_return"] = (
        result["close"]
        .pct_change(
            fill_method=None
        )
    )
    result["log_return"] = np.log(
        result["close"]
        / result["close"].shift(1)
    )
    return result
# ============================================================
# ROLLING RETURNS
# ============================================================
def add_rolling_returns(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (
        24,
        168,
        720,
    ),
) -> pd.DataFrame:
    """
    Add rolling close-to-close returns.
    Default horizons assume hourly data.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    horizon_names = {
        24: "24h",
        168: "7d",
        720: "30d",
    }
    for horizon in horizons:
        if horizon <= 0:
            raise PriceFeatureError(
                "Return horizons must be greater than zero"
            )
        name = horizon_names.get(
            horizon,
            str(horizon),
        )
        result[
            f"return_{name}"
        ] = (
            result["close"]
            / result["close"].shift(horizon)
            - 1.0
        )
    return result
# ============================================================
# RANGE / TRUE RANGE / ATR
# ============================================================
def add_range_features(
    df: pd.DataFrame,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Add candle range, True Range and ATR.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    previous_close = (
        result["close"]
        .shift(1)
    )
    result["high_low_range"] = (
        result["high"]
        - result["low"]
    )
    result["range_pct"] = (
        result["high_low_range"]
        / result["open"]
    )
    high_low = (
        result["high"]
        - result["low"]
    )
    high_previous_close = (
        result["high"]
        - previous_close
    ).abs()
    low_previous_close = (
        result["low"]
        - previous_close
    ).abs()
    result["true_range"] = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    ).max(
        axis=1
    )
    atr_column = (
        f"atr_{atr_window}"
    )
    atr_pct_column = (
        f"atr_{atr_window}_pct"
    )
    result[atr_column] = (
        result["true_range"]
        .rolling(
            window=atr_window,
            min_periods=atr_window,
        )
        .mean()
    )
    result[atr_pct_column] = (
        result[atr_column]
        / result["close"]
    )
    return result
# ============================================================
# ROLLING HIGH / LOW / RANGE POSITION
# ============================================================
def add_rolling_range_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (
        24,
        168,
        720,
    ),
) -> pd.DataFrame:
    """
    Add rolling high, low, range and range position.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    for window in windows:
        rolling_high_column = (
            f"rolling_high_{window}"
        )
        rolling_low_column = (
            f"rolling_low_{window}"
        )
        rolling_range_column = (
            f"rolling_range_{window}"
        )
        range_position_column = (
            f"range_position_{window}"
        )
        result[
            rolling_high_column
        ] = (
            result["high"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .max()
        )
        result[
            rolling_low_column
        ] = (
            result["low"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .min()
        )
        result[
            rolling_range_column
        ] = (
            result[rolling_high_column]
            - result[rolling_low_column]
        )
        rolling_range = (
            result[
                rolling_range_column
            ]
        )
        result[
            range_position_column
        ] = np.where(
            rolling_range > 0,
            (
                result["close"]
                - result[
                    rolling_low_column
                ]
            )
            / rolling_range,
            np.nan,
        )
    return result
# ============================================================
# REALIZED VOLATILITY
# ============================================================
def add_volatility_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add annualized realized-volatility features.
    Assumes hourly log returns.
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    if "log_return" not in df.columns:
        raise PriceFeatureError(
            "log_return is required"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    hours_per_year = (
        24 * 365
    )
    annualization_factor = np.sqrt(
        hours_per_year
    )
    volatility_windows = {
        "24h": 24,
        "7d": 168,
        "30d": 720,
        "90d": 2160,
    }
    for name, window in (
        volatility_windows.items()
    ):
        result[
            f"rv_{name}"
        ] = (
            result["log_return"]
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(
                ddof=1
            )
            * annualization_factor
        )
    result[
        "rv_30d_percentile_1y"
    ] = (
        result["rv_30d"]
        .rolling(
            window=hours_per_year,
            min_periods=hours_per_year,
        )
        .rank(
            pct=True
        )
    )
    rolling_mean = (
        result["rv_30d"]
        .rolling(
            window=hours_per_year,
            min_periods=hours_per_year,
        )
        .mean()
    )
    rolling_std = (
        result["rv_30d"]
        .rolling(
            window=hours_per_year,
            min_periods=hours_per_year,
        )
        .std(
            ddof=1
        )
    )
    result[
        "rv_30d_zscore_1y"
    ] = np.where(
        rolling_std > 0,
        (
            result["rv_30d"]
            - rolling_mean
        )
        / rolling_std,
        np.nan,
    )
    return result
# ============================================================
# VWAP
# ============================================================
def add_vwap_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add session VWAP features from hourly Binance data.
    VWAP is calculated as:
        cumulative quote_volume
        -----------------------
        cumulative base_volume
    Sessions:
        daily   -> 00:00 UTC
        weekly  -> Monday 00:00 UTC
        monthly -> first day of month 00:00 UTC
    Generated fields:
        daily_vwap
        weekly_vwap
        monthly_vwap
        distance_to_daily_vwap
        distance_to_weekly_vwap
        distance_to_monthly_vwap
        distance_to_daily_vwap_pct
        distance_to_weekly_vwap_pct
        distance_to_monthly_vwap_pct
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    required_columns = [
        "open_time",
        "close",
        "base_volume",
        "quote_volume",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise PriceFeatureError(
            f"Missing columns: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    # --------------------------------------------------------
    # DAILY SESSION
    # --------------------------------------------------------
    result["_daily_session"] = (
        result["open_time"]
        .dt.floor("D")
    )
    daily_quote_cumulative = (
        result
        .groupby(
            "_daily_session",
            sort=False,
        )["quote_volume"]
        .cumsum()
    )
    daily_base_cumulative = (
        result
        .groupby(
            "_daily_session",
            sort=False,
        )["base_volume"]
        .cumsum()
    )
    result["daily_vwap"] = np.where(
        daily_base_cumulative > 0,
        daily_quote_cumulative
        / daily_base_cumulative,
        np.nan,
    )
    # --------------------------------------------------------
    # WEEKLY SESSION
    # Monday 00:00 UTC
    # --------------------------------------------------------
    weekday = (
        result["open_time"]
        .dt.weekday
    )
    result["_weekly_session"] = (
        result["open_time"].dt.floor("D")
        - pd.to_timedelta(
            weekday,
            unit="D",
        )
    )
    weekly_quote_cumulative = (
        result
        .groupby(
            "_weekly_session",
            sort=False,
        )["quote_volume"]
        .cumsum()
    )
    weekly_base_cumulative = (
        result
        .groupby(
            "_weekly_session",
            sort=False,
        )["base_volume"]
        .cumsum()
    )
    result["weekly_vwap"] = np.where(
        weekly_base_cumulative > 0,
        weekly_quote_cumulative
        / weekly_base_cumulative,
        np.nan,
    )
    # --------------------------------------------------------
    # MONTHLY SESSION
    #
    # Build the first day of each UTC month directly.
    # This avoids converting timezone-aware timestamps
    # to Period objects and therefore avoids timezone warnings.
    # --------------------------------------------------------
    result["_monthly_session"] = pd.to_datetime(
        {
            "year": (
                result["open_time"]
                .dt.year
            ),
            "month": (
                result["open_time"]
                .dt.month
            ),
            "day": 1,
        },
        utc=True,
    )
    monthly_quote_cumulative = (
        result
        .groupby(
            "_monthly_session",
            sort=False,
        )["quote_volume"]
        .cumsum()
    )
    monthly_base_cumulative = (
        result
        .groupby(
            "_monthly_session",
            sort=False,
        )["base_volume"]
        .cumsum()
    )
    result["monthly_vwap"] = np.where(
        monthly_base_cumulative > 0,
        monthly_quote_cumulative
        / monthly_base_cumulative,
        np.nan,
    )
    # --------------------------------------------------------
    # DISTANCE TO VWAP
    # --------------------------------------------------------
    for session in [
        "daily",
        "weekly",
        "monthly",
    ]:
        vwap_column = (
            f"{session}_vwap"
        )
        distance_column = (
            f"distance_to_{session}_vwap"
        )
        distance_pct_column = (
            f"distance_to_{session}_vwap_pct"
        )
        result[
            distance_column
        ] = (
            result["close"]
            - result[vwap_column]
        )
        result[
            distance_pct_column
        ] = np.where(
            result[vwap_column] > 0,
            (
                result["close"]
                / result[vwap_column]
                - 1.0
            ),
            np.nan,
        )
    result = result.drop(
        columns=[
            "_daily_session",
            "_weekly_session",
            "_monthly_session",
        ]
    )
    return result
# ============================================================
# CORE PRICE FEATURE PIPELINE
# ============================================================
def build_price_features(
    df: pd.DataFrame,
    atr_window: int = 14,
    rolling_windows: tuple[int, ...] = (
        24,
        168,
        720,
    ),
) -> pd.DataFrame:
    """
    Build the core candle-level Price Layer.
    Pipeline:
        OHLCV
          ↓
        Returns
          ↓
        Rolling Returns
          ↓
        Range / ATR
          ↓
        Rolling Location
          ↓
        Realized Volatility
          ↓
        Session VWAP
    """
    if df.empty:
        raise PriceFeatureError(
            "Input dataframe is empty"
        )
    result = add_returns(
        df
    )
    result = add_rolling_returns(
        result
    )
    result = add_range_features(
        result,
        atr_window=atr_window,
    )
    result = add_rolling_range_features(
        result,
        windows=rolling_windows,
    )
    result = add_volatility_features(
        result
    )
    result = add_vwap_features(
        result
    )
    return result