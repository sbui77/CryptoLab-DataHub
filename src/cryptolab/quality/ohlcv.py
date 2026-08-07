from __future__ import annotations

import pandas as pd


class OHLCVQualityError(ValueError):
    """Raised when OHLCV data fails quality checks."""


def validate_ohlcv(
    df: pd.DataFrame,
) -> None:
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
        raise OHLCVQualityError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise OHLCVQualityError(
            "OHLCV dataframe is empty"
        )

    if df[required_columns].isna().any().any():
        raise OHLCVQualityError(
            "OHLCV contains missing values"
        )

    if (df["high"] < df["low"]).any():
        raise OHLCVQualityError(
            "Found high < low"
        )

    if (df["high"] < df["open"]).any():
        raise OHLCVQualityError(
            "Found high < open"
        )

    if (df["high"] < df["close"]).any():
        raise OHLCVQualityError(
            "Found high < close"
        )

    if (df["low"] > df["open"]).any():
        raise OHLCVQualityError(
            "Found low > open"
        )

    if (df["low"] > df["close"]).any():
        raise OHLCVQualityError(
            "Found low > close"
        )

    non_negative_columns = [
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    if (df[non_negative_columns] < 0).any().any():
        raise OHLCVQualityError(
            "Negative values detected"
        )

    duplicates = df.duplicated(
        subset=[
            "exchange",
            "symbol",
            "timeframe",
            "open_time",
        ]
    )

    if duplicates.any():
        raise OHLCVQualityError(
            "Duplicate candles detected"
        )

    if not df["open_time"].is_monotonic_increasing:
        raise OHLCVQualityError(
            "open_time is not sorted"
        )

    if (df["open_time"] >= df["close_time"]).any():
        raise OHLCVQualityError(
            "Invalid candle timestamps"
        )
def find_time_gaps(
    df: pd.DataFrame,
    expected_frequency: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    expected = pd.date_range(
        start=df["open_time"].min(),
        end=df["open_time"].max(),
        freq=expected_frequency,
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        df["open_time"]
    )

    missing = expected.difference(actual)

    return pd.DataFrame(
        {
            "missing_open_time": missing,
        }
    )