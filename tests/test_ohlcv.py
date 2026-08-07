import pandas as pd
import pytest

from cryptolab.quality.ohlcv import (
    OHLCVQualityError,
    validate_ohlcv,
)


def make_valid_df():
    return pd.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h"],
            "open_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z"]
            ),
            "open": [90000.0],
            "high": [91000.0],
            "low": [89000.0],
            "close": [90500.0],
            "base_volume": [100.0],
            "quote_volume": [9_000_000.0],
            "close_time": pd.to_datetime(
                ["2026-01-01T00:59:59Z"]
            ),
            "trade_count": [1000],
            "taker_buy_base_volume": [55.0],
            "taker_buy_quote_volume": [
                5_000_000.0
            ],
            "ingested_at": pd.to_datetime(
                ["2026-01-01T01:00:01Z"]
            ),
        }
    )


def test_valid_ohlcv():
    df = make_valid_df()

    validate_ohlcv(df)


def test_invalid_high_low():
    df = make_valid_df()

    df.loc[0, "high"] = 88000.0

    with pytest.raises(OHLCVQualityError):
        validate_ohlcv(df)


def test_duplicate_candle():
    df = make_valid_df()

    df = pd.concat(
        [df, df],
        ignore_index=True,
    )

    with pytest.raises(OHLCVQualityError):
        validate_ohlcv(df)
