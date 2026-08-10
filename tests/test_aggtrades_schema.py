import pandas as pd
from cryptolab.schemas.aggtrades import (
    AGGTRADE_COLUMNS,
    normalize_aggtrades,
)
def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
            ],
            "agg_trade_id": [
                100,
                101,
            ],
            "price": [
                60000.0,
                60100.0,
            ],
            "quantity": [
                0.5,
                0.25,
            ],
            "first_trade_id": [
                1000,
                1001,
            ],
            "last_trade_id": [
                1000,
                1002,
            ],
            "trade_time": [
                "2026-08-10T00:00:00Z",
                "2026-08-10T00:00:01Z",
            ],
            "buyer_is_maker": [
                False,
                True,
            ],
        }
    )
def test_columns():
    result = normalize_aggtrades(
        make_df()
    )
    assert (
        list(result.columns)
        == AGGTRADE_COLUMNS
    )
def test_quote_quantity():
    result = normalize_aggtrades(
        make_df()
    )
    assert (
        result.loc[
            0,
            "quote_quantity",
        ]
        == 30000.0
    )
def test_taker_buy():
    result = normalize_aggtrades(
        make_df()
    )
    assert (
        result.loc[
            0,
            "taker_side",
        ]
        == "buy"
    )
def test_taker_sell():
    result = normalize_aggtrades(
        make_df()
    )
    assert (
        result.loc[
            1,
            "taker_side",
        ]
        == "sell"
    )
def test_timestamp_utc():
    result = normalize_aggtrades(
        make_df()
    )
    assert str(
        result[
            "trade_time"
        ].dt.tz
    ) == "UTC"
