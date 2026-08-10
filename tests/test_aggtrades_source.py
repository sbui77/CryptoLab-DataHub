import pandas as pd
from cryptolab.sources.aggtrades import (
    _parse_aggtrades,
)
def test_parse_aggtrades():
    payload = [
        {
            "a": 100,
            "p": "60000.00",
            "q": "0.50000000",
            "f": 1000,
            "l": 1001,
            "T": 1786320000000,
            "m": False,
            "M": True,
        },
        {
            "a": 101,
            "p": "60100.00",
            "q": "0.25000000",
            "f": 1002,
            "l": 1003,
            "T": 1786320001000,
            "m": True,
            "M": True,
        },
    ]
    result = _parse_aggtrades(
        payload,
        symbol="BTCUSDT",
    )
    assert len(result) == 2
    assert (
        result.loc[
            0,
            "agg_trade_id",
        ]
        == 100
    )
    assert (
        result.loc[
            0,
            "taker_side",
        ]
        == "buy"
    )
    assert (
        result.loc[
            1,
            "taker_side",
        ]
        == "sell"
    )
    assert (
        result.loc[
            0,
            "quote_quantity",
        ]
        == 30000.0
    )
    assert str(
        result[
            "trade_time"
        ].dt.tz
    ) == "UTC"
def test_parse_empty():
    result = _parse_aggtrades(
        [],
        symbol="BTCUSDT",
    )
    assert isinstance(
        result,
        pd.DataFrame,
    )
    assert result.empty
