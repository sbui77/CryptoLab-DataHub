import numpy as np

from cryptolab.sources.derivatives_liquidations import (
    get_liquidation_stream_url,
    parse_liquidation_message,
)


def make_event(
    side: str = "SELL",
) -> dict:
    return {
        "e": "forceOrder",
        "E": 1568014460893,
        "o": {
            "s": "BTCUSDT",
            "S": side,
            "o": "LIMIT",
            "f": "IOC",
            "q": "0.014",
            "p": "9910",
            "ap": "9910",
            "X": "FILLED",
            "l": "0.014",
            "z": "0.014",
            "T": 1568014460893,
        },
    }


def test_stream_url():
    url = get_liquidation_stream_url(
        "BTCUSDT"
    )

    assert (
        url
        == (
            "wss://fstream.binance.com/"
            "market/ws/btcusdt@forceOrder"
        )
    )


def test_long_liquidation_mapping():
    df = parse_liquidation_message(
        make_event(
            "SELL"
        )
    )

    row = df.iloc[0]

    assert (
        row["liquidation_side"]
        == "long"
    )


def test_short_liquidation_mapping():
    df = parse_liquidation_message(
        make_event(
            "BUY"
        )
    )

    row = df.iloc[0]

    assert (
        row["liquidation_side"]
        == "short"
    )


def test_liquidation_notional():
    df = parse_liquidation_message(
        make_event()
    )

    row = df.iloc[0]

    expected = (
        0.014
        * 9910
    )

    assert np.isclose(
        row[
            "liquidation_notional"
        ],
        expected,
    )
