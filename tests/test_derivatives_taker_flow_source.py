import pandas as pd

from cryptolab.sources.derivatives_taker_flow import (
    MAX_TAKER_FLOW_LIMIT,
    fetch_taker_flow_history,
)


def test_max_limit():
    assert MAX_TAKER_FLOW_LIMIT == 500


def test_invalid_limit():
    try:
        fetch_taker_flow_history(
            symbol="BTCUSDT",
            period="5m",
            limit=501,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_invalid_time_range():
    try:
        fetch_taker_flow_history(
            symbol="BTCUSDT",
            period="5m",
            start_time=pd.Timestamp(
                "2026-08-10T10:00:00Z"
            ),
            end_time=pd.Timestamp(
                "2026-08-10T09:00:00Z"
            ),
            limit=10,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
