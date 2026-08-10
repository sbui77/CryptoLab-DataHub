import pandas as pd

from cryptolab.sources.derivatives_funding import (
    MAX_FUNDING_LIMIT,
    fetch_funding_rate_history,
)


def test_max_limit():
    assert MAX_FUNDING_LIMIT == 1000


def test_invalid_limit():
    try:
        fetch_funding_rate_history(
            symbol="BTCUSDT",
            limit=1001,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_invalid_time_range():
    try:
        fetch_funding_rate_history(
            symbol="BTCUSDT",
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
