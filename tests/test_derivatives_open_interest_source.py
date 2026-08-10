from __future__ import annotations

import pandas as pd

from cryptolab.sources.derivatives_open_interest import (
    MAX_HISTORY_LIMIT,
    fetch_open_interest_history,
)


def test_history_limit_constant():
    assert MAX_HISTORY_LIMIT == 500


def test_invalid_limit_rejected():
    try:
        fetch_open_interest_history(
            symbol="BTCUSDT",
            period="5m",
            limit=501,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )


def test_timestamp_input_validation():
    try:
        fetch_open_interest_history(
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
