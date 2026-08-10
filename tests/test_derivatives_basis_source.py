import pandas as pd

from cryptolab.sources.derivatives_basis import (
    MAX_BASIS_LIMIT,
    fetch_basis_history,
)


def test_max_limit():
    assert MAX_BASIS_LIMIT == 500


def test_invalid_limit():
    try:
        fetch_basis_history(
            pair="BTCUSDT",
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
        fetch_basis_history(
            pair="BTCUSDT",
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
