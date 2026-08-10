from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.price_oi_state import (
    build_price_oi_state,
    resample_open_interest_to_1h,
)


def make_price(
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-08-01T00:00:00Z",
                periods=len(closes),
                freq="1h",
            ),
            "close": closes,
        }
    )


def make_derivatives(
    hourly_oi: list[float],
) -> pd.DataFrame:
    """
    Create 5m OI where each hour ends at supplied OI.
    """

    rows = len(hourly_oi) * 12

    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    values = []

    for hourly_value in hourly_oi:
        start_value = (
            hourly_value
            - 11.0
        )

        values.extend(
            np.linspace(
                start_value,
                hourly_value,
                12,
            )
        )

    values_array = np.array(
        values,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,

            "open_interest_base": (
                values_array
            ),

            "open_interest_quote": (
                values_array
                * 60_000.0
            ),
        }
    )


def test_oi_resample_uses_last():
    derivatives = make_derivatives(
        [
            1000.0,
            1010.0,
        ]
    )

    result = (
        resample_open_interest_to_1h(
            derivatives
        )
    )

    assert len(result) == 2

    assert np.isclose(
        result.loc[
            0,
            "open_interest_base",
        ],
        1000.0,
    )

    assert np.isclose(
        result.loc[
            1,
            "open_interest_base",
        ],
        1010.0,
    )


def test_position_build_up():
    price = make_price(
        [
            100.0,
            101.0,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            1010.0,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
        price_threshold=0.001,
        oi_threshold=0.002,
    )

    assert (
        result.iloc[-1][
            "price_oi_state"
        ]
        == "position_build_up"
    )


def test_short_covering_candidate():
    price = make_price(
        [
            100.0,
            101.0,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            990.0,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
        price_threshold=0.001,
        oi_threshold=0.002,
    )

    assert (
        result.iloc[-1][
            "price_oi_state"
        ]
        == "short_covering_candidate"
    )


def test_short_build_up_candidate():
    price = make_price(
        [
            100.0,
            99.0,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            1010.0,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
        price_threshold=0.001,
        oi_threshold=0.002,
    )

    assert (
        result.iloc[-1][
            "price_oi_state"
        ]
        == "short_build_up_candidate"
    )


def test_long_deleveraging_candidate():
    price = make_price(
        [
            100.0,
            99.0,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            990.0,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
        price_threshold=0.001,
        oi_threshold=0.002,
    )

    assert (
        result.iloc[-1][
            "price_oi_state"
        ]
        == "long_deleveraging_candidate"
    )


def test_neutral_when_move_too_small():
    price = make_price(
        [
            100.0,
            100.01,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            1000.5,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
        price_threshold=0.001,
        oi_threshold=0.002,
    )

    assert (
        result.iloc[-1][
            "price_oi_state"
        ]
        == "neutral"
    )


def test_first_row_unknown():
    price = make_price(
        [
            100.0,
            101.0,
        ]
    )

    derivatives = make_derivatives(
        [
            1000.0,
            1010.0,
        ]
    )

    result = build_price_oi_state(
        price,
        derivatives,
    )

    assert (
        result.iloc[0][
            "price_oi_state"
        ]
        == "unknown"
    )

    assert (
        bool(
            result.iloc[0][
                "price_oi_state_valid"
            ]
        )
        is False
    )
