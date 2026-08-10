from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.spot_perp_flow import (
    build_spot_perp_flow_state,
    resample_spot_flow_to_5m,
)


def make_spot(
    spot_delta_pct_5m: list[float],
) -> pd.DataFrame:
    """
    Build 1m Spot flow that aggregates exactly to the
    supplied 5m normalized delta values.
    """

    rows = []

    start = pd.Timestamp(
        "2026-08-01T00:00:00Z"
    )

    for bucket_index, delta_pct in enumerate(
        spot_delta_pct_5m
    ):
        for minute in range(5):
            open_time = (
                start
                + pd.Timedelta(
                    minutes=(
                        bucket_index * 5
                        + minute
                    )
                )
            )

            quote_volume = 100.0

            rows.append(
                {
                    "open_time": open_time,
                    "quote_volume": quote_volume,
                    "quote_delta": (
                        quote_volume
                        * delta_pct
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_perp(
    perp_delta_pct: list[float],
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=len(perp_delta_pct),
        freq="5min",
    )

    total = np.array(
        [100.0] * len(perp_delta_pct),
        dtype=float,
    )

    delta = (
        total
        * np.array(
            perp_delta_pct,
            dtype=float,
        )
    )

    buy = (
        total
        + delta
    ) / 2.0

    sell = (
        total
        - delta
    ) / 2.0

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "futures_buy_volume": buy,
            "futures_sell_volume": sell,
            "futures_total_volume": total,
            "futures_taker_delta": delta,
            "futures_taker_delta_pct": (
                perp_delta_pct
            ),
        }
    )


def test_spot_resample():
    spot = make_spot(
        [0.20]
    )

    result = resample_spot_flow_to_5m(
        spot
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "spot_delta_pct",
        ],
        0.20,
    )


def test_confirmed_buy():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.20]
        ),
        make_perp(
            [0.18]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "confirmed_buy"
    )


def test_confirmed_sell():
    result = build_spot_perp_flow_state(
        make_spot(
            [-0.20]
        ),
        make_perp(
            [-0.18]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "confirmed_sell"
    )


def test_spot_led_buy():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.30]
        ),
        make_perp(
            [0.05]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "spot_led_buy"
    )


def test_perp_led_buy():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.05]
        ),
        make_perp(
            [0.30]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "perp_led_buy"
    )


def test_spot_buy_perp_sell():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.20]
        ),
        make_perp(
            [-0.25]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "spot_buy_perp_sell"
    )


def test_spot_sell_perp_buy():
    result = build_spot_perp_flow_state(
        make_spot(
            [-0.20]
        ),
        make_perp(
            [0.25]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "spot_sell_perp_buy"
    )


def test_neutral():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.03]
        ),
        make_perp(
            [-0.04]
        ),
    )

    assert (
        result.iloc[0][
            "spot_perp_flow_state"
        ]
        == "neutral"
    )


def test_delta_spread():
    result = build_spot_perp_flow_state(
        make_spot(
            [0.20]
        ),
        make_perp(
            [0.10]
        ),
    )

    assert np.isclose(
        result.iloc[0][
            "spot_perp_delta_spread"
        ],
        0.10,
    )
