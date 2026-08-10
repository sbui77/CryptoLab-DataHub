from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.derivatives_regime import (
    aggregate_derivatives_to_1h,
    aggregate_spot_perp_to_1h,
    build_derivatives_regime,
)


def make_price_oi(
    states: list[str],
) -> pd.DataFrame:
    rows = len(states)

    valid = [
        state != "unknown"
        for state in states
    ]

    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-08-01T00:00:00Z",
                periods=rows,
                freq="1h",
            ),
            "close": [
                60_000.0
            ] * rows,
            "price_return_1h": [
                0.01
            ] * rows,
            "oi_quote_change_pct_1h": [
                0.01
            ] * rows,
            "price_oi_state": states,
            "price_oi_state_valid": valid,
        }
    )


def make_spot_perp(
    hours: int,
    spot_delta: float,
    perp_delta: float,
) -> pd.DataFrame:
    rows = hours * 12

    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    spot_volume = 100.0
    perp_volume = 100.0

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "spot_quote_volume": [
                spot_volume
            ] * rows,
            "spot_quote_delta": [
                spot_volume
                * spot_delta
            ] * rows,
            "perp_total_volume": [
                perp_volume
            ] * rows,
            "perp_taker_delta": [
                perp_volume
                * perp_delta
            ] * rows,
            "spot_perp_flow_valid": [
                True
            ] * rows,
        }
    )


def make_derivatives(
    hours: int,
    funding_rate: float = 0.0001,
    basis_rate: float = 0.0001,
    long_liq: float = 0.0,
    short_liq: float = 0.0,
) -> pd.DataFrame:
    rows = hours * 12

    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    funding_times = pd.Series(
        [
            pd.Timestamp(
                "2026-08-01T00:00:00Z"
            )
        ] * rows,
        dtype="datetime64[ns, UTC]",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open_interest_quote": [
                6_000_000_000.0
            ] * rows,
            "funding_time": funding_times,
            "funding_rate": [
                funding_rate
            ] * rows,
            "funding_rate_bps": [
                funding_rate * 10_000
            ] * rows,
            "basis_rate": [
                basis_rate
            ] * rows,
            "basis_bps": [
                basis_rate * 10_000
            ] * rows,
            "basis_zscore_24h": [
                0.0
            ] * rows,
            "long_liquidation_notional": [
                long_liq
            ] * rows,
            "short_liquidation_notional": [
                short_liq
            ] * rows,
        }
    )


def test_spot_perp_hourly_aggregation():
    result = aggregate_spot_perp_to_1h(
        make_spot_perp(
            hours=1,
            spot_delta=0.20,
            perp_delta=0.10,
        )
    )

    assert len(result) == 1

    assert np.isclose(
        result.loc[
            0,
            "spot_delta_pct_1h",
        ],
        0.20,
    )

    assert np.isclose(
        result.loc[
            0,
            "perp_delta_pct_1h",
        ],
        0.10,
    )


def test_derivatives_hourly_liquidations():
    result = aggregate_derivatives_to_1h(
        make_derivatives(
            hours=1,
            long_liq=100.0,
            short_liq=50.0,
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "long_liquidation_notional_1h",
        ],
        1200.0,
    )

    assert np.isclose(
        result.loc[
            0,
            "short_liquidation_notional_1h",
        ],
        600.0,
    )


def test_funding_time_preserved():
    result = aggregate_derivatives_to_1h(
        make_derivatives(
            hours=1,
        )
    )

    assert (
        result.loc[
            0,
            "funding_time",
        ]
        == pd.Timestamp(
            "2026-08-01T00:00:00Z"
        )
    )


def test_hourly_as_of_time():
    result = aggregate_derivatives_to_1h(
        make_derivatives(
            hours=1,
        )
    )

    assert (
        result.loc[
            0,
            "open_time",
        ]
        == pd.Timestamp(
            "2026-08-01T00:00:00Z"
        )
    )

    assert (
        result.loc[
            0,
            "as_of_time",
        ]
        == pd.Timestamp(
            "2026-08-01T01:00:00Z"
        )
    )


def test_leveraged_long_build():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "position_build_up",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=0.20,
            perp_delta=0.20,
        ),
        derivatives_df=make_derivatives(
            hours=1,
            funding_rate=0.0001,
            basis_rate=0.0001,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "leveraged_long_build"
    )


def test_leveraged_short_build():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "short_build_up_candidate",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=-0.20,
            perp_delta=-0.20,
        ),
        derivatives_df=make_derivatives(
            hours=1,
            funding_rate=-0.0001,
            basis_rate=-0.0001,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "leveraged_short_build"
    )


def test_short_covering():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "short_covering_candidate",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=0.20,
            perp_delta=0.20,
        ),
        derivatives_df=make_derivatives(
            hours=1,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "short_covering"
    )


def test_long_deleveraging():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "long_deleveraging_candidate",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=-0.20,
            perp_delta=-0.20,
        ),
        derivatives_df=make_derivatives(
            hours=1,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "long_deleveraging"
    )


def test_balanced():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "neutral",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=0.02,
            perp_delta=-0.02,
        ),
        derivatives_df=make_derivatives(
            hours=1,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "balanced"
    )


def test_unknown_when_price_oi_invalid():
    result = build_derivatives_regime(
        price_oi_df=make_price_oi(
            [
                "unknown",
            ]
        ),
        spot_perp_df=make_spot_perp(
            hours=1,
            spot_delta=0.20,
            perp_delta=0.20,
        ),
        derivatives_df=make_derivatives(
            hours=1,
        ),
    )

    assert (
        result.iloc[0][
            "derivatives_regime"
        ]
        == "unknown"
    )

    assert (
        bool(
            result.iloc[0][
                "derivatives_regime_valid"
            ]
        )
        is False
    )
