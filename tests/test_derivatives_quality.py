from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.derivatives_quality import (
    add_derivatives_quality_masks,
)


def make_df() -> pd.DataFrame:
    """
    Build synthetic hourly derivatives-regime data.

    Time contract
    -------------
    open_time:
        start of hourly bucket

    as_of_time:
        open_time + 1h

    funding_time:
        timestamp of latest known funding observation
    """

    open_time = pd.to_datetime(
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T01:00:00Z",
        ],
        utc=True,
    )

    as_of_time = (
        open_time
        + pd.Timedelta(hours=1)
    )

    funding_time = pd.to_datetime(
        [
            "2026-08-10T00:00:00Z",
            "2026-08-10T00:00:00Z",
        ],
        utc=True,
    )

    return pd.DataFrame(
        {
            "open_time": open_time,
            "as_of_time": as_of_time,

            "open_interest_quote": [
                6_000_000_000.0,
                6_100_000_000.0,
            ],

            "oi_quote_change_pct_1h": [
                0.01,
                0.02,
            ],

            "price_oi_state_valid": [
                True,
                True,
            ],

            "funding_time": funding_time,

            "funding_rate": [
                0.0001,
                0.0001,
            ],

            "basis_rate": [
                0.0002,
                0.0003,
            ],

            "spot_quote_volume_1h": [
                10_000_000.0,
                12_000_000.0,
            ],

            "spot_delta_pct_1h": [
                0.10,
                0.05,
            ],

            "perp_total_volume_1h": [
                1000.0,
                1200.0,
            ],

            "perp_delta_pct_1h": [
                0.20,
                0.15,
            ],

            "spot_perp_valid_share_1h": [
                1.0,
                1.0,
            ],

            "long_liquidation_notional_1h": [
                0.0,
                100_000.0,
            ],

            "short_liquidation_notional_1h": [
                0.0,
                50_000.0,
            ],

            "total_liquidation_notional_1h": [
                0.0,
                150_000.0,
            ],

            "derivatives_regime": [
                "leveraged_long_build",
                "leveraged_long_build",
            ],

            "derivatives_regime_valid": [
                True,
                True,
            ],

            "derivatives_regime_confidence": [
                0.75,
                0.75,
            ],
        }
    )


def test_as_of_time_contract():
    df = make_df()

    result = add_derivatives_quality_masks(
        df
    )

    expected = (
        result["open_time"]
        + pd.Timedelta(hours=1)
    )

    assert (
        result["as_of_time"]
        .eq(expected)
        .all()
    )


def test_core_valid():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert (
        result[
            "derivatives_core_valid"
        ].all()
    )


def test_zero_liquidation_not_assumed_valid():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_observed",
            ]
        )
        is False
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_valid",
            ]
        )
        is False
    )


def test_observed_liquidation_valid():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert (
        bool(
            result.loc[
                1,
                "liquidation_observed",
            ]
        )
        is True
    )

    assert (
        bool(
            result.loc[
                1,
                "liquidation_valid",
            ]
        )
        is True
    )


def test_stale_funding_invalid():
    df = make_df()

    df.loc[
        1,
        "funding_time",
    ] = pd.Timestamp(
        "2026-08-08T00:00:00Z"
    )

    result = add_derivatives_quality_masks(
        df,
        funding_max_age_hours=12.0,
    )

    assert (
        bool(
            result.loc[
                1,
                "funding_valid",
            ]
        )
        is False
    )

    assert (
        bool(
            result.loc[
                1,
                "derivatives_core_valid",
            ]
        )
        is False
    )


def test_future_funding_invalid():
    df = make_df()

    df.loc[
        0,
        "funding_time",
    ] = pd.Timestamp(
        "2026-08-10T02:00:00Z"
    )

    result = add_derivatives_quality_masks(
        df
    )

    assert (
        bool(
            result.loc[
                0,
                "funding_causal",
            ]
        )
        is False
    )

    assert (
        bool(
            result.loc[
                0,
                "funding_valid",
            ]
        )
        is False
    )


def test_funding_age_uses_as_of_time():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert np.isclose(
        result.loc[
            0,
            "funding_age_hours",
        ],
        1.0,
    )

    assert np.isclose(
        result.loc[
            1,
            "funding_age_hours",
        ],
        2.0,
    )


def test_bad_spot_perp_coverage_invalid():
    df = make_df()

    df.loc[
        1,
        "spot_perp_valid_share_1h",
    ] = 0.50

    result = add_derivatives_quality_masks(
        df
    )

    assert (
        bool(
            result.loc[
                1,
                "spot_perp_valid",
            ]
        )
        is False
    )


def test_quality_score_bounds():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert (
        result[
            "derivatives_quality_score"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_quality_tier_exists():
    result = add_derivatives_quality_masks(
        make_df()
    )

    allowed = {
        "insufficient",
        "partial",
        "good",
        "complete",
    }

    assert (
        set(
            result[
                "derivatives_quality_tier"
            ].unique()
        )
        <= allowed
    )


def test_evidence_count_range():
    result = add_derivatives_quality_masks(
        make_df()
    )

    assert (
        result[
            "derivatives_regime_evidence_count"
        ]
        .between(
            0,
            6,
        )
        .all()
    )


def test_invalid_regime_confidence_removed():
    df = make_df()

    df.loc[
        1,
        "open_interest_quote",
    ] = np.nan

    result = add_derivatives_quality_masks(
        df
    )

    assert (
        bool(
            result.loc[
                1,
                "derivatives_regime_quality_valid",
            ]
        )
        is False
    )

    assert np.isnan(
        result.loc[
            1,
            "derivatives_regime_confidence",
        ]
    )
