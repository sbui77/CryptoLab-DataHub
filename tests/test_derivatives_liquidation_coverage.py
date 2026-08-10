from __future__ import annotations

import pandas as pd

from cryptolab.features.derivatives_liquidation_coverage import (
    apply_liquidation_coverage,
)


def make_quality() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-08-10T10:00:00Z",
                    "2026-08-10T11:00:00Z",
                    "2026-08-10T12:00:00Z",
                ],
                utc=True,
            ),

            "long_liquidation_notional_1h": [
                0.0,
                100.0,
                0.0,
            ],

            "short_liquidation_notional_1h": [
                0.0,
                50.0,
                0.0,
            ],

            "total_liquidation_notional_1h": [
                0.0,
                150.0,
                0.0,
            ],

            "oi_valid": [
                True,
                True,
                True,
            ],

            "funding_valid": [
                True,
                True,
                True,
            ],

            "basis_valid": [
                True,
                True,
                True,
            ],

            "taker_flow_valid": [
                True,
                True,
                True,
            ],

            "spot_perp_valid": [
                True,
                True,
                True,
            ],

            "liquidation_valid": [
                False,
                True,
                False,
            ],

            "derivatives_regime_valid": [
                True,
                True,
                True,
            ],

            "derivatives_core_valid": [
                True,
                True,
                True,
            ],

            "derivatives_regime_confidence": [
                0.75,
                0.75,
                0.75,
            ],
        }
    )


def make_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-08-10T10:00:00Z",
                    "2026-08-10T11:00:00Z",
                ],
                utc=True,
            ),

            "collector_coverage_ratio": [
                1.0,
                0.90,
            ],

            "liquidation_collector_covered": [
                True,
                True,
            ],
        }
    )


def test_covered_zero_is_valid():
    result = apply_liquidation_coverage(
        make_quality(),
        make_coverage(),
    )

    assert (
        bool(
            result.loc[
                0,
                "liquidation_valid",
            ]
        )
        is True
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


def test_covered_event_is_valid():
    result = apply_liquidation_coverage(
        make_quality(),
        make_coverage(),
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

    assert (
        bool(
            result.loc[
                1,
                "liquidation_observed",
            ]
        )
        is True
    )


def test_uncovered_zero_invalid():
    result = apply_liquidation_coverage(
        make_quality(),
        make_coverage(),
    )

    assert (
        bool(
            result.loc[
                2,
                "liquidation_valid",
            ]
        )
        is False
    )


def test_covered_zero_increases_evidence():
    result = apply_liquidation_coverage(
        make_quality(),
        make_coverage(),
    )

    assert (
        result.loc[
            0,
            "derivatives_regime_evidence_count",
        ]
        == 6
    )


def test_complete_score_with_all_evidence():
    result = apply_liquidation_coverage(
        make_quality(),
        make_coverage(),
    )

    assert (
        result.loc[
            0,
            "derivatives_quality_score",
        ]
        == 1.0
    )

    assert (
        result.loc[
            0,
            "derivatives_quality_tier",
        ]
        == "complete"
    )


def test_missing_coverage_defaults_false():
    result = apply_liquidation_coverage(
        make_quality(),
        pd.DataFrame(),
    )

    assert (
        result[
            "liquidation_valid"
        ]
        .eq(False)
        .all()
    )

    assert (
        result[
            "liquidation_coverage_ratio"
        ]
        .eq(0.0)
        .all()
    )
