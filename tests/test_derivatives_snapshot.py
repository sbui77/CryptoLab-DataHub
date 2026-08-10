from __future__ import annotations

import pandas as pd

from cryptolab.features.derivatives_snapshot import (
    build_derivatives_snapshot,
    derivatives_snapshot_to_dataframe,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-08-10T04:00:00Z",
                    "2026-08-10T05:00:00Z",
                    "2026-08-10T06:00:00Z",
                ],
                utc=True,
            ),

            "as_of_time": pd.to_datetime(
                [
                    "2026-08-10T05:00:00Z",
                    "2026-08-10T06:00:00Z",
                    "2026-08-10T07:00:00Z",
                ],
                utc=True,
            ),

            "close": [
                64_000.0,
                64_500.0,
                65_000.0,
            ],

            "price_return_1h": [
                0.001,
                0.002,
                0.003,
            ],

            "oi_quote_change_pct_1h": [
                0.002,
                0.003,
                0.004,
            ],

            "spot_delta_pct_1h": [
                0.05,
                0.10,
                0.20,
            ],

            "perp_delta_pct_1h": [
                0.06,
                0.12,
                0.25,
            ],

            "funding_time": pd.to_datetime(
                [
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T00:00:00Z",
                ],
                utc=True,
            ),

            "funding_rate": [
                0.0001,
                0.0001,
                0.0001,
            ],

            "basis_rate": [
                0.0002,
                0.0002,
                0.0003,
            ],

            "long_liquidation_notional_1h": [
                0.0,
                0.0,
                0.0,
            ],

            "short_liquidation_notional_1h": [
                0.0,
                0.0,
                0.0,
            ],

            "liquidation_imbalance_1h": [
                0.0,
                0.0,
                0.0,
            ],

            "price_oi_state": [
                "neutral",
                "position_build_up",
                "position_build_up",
            ],

            "derivatives_regime": [
                "balanced",
                "leveraged_long_build",
                "leveraged_long_build",
            ],

            "derivatives_regime_confidence": [
                1.0,
                0.75,
                0.75,
            ],

            "derivatives_quality_score": [
                0.75,
                0.95,
                0.60,
            ],

            "derivatives_quality_tier": [
                "good",
                "complete",
                "partial",
            ],

            "derivatives_regime_evidence_count": [
                4,
                5,
                3,
            ],

            "oi_valid": [
                True,
                True,
                True,
            ],

            "funding_valid": [
                True,
                True,
                False,
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
                False,
                True,
                False,
            ],

            "liquidation_valid": [
                False,
                False,
                False,
            ],

            "derivatives_core_valid": [
                True,
                True,
                False,
            ],

            "derivatives_regime_quality_valid": [
                True,
                True,
                False,
            ],
        }
    )


def test_selects_latest_quality_valid_row():
    snapshot = build_derivatives_snapshot(
        make_df()
    )

    assert (
        snapshot.open_time
        == pd.Timestamp(
            "2026-08-10T05:00:00Z"
        )
    )

    assert (
        snapshot.as_of_time
        == pd.Timestamp(
            "2026-08-10T06:00:00Z"
        )
    )


def test_does_not_use_newer_invalid_row():
    snapshot = build_derivatives_snapshot(
        make_df()
    )

    assert (
        snapshot.derivatives_quality_score
        == 0.95
    )

    assert (
        snapshot.derivatives_quality_tier
        == "complete"
    )


def test_snapshot_regime():
    snapshot = build_derivatives_snapshot(
        make_df()
    )

    assert (
        snapshot.derivatives_regime
        == "leveraged_long_build"
    )

    assert (
        snapshot.derivatives_regime_quality_valid
        is True
    )


def test_snapshot_evidence_masks():
    snapshot = build_derivatives_snapshot(
        make_df()
    )

    assert snapshot.oi_valid is True
    assert snapshot.funding_valid is True
    assert snapshot.basis_valid is True
    assert snapshot.taker_flow_valid is True
    assert snapshot.spot_perp_valid is True
    assert snapshot.liquidation_valid is False


def test_snapshot_to_dataframe():
    snapshot = build_derivatives_snapshot(
        make_df()
    )

    frame = derivatives_snapshot_to_dataframe(
        snapshot
    )

    assert len(frame) == 1

    assert (
        frame.loc[
            0,
            "derivatives_regime",
        ]
        == "leveraged_long_build"
    )


def test_no_quality_valid_rows_raises():
    df = make_df()

    df[
        "derivatives_regime_quality_valid"
    ] = False

    try:
        build_derivatives_snapshot(
            df
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
