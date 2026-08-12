"""
Conformance between the declared availability contract in the
feature catalog and what the feature layer actually produces.

The point-in-time availability layer is being rolled out one
feature family at a time. This module keeps the remaining work
visible and makes a half-migrated feature a test failure rather
than a silent correctness gap.
"""

from cryptolab.features.availability import (
    AVAILABLE_AT,
    AVAILABLE_AT_QUALITY,
)
from cryptolab.features.catalog import (
    FEATURE_REGISTRY,
    FEATURE_SPECS,
)


# ================================================================
# MIGRATION DEBT
# ================================================================
#
# Features that do NOT yet emit point-in-time availability.
#
# Each entry is a known gap: research cannot apply the canonical
# anti-lookahead rule to these artifacts yet.
#
# This set is asserted by EXACT equality below, so it cannot grow
# silently. A newly added feature without an availability
# contract fails the suite until it is either migrated or
# explicitly recorded here as debt.
#
# Order of migration: price.* and trade_flow.* carry no
# availability at all, and the remaining derivatives.* features
# join those layers, so they cannot be migrated before them.

PENDING_AVAILABILITY_MIGRATION = frozenset(
    {
        "price.core",
        "price.market_structure",
        "price.structural_levels",
        "price.structural_state",
        "price.quality",
        "price.regime",
        "price.snapshot",
        "trade_flow.aggregate",
        "trade_flow.cvd",
        "trade_flow.large_trade_flags",
        "trade_flow.large_trade_aggregate",
        "trade_flow.regime",
        "trade_flow.snapshot",
        "derivatives.price_oi_state",
        "derivatives.spot_perp_state",
        "derivatives.regime",
        "derivatives.quality",
        "derivatives.liquidation_coverage",
        "derivatives.snapshot",
    }
)


def declares_availability(
    spec,
) -> bool:
    """
    A migrated feature declares the strict metadata pair.

    An availability timestamp with no declared evidence quality
    is not a complete contract, so it does not count as migrated.
    """

    return {
        AVAILABLE_AT,
        AVAILABLE_AT_QUALITY,
    } <= set(
        spec.output_columns
    )


def test_no_feature_declares_half_the_pair():
    """
    available_at and available_at_quality are one contract. A
    feature declaring one without the other would publish a
    timestamp whose evidence cannot be told apart from exact.
    """

    for spec in FEATURE_SPECS:
        declared = {
            AVAILABLE_AT,
            AVAILABLE_AT_QUALITY,
        } & set(
            spec.output_columns
        )

        assert declared in (
            set(),
            {
                AVAILABLE_AT,
                AVAILABLE_AT_QUALITY,
            },
        ), spec.feature_id


def test_migration_debt_is_exactly_enumerated():
    """
    The set of features lacking an availability contract must
    match the recorded debt exactly.

    Fails when a feature is migrated (remove it from the list)
    and when a new feature arrives without availability (migrate
    it or record it).
    """

    undeclared = {
        spec.feature_id
        for spec in FEATURE_SPECS
        if not declares_availability(
            spec
        )
    }

    assert (
        undeclared
        == PENDING_AVAILABILITY_MIGRATION
    )


def test_debt_entries_reference_real_features():
    """
    A stale entry would silently excuse a feature that no longer
    exists while hiding one that does.
    """

    feature_ids = {
        spec.feature_id
        for spec in FEATURE_SPECS
    }

    assert (
        PENDING_AVAILABILITY_MIGRATION
        <= feature_ids
    )


def test_at_least_one_feature_is_migrated():
    """
    Guards against the debt list quietly swallowing everything.
    """

    migrated = {
        spec.feature_id
        for spec in FEATURE_SPECS
        if declares_availability(
            spec
        )
    }

    assert migrated


def test_migrated_features_are_max_input_or_passthrough():
    """
    A feature emitting availability must declare a policy that
    actually defines how it was derived.
    """

    for spec in FEATURE_SPECS:
        if not declares_availability(
            spec
        ):
            continue

        assert spec.availability_policy.value in {
            "max_input_available_at",
            "passthrough",
        }, spec.feature_id


def test_derivatives_core_declares_availability():
    spec = FEATURE_REGISTRY.get(
        "derivatives.core"
    )

    assert declares_availability(
        spec
    )

    assert (
        spec.availability_policy.value
        == "max_input_available_at"
    )


def test_derivatives_core_output_matches_declaration():
    """
    The declared contract must match what the callable actually
    produces, not merely be asserted in metadata.
    """

    import numpy as np
    import pandas as pd

    from cryptolab.features.derivatives import (
        build_derivatives_features,
    )

    times = pd.date_range(
        "2024-01-01",
        periods=12,
        freq="5min",
        tz="UTC",
    )

    open_interest = pd.DataFrame(
        {
            "exchange": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "period": "5m",
            "timestamp": times,
            "open_interest_base": np.linspace(
                100.0,
                110.0,
                12,
            ),
            "open_interest_quote": np.linspace(
                1_000_000.0,
                1_100_000.0,
                12,
            ),
            "available_at": times,
            "available_at_quality": "derived",
        }
    )

    # Funding is supplied because the empty-funding branch has a
    # pre-existing tz-naive/tz-aware defect unrelated to
    # availability propagation (reported as deferred).
    funding = pd.DataFrame(
        {
            "funding_time": [
                times[2],
            ],
            "funding_rate": [
                0.0001,
            ],
            "mark_price": [
                42_000.0,
            ],
            "rate_type": pd.array(
                [
                    "actual",
                ],
                dtype="string",
            ),
            "available_at": [
                times[2],
            ],
            "available_at_quality": [
                "derived",
            ],
        }
    )

    empty = pd.DataFrame()

    result = build_derivatives_features(
        open_interest=open_interest,
        funding_rate=funding,
        basis=empty,
        taker_flow=empty,
        liquidations=empty,
    )

    spec = FEATURE_REGISTRY.get(
        "derivatives.core"
    )

    missing = [
        column
        for column in spec.output_columns
        if column not in result.columns
    ]

    assert not missing
