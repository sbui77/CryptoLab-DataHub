from __future__ import annotations

import importlib

from cryptolab.features.catalog import (
    FEATURE_REGISTRY,
    FEATURE_SPECS,
)
from cryptolab.features.registry import (
    FeatureKind,
)


EXPECTED_FEATURE_IDS = {
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

    "derivatives.core",
    "derivatives.price_oi_state",
    "derivatives.spot_perp_state",
    "derivatives.regime",
    "derivatives.quality",
    "derivatives.liquidation_coverage",
    "derivatives.snapshot",
}


def _resolve_callable(
    callable_path: str,
):
    module_name, function_name = (
        callable_path.split(
            ":",
            1,
        )
    )

    module = importlib.import_module(
        module_name
    )

    return getattr(
        module,
        function_name,
    )


def test_catalog_contains_expected_feature_ids():
    assert set(
        FEATURE_REGISTRY.list_feature_ids()
    ) == EXPECTED_FEATURE_IDS


def test_catalog_has_twenty_entries():
    assert len(
        FEATURE_SPECS
    ) == 20

    assert len(
        FEATURE_REGISTRY
    ) == 20


def test_every_callable_path_resolves():
    for spec in FEATURE_SPECS:
        function = _resolve_callable(
            spec.callable_path
        )

        assert callable(
            function
        )


def test_every_feature_dependency_exists():
    feature_ids = set(
        FEATURE_REGISTRY.list_feature_ids()
    )

    for spec in FEATURE_SPECS:
        for dependency in spec.input_features:
            assert dependency in feature_ids, (
                f"{spec.feature_id} depends on "
                f"unknown feature {dependency}"
            )


def test_no_feature_depends_on_itself():
    for spec in FEATURE_SPECS:
        assert (
            spec.feature_id
            not in spec.input_features
        )


def test_snapshot_entries_are_explicit_read_models():
    snapshot_ids = {
        spec.feature_id
        for spec in FEATURE_REGISTRY.by_kind(
            FeatureKind.SNAPSHOT
        )
    }

    assert snapshot_ids == {
        "price.snapshot",
        "trade_flow.snapshot",
        "derivatives.snapshot",
    }


def test_registry_dependency_graph_is_acyclic():
    graph = {
        spec.feature_id: set(
            spec.input_features
        )
        for spec in FEATURE_SPECS
    }

    visiting = set()
    visited = set()

    def visit(
        node: str,
    ) -> None:
        if node in visited:
            return

        assert node not in visiting, (
            f"Feature dependency cycle detected "
            f"at {node}"
        )

        visiting.add(
            node
        )

        for dependency in graph[
            node
        ]:
            visit(
                dependency
            )

        visiting.remove(
            node
        )

        visited.add(
            node
        )

    for feature_id in graph:
        visit(
            feature_id
        )


def test_derivatives_core_declares_all_raw_streams():
    spec = FEATURE_REGISTRY.get(
        "derivatives.core"
    )

    assert set(
        spec.input_datasets
    ) == {
        "binance.open_interest",
        "binance.funding_rate",
        "binance.basis",
        "binance.taker_flow",
        "binance.liquidations",
    }


def test_price_regime_declares_structural_dependencies():
    spec = FEATURE_REGISTRY.get(
        "price.regime"
    )

    assert set(
        spec.input_features
    ) == {
        "price.core",
        "price.structural_levels",
        "price.structural_state",
    }


def test_derivatives_regime_dependency_chain():
    spec = FEATURE_REGISTRY.get(
        "derivatives.regime"
    )

    assert set(
        spec.input_features
    ) == {
        "derivatives.price_oi_state",
        "derivatives.spot_perp_state",
        "derivatives.core",
    }


# Features whose published output contract has moved beyond the
# initial 1.0.0 release, with the reason for each bump.
#
# A minor bump means the output contract gained columns without
# changing the meaning of existing ones.
EXPECTED_VERSION_OVERRIDES = {
    # Emits available_at (point-in-time availability layer).
    "derivatives.core": "1.1.0",
}


def test_catalog_versions_match_expected_bumps():
    """
    Every feature sits at 1.0.0 unless an intentional bump is
    recorded above.

    This asserts the intended version mapping rather than mere
    semver well-formedness, which FeatureSpec already enforces.
    """

    expected = {
        spec.feature_id: (
            EXPECTED_VERSION_OVERRIDES.get(
                spec.feature_id,
                "1.0.0",
            )
        )
        for spec in FEATURE_SPECS
    }

    actual = {
        spec.feature_id: spec.version
        for spec in FEATURE_SPECS
    }

    assert actual == expected


def test_version_overrides_reference_real_features():
    """
    A stale override would silently stop asserting anything.
    """

    feature_ids = {
        spec.feature_id
        for spec in FEATURE_SPECS
    }

    assert set(
        EXPECTED_VERSION_OVERRIDES
    ) <= feature_ids
