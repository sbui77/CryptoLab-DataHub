from __future__ import annotations

import re

import pytest

from cryptolab.features.catalog import (
    FEATURE_REGISTRY,
)
from cryptolab.features.registry import (
    AvailabilityPolicy,
    FeatureKind,
    FeatureParameter,
    FeatureRegistryError,
    FeatureSpec,
)


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


def test_price_core_defaults_resolve_to_runtime_values():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    resolved = spec.resolve_parameters()

    assert resolved == {
        "atr_window": 14,
        "rolling_windows": [
            24,
            168,
            720,
        ],
    }


def test_required_parameter_must_be_supplied():
    spec = FEATURE_REGISTRY.get(
        "trade_flow.aggregate"
    )

    with pytest.raises(
        FeatureRegistryError
    ):
        spec.resolve_parameters()


def test_required_parameter_can_be_resolved():
    spec = FEATURE_REGISTRY.get(
        "trade_flow.aggregate"
    )

    resolved = spec.resolve_parameters(
        {
            "timeframe": "5m",
        }
    )

    assert resolved == {
        "timeframe": "5m",
    }


def test_unknown_parameter_override_rejected():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    with pytest.raises(
        FeatureRegistryError
    ):
        spec.resolve_parameters(
            {
                "not_a_parameter": 1,
            }
        )


def test_default_and_equivalent_overrides_have_same_artifact():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    default_fingerprint = (
        spec.artifact_fingerprint()
    )

    explicit_fingerprint = (
        spec.artifact_fingerprint(
            {
                "atr_window": 14,
                "rolling_windows": (
                    24,
                    168,
                    720,
                ),
            }
        )
    )

    list_fingerprint = (
        spec.artifact_fingerprint(
            {
                "atr_window": 14,
                "rolling_windows": [
                    24,
                    168,
                    720,
                ],
            }
        )
    )

    assert (
        default_fingerprint
        == explicit_fingerprint
        == list_fingerprint
    )


def test_parameter_change_changes_artifact_fingerprint():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    baseline = (
        spec.artifact_fingerprint()
    )

    changed = (
        spec.artifact_fingerprint(
            {
                "atr_window": 21,
            }
        )
    )

    assert (
        baseline
        != changed
    )


def test_parameter_change_does_not_change_definition_fingerprint():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    before = (
        spec.definition_fingerprint
    )

    spec.artifact_fingerprint(
        {
            "atr_window": 21,
        }
    )

    after = (
        spec.definition_fingerprint
    )

    assert before == after


def test_definition_fingerprint_is_sha256():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    assert SHA256_PATTERN.fullmatch(
        spec.definition_fingerprint
    )


def test_artifact_fingerprint_is_sha256():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    assert SHA256_PATTERN.fullmatch(
        spec.artifact_fingerprint()
    )


def test_definition_fingerprint_is_deterministic():
    spec = FEATURE_REGISTRY.get(
        "price.core"
    )

    assert (
        spec.definition_fingerprint
        == spec.definition_fingerprint
    )


def test_metadata_description_does_not_change_definition_identity():
    first = FeatureSpec(
        feature_id="example.feature",
        version="1.0.0",
        callable_path=(
            "cryptolab.features.price:"
            "build_price_features"
        ),
        kind=FeatureKind.FEATURE,
        category="price",
        input_datasets=(
            "binance.ohlcv",
        ),
        input_features=(),
        output_columns=(
            "simple_return",
        ),
        event_time_column="open_time",
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
        parameters=(
            FeatureParameter(
                name="atr_window",
                default="14",
            ),
        ),
        description="First description",
        tags=(
            "one",
        ),
    )

    second = FeatureSpec(
        feature_id="example.feature",
        version="1.0.0",
        callable_path=(
            "cryptolab.features.price:"
            "build_price_features"
        ),
        kind=FeatureKind.FEATURE,
        category="price",
        input_datasets=(
            "binance.ohlcv",
        ),
        input_features=(),
        output_columns=(
            "simple_return",
        ),
        event_time_column="open_time",
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
        parameters=(
            FeatureParameter(
                name="atr_window",
                default="14",
            ),
        ),
        description="Different editorial description",
        tags=(
            "different",
        ),
    )

    assert (
        first.definition_fingerprint
        == second.definition_fingerprint
    )


def test_computational_change_changes_definition_identity():
    first = FeatureSpec(
        feature_id="example.feature",
        version="1.0.0",
        callable_path=(
            "cryptolab.features.price:"
            "build_price_features"
        ),
        kind=FeatureKind.FEATURE,
        category="price",
        input_datasets=(
            "binance.ohlcv",
        ),
        input_features=(),
        output_columns=(
            "simple_return",
        ),
        event_time_column="open_time",
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
        parameters=(
            FeatureParameter(
                name="window",
                default="14",
            ),
        ),
    )

    second = FeatureSpec(
        feature_id="example.feature",
        version="1.0.0",
        callable_path=(
            "cryptolab.features.price:"
            "build_price_features"
        ),
        kind=FeatureKind.FEATURE,
        category="price",
        input_datasets=(
            "binance.ohlcv",
        ),
        input_features=(),
        output_columns=(
            "simple_return",
        ),
        event_time_column="open_time",
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
        parameters=(
            FeatureParameter(
                name="window",
                default="21",
            ),
        ),
    )

    assert (
        first.definition_fingerprint
        != second.definition_fingerprint
    )


def test_trade_flow_large_feature_dependency_is_optional():
    spec = FEATURE_REGISTRY.get(
        "trade_flow.regime"
    )

    assert spec.input_features == (
        "trade_flow.aggregate",
        "trade_flow.cvd",
    )

    assert (
        spec.optional_input_features
        == (
            "trade_flow.large_trade_aggregate",
        )
    )

    assert (
        "trade_flow.large_trade_aggregate"
        in spec.all_input_features
    )


def test_required_optional_dependency_overlap_rejected():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="example.feature",
            version="1.0.0",
            callable_path=(
                "cryptolab.features.price:"
                "build_price_features"
            ),
            kind=FeatureKind.FEATURE,
            category="price",
            input_datasets=(),
            input_features=(
                "price.core",
            ),
            optional_input_features=(
                "price.core",
            ),
            output_columns=(
                "example",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )
