from __future__ import annotations

import pytest

from cryptolab.features.registry import (
    AvailabilityPolicy,
    FeatureKind,
    FeatureParameter,
    FeatureRegistry,
    FeatureRegistryError,
    FeatureSpec,
)


def _price_spec(
    version: str = "1.0.0",
) -> FeatureSpec:
    return FeatureSpec(
        feature_id=(
            "price.core"
        ),
        version=version,
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
            "return_1",
            "true_range",
        ),
        event_time_column=(
            "open_time"
        ),
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
        parameters=(
            FeatureParameter(
                name="timeframe",
                default="1h",
            ),
        ),
        description=(
            "Canonical price feature builder."
        ),
        tags=(
            "market",
            "price",
        ),
    )


def test_feature_spec_identity():
    spec = _price_spec()

    assert (
        spec.identity
        == "price.core@1.0.0"
    )


def test_registry_register_and_get():
    spec = _price_spec()

    registry = FeatureRegistry()

    registry.register(
        spec
    )

    assert len(
        registry
    ) == 1

    assert registry.get(
        "price.core"
    ) == spec

    assert registry.get(
        "price.core",
        "1.0.0",
    ) == spec


def test_registry_rejects_duplicate_version():
    spec = _price_spec()

    registry = FeatureRegistry(
        [
            spec,
        ]
    )

    with pytest.raises(
        FeatureRegistryError
    ):
        registry.register(
            spec
        )


def test_multiple_versions_require_explicit_version():
    registry = FeatureRegistry(
        [
            _price_spec(
                "1.0.0"
            ),
            _price_spec(
                "1.1.0"
            ),
        ]
    )

    with pytest.raises(
        FeatureRegistryError
    ):
        registry.get(
            "price.core"
        )

    assert (
        registry.get(
            "price.core",
            "1.1.0",
        ).version
        == "1.1.0"
    )


def test_registry_category_filter():
    spec = _price_spec()

    registry = FeatureRegistry(
        [
            spec,
        ]
    )

    assert registry.by_category(
        "price"
    ) == (
        spec,
    )

    assert registry.by_category(
        "derivatives"
    ) == ()


def test_registry_kind_filter():
    spec = _price_spec()

    registry = FeatureRegistry(
        [
            spec,
        ]
    )

    assert registry.by_kind(
        FeatureKind.FEATURE
    ) == (
        spec,
    )

    assert registry.by_kind(
        FeatureKind.REGIME
    ) == ()


def test_invalid_semver_rejected():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="price.core",
            version="v1",
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
                "return_1",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )


def test_invalid_feature_id_rejected():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="Price Core",
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
                "return_1",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )


def test_feature_requires_dependency():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="price.core",
            version="1.0.0",
            callable_path=(
                "cryptolab.features.price:"
                "build_price_features"
            ),
            kind=FeatureKind.FEATURE,
            category="price",
            input_datasets=(),
            input_features=(),
            output_columns=(
                "return_1",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )


def test_output_columns_cannot_be_empty():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="price.core",
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
            output_columns=(),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )


def test_duplicate_output_columns_rejected():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="price.core",
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
                "return_1",
                "return_1",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
        )


def test_duplicate_parameter_names_rejected():
    with pytest.raises(
        FeatureRegistryError
    ):
        FeatureSpec(
            feature_id="price.core",
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
                "return_1",
            ),
            event_time_column="open_time",
            availability_policy=(
                AvailabilityPolicy
                .MAX_INPUT_AVAILABLE_AT
            ),
            parameters=(
                FeatureParameter(
                    name="window",
                    default="20",
                ),
                FeatureParameter(
                    name="window",
                    default="30",
                ),
            ),
        )


def test_contains():
    registry = FeatureRegistry(
        [
            _price_spec(),
        ]
    )

    assert registry.contains(
        "price.core"
    )

    assert registry.contains(
        "price.core",
        "1.0.0",
    )

    assert not registry.contains(
        "price.core",
        "9.9.9",
    )


def test_list_feature_ids_is_stable_and_sorted():
    first = FeatureSpec(
        feature_id="trade_flow.core",
        version="1.0.0",
        callable_path=(
            "cryptolab.features.trade_flow:"
            "add_cvd_features"
        ),
        kind=FeatureKind.FEATURE,
        category="trade_flow",
        input_datasets=(
            "binance.aggtrades",
        ),
        input_features=(),
        output_columns=(
            "base_cvd",
        ),
        event_time_column="open_time",
        availability_policy=(
            AvailabilityPolicy
            .MAX_INPUT_AVAILABLE_AT
        ),
    )

    second = _price_spec()

    registry = FeatureRegistry(
        [
            first,
            second,
        ]
    )

    assert registry.list_feature_ids() == (
        "price.core",
        "trade_flow.core",
    )
