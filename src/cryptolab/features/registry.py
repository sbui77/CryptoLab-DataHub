from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


class FeatureRegistryError(ValueError):
    """Raised when feature registry metadata is invalid."""


class FeatureKind(str, Enum):
    """
    High-level role of a registered feature-producing unit.
    """

    FEATURE = "feature"
    QUALITY = "quality"
    REGIME = "regime"
    SNAPSHOT = "snapshot"
    AGGREGATION = "aggregation"


class AvailabilityPolicy(str, Enum):
    """
    Point-in-time availability policy for a feature.

    MAX_INPUT_AVAILABLE_AT
        Feature becomes available only when every contributing
        input observation is available.

        Canonical rule:

            feature_available_at
                = max(input available_at)

    PASSTHROUGH
        Output preserves the availability timestamp of the
        canonical input observation.

    CUSTOM
        Feature requires explicitly implemented availability
        semantics that cannot be represented safely by the two
        standard policies.
    """

    MAX_INPUT_AVAILABLE_AT = (
        "max_input_available_at"
    )

    PASSTHROUGH = "passthrough"

    CUSTOM = "custom"


_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)


_FEATURE_ID_PATTERN = re.compile(
    r"^[a-z0-9]+"
    r"(?:[._-][a-z0-9]+)*$"
)


_CALLABLE_PATH_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:"
    r"[A-Za-z_][A-Za-z0-9_]*$"
)


_REQUIRED_PARAMETER_SENTINEL = "<required>"


def _canonical_json_value(
    value: Any,
) -> Any:
    """
    Convert supported runtime parameter values into a stable,
    JSON-serializable canonical representation.

    Tuples and lists intentionally canonicalize to the same
    sequence representation. This prevents a tuple default such
    as:

        (24, 168, 720)

    from producing a different artifact identity than an
    equivalent runtime list:

        [24, 168, 720]
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _canonical_json_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        result = {}

        for key in sorted(
            value,
            key=lambda item: str(
                item
            ),
        ):
            if not isinstance(
                key,
                str,
            ):
                raise FeatureRegistryError(
                    "Feature parameter dictionaries "
                    "must use string keys"
                )

            result[key] = (
                _canonical_json_value(
                    value[key]
                )
            )

        return result

    raise FeatureRegistryError(
        "Unsupported feature parameter value type: "
        f"{type(value).__name__}"
    )


def _parse_default_value(
    default: str,
) -> Any:
    """
    Convert canonical textual parameter defaults into runtime
    values where possible.

    Examples
    --------
    "14"                -> 14
    "0.95"              -> 0.95
    "(24, 168, 720)"    -> (24, 168, 720)

    Plain textual defaults such as:

        "binance"
        "BTCUSDT"
        "1h"

    remain strings.
    """

    if default == _REQUIRED_PARAMETER_SENTINEL:
        return _REQUIRED_PARAMETER_SENTINEL

    try:
        return ast.literal_eval(
            default
        )

    except (
        ValueError,
        SyntaxError,
    ):
        return default


def _stable_hash(
    payload: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Return deterministic SHA-256 fingerprint for metadata.
    """

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


@dataclass(
    frozen=True,
    order=True,
)
class FeatureParameter:
    """
    One declared feature parameter.

    default
    -------
    Canonical textual representation of the default value.

    Use:

        "<required>"

    for parameters with no callable default.
    """

    name: str
    default: str
    description: str = ""

    def __post_init__(
        self,
    ) -> None:
        name = self.name.strip()

        if not name:
            raise FeatureRegistryError(
                "Feature parameter name cannot be empty"
            )

        if name != self.name:
            raise FeatureRegistryError(
                "Feature parameter name cannot contain "
                "leading/trailing whitespace"
            )

        if not isinstance(
            self.default,
            str,
        ):
            raise FeatureRegistryError(
                "Feature parameter default must be a string"
            )

    @property
    def required(
        self,
    ) -> bool:
        return (
            self.default
            == _REQUIRED_PARAMETER_SENTINEL
        )

    def parsed_default(
        self,
    ) -> Any:
        return _parse_default_value(
            self.default
        )


@dataclass(
    frozen=True,
)
class FeatureSpec:
    """
    Canonical metadata contract for one feature-producing unit.

    input_features
    --------------
    Required upstream registered feature IDs.

    optional_input_features
    -----------------------
    Upstream feature IDs that may enrich the feature but are
    not required for valid execution.

    parameters
    ----------
    Parameters affecting computational output.

    Definition and artifact identity
    --------------------------------
    logical identity:
        feature_id@version

    definition fingerprint:
        deterministic hash of the computational metadata
        contract.

    artifact fingerprint:
        definition fingerprint plus resolved runtime parameters.
    """

    feature_id: str
    version: str
    callable_path: str

    kind: FeatureKind
    category: str

    input_datasets: tuple[str, ...]
    input_features: tuple[str, ...]

    output_columns: tuple[str, ...]

    event_time_column: str

    availability_policy: AvailabilityPolicy

    parameters: tuple[
        FeatureParameter,
        ...
    ] = ()

    optional_input_features: tuple[
        str,
        ...
    ] = ()

    description: str = ""

    tags: tuple[str, ...] = ()

    def __post_init__(
        self,
    ) -> None:
        _validate_feature_spec(
            self
        )

    @property
    def identity(
        self,
    ) -> str:
        """
        Stable logical version identity.
        """

        return (
            f"{self.feature_id}@{self.version}"
        )

    @property
    def all_input_features(
        self,
    ) -> tuple[
        str,
        ...
    ]:
        """
        Required and optional feature dependencies.
        """

        return (
            self.input_features
            + self.optional_input_features
        )

    def resolve_parameters(
        self,
        overrides: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Resolve parameter defaults plus runtime overrides.

        Unknown overrides are rejected.

        Parameters declared with default="<required>" must be
        explicitly supplied.
        """

        overrides = (
            {}
            if overrides is None
            else dict(
                overrides
            )
        )

        declared = {
            parameter.name: parameter
            for parameter in self.parameters
        }

        unknown = sorted(
            set(
                overrides
            )
            - set(
                declared
            )
        )

        if unknown:
            raise FeatureRegistryError(
                "Unknown parameter overrides for "
                f"{self.identity}: {unknown}"
            )

        resolved: dict[
            str,
            Any,
        ] = {}

        for parameter in self.parameters:
            if parameter.name in overrides:
                value = overrides[
                    parameter.name
                ]

            else:
                value = (
                    parameter.parsed_default()
                )

                if (
                    value
                    == _REQUIRED_PARAMETER_SENTINEL
                ):
                    raise FeatureRegistryError(
                        "Missing required feature parameter "
                        f"{parameter.name!r} for "
                        f"{self.identity}"
                    )

            resolved[
                parameter.name
            ] = _canonical_json_value(
                value
            )

        return {
            key: resolved[key]
            for key in sorted(
                resolved
            )
        }

    def definition_payload(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Canonical computational definition metadata.

        Description and tags are deliberately excluded because
        editorial metadata must not change computational
        identity.
        """

        return {
            "feature_id": self.feature_id,
            "version": self.version,
            "callable_path": (
                self.callable_path
            ),
            "kind": self.kind.value,
            "category": self.category,
            "input_datasets": list(
                self.input_datasets
            ),
            "input_features": list(
                self.input_features
            ),
            "optional_input_features": list(
                self.optional_input_features
            ),
            "output_columns": list(
                self.output_columns
            ),
            "event_time_column": (
                self.event_time_column
            ),
            "availability_policy": (
                self.availability_policy.value
            ),
            "parameters": [
                {
                    "name": parameter.name,
                    "default": parameter.default,
                }
                for parameter in self.parameters
            ],
        }

    @property
    def definition_fingerprint(
        self,
    ) -> str:
        """
        SHA-256 fingerprint of feature definition metadata.
        """

        return _stable_hash(
            self.definition_payload()
        )

    def artifact_payload(
        self,
        overrides: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Canonical identity payload for one parameterized feature
        artifact.
        """

        return {
            "definition_fingerprint": (
                self.definition_fingerprint
            ),
            "parameters": (
                self.resolve_parameters(
                    overrides
                )
            ),
        }

    def artifact_fingerprint(
        self,
        overrides: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> str:
        """
        Deterministic SHA-256 fingerprint for one resolved
        parameter configuration.
        """

        return _stable_hash(
            self.artifact_payload(
                overrides
            )
        )


def _validate_unique_strings(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    if (
        not allow_empty
        and not values
    ):
        raise FeatureRegistryError(
            f"{field_name} cannot be empty"
        )

    cleaned = []

    for value in values:
        if not isinstance(
            value,
            str,
        ):
            raise FeatureRegistryError(
                f"{field_name} values must be strings"
            )

        if not value:
            raise FeatureRegistryError(
                f"{field_name} cannot contain empty values"
            )

        if value != value.strip():
            raise FeatureRegistryError(
                f"{field_name} values cannot contain "
                "leading/trailing whitespace"
            )

        cleaned.append(
            value
        )

    if len(cleaned) != len(
        set(cleaned)
    ):
        raise FeatureRegistryError(
            f"{field_name} contains duplicates"
        )


def _validate_feature_spec(
    spec: FeatureSpec,
) -> None:
    if not isinstance(
        spec.feature_id,
        str,
    ):
        raise FeatureRegistryError(
            "feature_id must be a string"
        )

    if not _FEATURE_ID_PATTERN.fullmatch(
        spec.feature_id
    ):
        raise FeatureRegistryError(
            "Invalid feature_id: "
            f"{spec.feature_id!r}"
        )

    if not isinstance(
        spec.version,
        str,
    ):
        raise FeatureRegistryError(
            "version must be a string"
        )

    if not _SEMVER_PATTERN.fullmatch(
        spec.version
    ):
        raise FeatureRegistryError(
            "Feature version must use semantic "
            f"versioning: {spec.version!r}"
        )

    if not isinstance(
        spec.callable_path,
        str,
    ):
        raise FeatureRegistryError(
            "callable_path must be a string"
        )

    if not _CALLABLE_PATH_PATTERN.fullmatch(
        spec.callable_path
    ):
        raise FeatureRegistryError(
            "callable_path must use "
            "'module.path:function_name' format"
        )

    if not isinstance(
        spec.kind,
        FeatureKind,
    ):
        raise FeatureRegistryError(
            "kind must be FeatureKind"
        )

    if not isinstance(
        spec.availability_policy,
        AvailabilityPolicy,
    ):
        raise FeatureRegistryError(
            "availability_policy must be "
            "AvailabilityPolicy"
        )

    if (
        not isinstance(
            spec.category,
            str,
        )
        or not spec.category.strip()
    ):
        raise FeatureRegistryError(
            "category cannot be empty"
        )

    if (
        spec.category
        != spec.category.strip()
    ):
        raise FeatureRegistryError(
            "category cannot contain "
            "leading/trailing whitespace"
        )

    if (
        not isinstance(
            spec.event_time_column,
            str,
        )
        or not spec.event_time_column.strip()
    ):
        raise FeatureRegistryError(
            "event_time_column cannot be empty"
        )

    if (
        spec.event_time_column
        != spec.event_time_column.strip()
    ):
        raise FeatureRegistryError(
            "event_time_column cannot contain "
            "leading/trailing whitespace"
        )

    _validate_unique_strings(
        spec.input_datasets,
        "input_datasets",
        allow_empty=True,
    )

    _validate_unique_strings(
        spec.input_features,
        "input_features",
        allow_empty=True,
    )

    _validate_unique_strings(
        spec.optional_input_features,
        "optional_input_features",
        allow_empty=True,
    )

    _validate_unique_strings(
        spec.output_columns,
        "output_columns",
        allow_empty=False,
    )

    _validate_unique_strings(
        spec.tags,
        "tags",
        allow_empty=True,
    )

    dependency_overlap = (
        set(
            spec.input_features
        )
        & set(
            spec.optional_input_features
        )
    )

    if dependency_overlap:
        raise FeatureRegistryError(
            "Feature dependencies cannot be both "
            "required and optional: "
            f"{sorted(dependency_overlap)}"
        )

    if (
        not spec.input_datasets
        and not spec.input_features
        and not spec.optional_input_features
    ):
        raise FeatureRegistryError(
            "Feature must declare at least one "
            "input dataset or input feature"
        )

    parameter_names = [
        parameter.name
        for parameter in spec.parameters
    ]

    if len(
        parameter_names
    ) != len(
        set(
            parameter_names
        )
    ):
        raise FeatureRegistryError(
            "Feature parameters contain duplicate names"
        )


class FeatureRegistry:
    """
    In-memory registry of versioned FeatureSpec objects.

    Registration is explicit.

    Import-time decorators and automatic module scanning are
    deliberately avoided.
    """

    def __init__(
        self,
        specs: Iterable[
            FeatureSpec
        ] = (),
    ) -> None:
        self._specs: dict[
            tuple[str, str],
            FeatureSpec,
        ] = {}

        for spec in specs:
            self.register(
                spec
            )

    def register(
        self,
        spec: FeatureSpec,
    ) -> None:
        if not isinstance(
            spec,
            FeatureSpec,
        ):
            raise FeatureRegistryError(
                "Registry accepts FeatureSpec objects only"
            )

        key = (
            spec.feature_id,
            spec.version,
        )

        if key in self._specs:
            raise FeatureRegistryError(
                "Feature already registered: "
                f"{spec.identity}"
            )

        self._specs[
            key
        ] = spec

    def get(
        self,
        feature_id: str,
        version: str | None = None,
    ) -> FeatureSpec:
        """
        Get one registered feature.

        If version is omitted, the feature ID must resolve to
        exactly one registered version.
        """

        if version is not None:
            key = (
                feature_id,
                version,
            )

            try:
                return self._specs[
                    key
                ]

            except KeyError as exc:
                raise FeatureRegistryError(
                    "Unknown feature: "
                    f"{feature_id}@{version}"
                ) from exc

        matches = [
            spec
            for (
                registered_id,
                _,
            ), spec in self._specs.items()
            if registered_id
            == feature_id
        ]

        if not matches:
            raise FeatureRegistryError(
                "Unknown feature: "
                f"{feature_id}"
            )

        if len(matches) != 1:
            versions = sorted(
                spec.version
                for spec in matches
            )

            raise FeatureRegistryError(
                "Feature has multiple registered "
                "versions; specify version explicitly: "
                f"{feature_id} -> {versions}"
            )

        return matches[0]

    def contains(
        self,
        feature_id: str,
        version: str | None = None,
    ) -> bool:
        if version is not None:
            return (
                feature_id,
                version,
            ) in self._specs

        return any(
            registered_id
            == feature_id
            for (
                registered_id,
                _,
            ) in self._specs
        )

    def list_specs(
        self,
    ) -> tuple[
        FeatureSpec,
        ...
    ]:
        return tuple(
            sorted(
                self._specs.values(),
                key=lambda spec: (
                    spec.feature_id,
                    spec.version,
                ),
            )
        )

    def list_feature_ids(
        self,
    ) -> tuple[
        str,
        ...
    ]:
        return tuple(
            sorted(
                {
                    feature_id
                    for (
                        feature_id,
                        _,
                    ) in self._specs
                }
            )
        )

    def by_category(
        self,
        category: str,
    ) -> tuple[
        FeatureSpec,
        ...
    ]:
        return tuple(
            spec
            for spec in self.list_specs()
            if spec.category
            == category
        )

    def by_kind(
        self,
        kind: FeatureKind,
    ) -> tuple[
        FeatureSpec,
        ...
    ]:
        return tuple(
            spec
            for spec in self.list_specs()
            if spec.kind
            == kind
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._specs
        )
