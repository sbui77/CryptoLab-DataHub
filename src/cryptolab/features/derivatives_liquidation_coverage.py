from __future__ import annotations

import numpy as np
import pandas as pd


class DerivativesLiquidationCoverageError(
    RuntimeError
):
    """Raised when liquidation coverage integration fails."""


EVIDENCE_COLUMNS = [
    "oi_valid",
    "funding_valid",
    "basis_valid",
    "taker_flow_valid",
    "spot_perp_valid",
    "liquidation_valid",
]


def apply_liquidation_coverage(
    quality_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate liquidation collector coverage into the
    derivatives quality dataset.

    Contract
    --------
    covered + zero liquidation:
        valid zero-liquidation observation

    covered + observed liquidation:
        valid liquidation observation

    not covered:
        liquidation evidence unknown / invalid

    This function deliberately does NOT reinterpret missing
    collector coverage as zero liquidation.
    """

    if quality_df.empty:
        raise DerivativesLiquidationCoverageError(
            "Derivatives quality dataframe is empty"
        )

    required_quality = [
        "open_time",
        "long_liquidation_notional_1h",
        "short_liquidation_notional_1h",
        "total_liquidation_notional_1h",
        "oi_valid",
        "funding_valid",
        "basis_valid",
        "taker_flow_valid",
        "spot_perp_valid",
        "derivatives_regime_valid",
        "derivatives_core_valid",
    ]

    missing = [
        column
        for column in required_quality
        if column not in quality_df.columns
    ]

    if missing:
        raise DerivativesLiquidationCoverageError(
            "Missing quality columns: "
            f"{missing}"
        )

    result = quality_df.copy()

    result["open_time"] = pd.to_datetime(
        result["open_time"],
        utc=True,
        errors="coerce",
    )

    if result["open_time"].isna().any():
        raise DerivativesLiquidationCoverageError(
            "Invalid quality open_time"
        )

    # ========================================================
    # COVERAGE
    # ========================================================

    if coverage_df.empty:
        result[
            "liquidation_coverage_ratio"
        ] = 0.0

        result[
            "liquidation_collector_covered"
        ] = False

    else:
        required_coverage = [
            "open_time",
            "collector_coverage_ratio",
            "liquidation_collector_covered",
        ]

        missing_coverage = [
            column
            for column in required_coverage
            if column not in coverage_df.columns
        ]

        if missing_coverage:
            raise (
                DerivativesLiquidationCoverageError(
                    "Missing coverage columns: "
                    f"{missing_coverage}"
                )
            )

        coverage = coverage_df[
            required_coverage
        ].copy()

        coverage["open_time"] = pd.to_datetime(
            coverage["open_time"],
            utc=True,
            errors="coerce",
        )

        coverage = (
            coverage
            .sort_values("open_time")
            .drop_duplicates(
                subset=["open_time"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        result = result.merge(
            coverage,
            on="open_time",
            how="left",
        )

        result[
            "liquidation_coverage_ratio"
        ] = (
            result[
                "collector_coverage_ratio"
            ]
            .fillna(0.0)
            .astype("float64")
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

        result[
            "liquidation_collector_covered"
        ] = (
            result[
                "liquidation_collector_covered"
            ]
            .fillna(False)
            .astype(bool)
        )

        result = result.drop(
            columns=[
                "collector_coverage_ratio",
            ]
        )

    # ========================================================
    # LIQUIDATION VALUES
    # ========================================================

    result[
        "liquidation_value_available"
    ] = (
        result[
            "total_liquidation_notional_1h"
        ].notna()
    )

    result[
        "liquidation_observed"
    ] = (
        result[
            "total_liquidation_notional_1h"
        ]
        > 0
    )

    result[
        "liquidation_values_valid"
    ] = (
        result[
            "liquidation_value_available"
        ]
        & (
            result[
                "long_liquidation_notional_1h"
            ]
            >= 0
        )
        & (
            result[
                "short_liquidation_notional_1h"
            ]
            >= 0
        )
        & (
            result[
                "total_liquidation_notional_1h"
            ]
            >= 0
        )
    )

    # ========================================================
    # NEW CONTRACT
    #
    # Coverage, not event occurrence, determines whether the
    # liquidation observation is trustworthy.
    # ========================================================

    result[
        "liquidation_valid"
    ] = (
        result[
            "liquidation_values_valid"
        ]
        & result[
            "liquidation_collector_covered"
        ]
    )

    # ========================================================
    # RECOMPUTE EVIDENCE COUNT
    # ========================================================

    for column in EVIDENCE_COLUMNS:
        result[column] = (
            result[column]
            .fillna(False)
            .astype(bool)
        )

    result[
        "derivatives_regime_evidence_count"
    ] = (
        result[
            EVIDENCE_COLUMNS
        ]
        .astype("int64")
        .sum(axis=1)
    )

    result[
        "derivatives_regime_evidence_total"
    ] = len(
        EVIDENCE_COLUMNS
    )

    # ========================================================
    # RECOMPUTE QUALITY SCORE
    #
    # Preserve the existing weights:
    # OI          0.25
    # Funding     0.15
    # Basis       0.15
    # Taker Flow  0.20
    # Spot×Perp   0.20
    # Liquidation 0.05
    # ========================================================

    result[
        "derivatives_quality_score"
    ] = (
        result[
            "oi_valid"
        ].astype(float)
        * 0.25
        + result[
            "funding_valid"
        ].astype(float)
        * 0.15
        + result[
            "basis_valid"
        ].astype(float)
        * 0.15
        + result[
            "taker_flow_valid"
        ].astype(float)
        * 0.20
        + result[
            "spot_perp_valid"
        ].astype(float)
        * 0.20
        + result[
            "liquidation_valid"
        ].astype(float)
        * 0.05
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    quality_tier = pd.Series(
        ["insufficient"] * len(result),
        index=result.index,
        dtype="string",
    )

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ]
        >= 0.50
    ] = "partial"

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ]
        >= 0.75
    ] = "good"

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ]
        >= 0.95
    ] = "complete"

    result[
        "derivatives_quality_tier"
    ] = quality_tier

    # ========================================================
    # REGIME QUALITY CONTRACT
    #
    # Core validity remains independent of liquidation.
    # ========================================================

    result[
        "derivatives_regime_quality_valid"
    ] = (
        result[
            "derivatives_regime_valid"
        ]
        .fillna(False)
        .astype(bool)
        & result[
            "derivatives_core_valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    result.loc[
        ~result[
            "derivatives_regime_quality_valid"
        ],
        "derivatives_regime_confidence",
    ] = np.nan

    return (
        result
        .sort_values("open_time")
        .reset_index(drop=True)
    )
