from __future__ import annotations

import numpy as np
import pandas as pd


class DerivativesQualityError(ValueError):
    """Raised when derivatives quality-mask generation fails."""


DEFAULT_FUNDING_MAX_AGE_HOURS = 12.0
DEFAULT_SPOT_PERP_MIN_VALID_SHARE = 0.80


def add_derivatives_quality_masks(
    df: pd.DataFrame,
    funding_max_age_hours: float = (
        DEFAULT_FUNDING_MAX_AGE_HOURS
    ),
    spot_perp_min_valid_share: float = (
        DEFAULT_SPOT_PERP_MIN_VALID_SHARE
    ),
) -> pd.DataFrame:
    """
    Add evidence-level validity masks.

    Time semantics
    --------------
    open_time:
        start of the 1h bucket.

    as_of_time:
        timestamp at which the complete bucket is known.

    Freshness and causality checks must use as_of_time.
    """

    if df.empty:
        raise DerivativesQualityError(
            "Derivatives regime dataframe is empty"
        )

    if funding_max_age_hours <= 0:
        raise DerivativesQualityError(
            "funding_max_age_hours must be positive"
        )

    if not (
        0.0
        <= spot_perp_min_valid_share
        <= 1.0
    ):
        raise DerivativesQualityError(
            "spot_perp_min_valid_share must be "
            "between 0 and 1"
        )

    required = [
        "open_time",
        "as_of_time",

        "open_interest_quote",
        "oi_quote_change_pct_1h",

        "funding_time",
        "funding_rate",

        "basis_rate",

        "spot_quote_volume_1h",
        "spot_delta_pct_1h",

        "perp_total_volume_1h",
        "perp_delta_pct_1h",

        "spot_perp_valid_share_1h",

        "long_liquidation_notional_1h",
        "short_liquidation_notional_1h",
        "total_liquidation_notional_1h",

        "price_oi_state_valid",

        "derivatives_regime",
        "derivatives_regime_valid",
        "derivatives_regime_confidence",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise DerivativesQualityError(
            "Missing derivatives quality columns: "
            f"{missing}"
        )

    result = (
        df.copy()
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    for column in [
        "open_time",
        "as_of_time",
        "funding_time",
    ]:
        result[column] = pd.to_datetime(
            result[column],
            utc=True,
            errors="coerce",
        )

    if result["open_time"].isna().any():
        raise DerivativesQualityError(
            "Invalid open_time"
        )

    if result["as_of_time"].isna().any():
        raise DerivativesQualityError(
            "Invalid as_of_time"
        )

    expected_as_of = (
        result["open_time"]
        + pd.Timedelta(hours=1)
    )

    if not (
        result["as_of_time"]
        .eq(expected_as_of)
        .all()
    ):
        raise DerivativesQualityError(
            "Invalid hourly as_of_time contract"
        )

    # ========================================================
    # OI
    # ========================================================

    result["oi_available"] = (
        result[
            "open_interest_quote"
        ].notna()
    )

    result["oi_valid"] = (
        result["oi_available"]
        & (
            result[
                "open_interest_quote"
            ] > 0
        )
        & result[
            "oi_quote_change_pct_1h"
        ].notna()
        & result[
            "price_oi_state_valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    # ========================================================
    # FUNDING
    # ========================================================

    result[
        "funding_age_hours"
    ] = (
        (
            result["as_of_time"]
            - result["funding_time"]
        )
        .dt.total_seconds()
        / 3600.0
    )

    result[
        "funding_available"
    ] = (
        result["funding_rate"].notna()
        & result["funding_time"].notna()
    )

    result[
        "funding_causal"
    ] = (
        result[
            "funding_available"
        ]
        & (
            result[
                "funding_time"
            ]
            <= result[
                "as_of_time"
            ]
        )
    )

    result[
        "funding_fresh"
    ] = (
        result[
            "funding_causal"
        ]
        & (
            result[
                "funding_age_hours"
            ] >= 0
        )
        & (
            result[
                "funding_age_hours"
            ]
            <= funding_max_age_hours
        )
    )

    result["funding_valid"] = (
        result[
            "funding_available"
        ]
        & result[
            "funding_causal"
        ]
        & result[
            "funding_fresh"
        ]
    )

    # ========================================================
    # BASIS
    # ========================================================

    result["basis_available"] = (
        result[
            "basis_rate"
        ].notna()
    )

    result["basis_valid"] = (
        result["basis_available"]
        & np.isfinite(
            result[
                "basis_rate"
            ]
        )
    )

    # ========================================================
    # FUTURES TAKER FLOW
    # ========================================================

    result[
        "taker_flow_available"
    ] = (
        result[
            "perp_total_volume_1h"
        ].notna()
        & result[
            "perp_delta_pct_1h"
        ].notna()
    )

    result[
        "taker_flow_valid"
    ] = (
        result[
            "taker_flow_available"
        ]
        & (
            result[
                "perp_total_volume_1h"
            ] > 0
        )
        & result[
            "perp_delta_pct_1h"
        ].between(
            -1.0,
            1.0,
        )
    )

    # ========================================================
    # SPOT
    # ========================================================

    result[
        "spot_flow_available"
    ] = (
        result[
            "spot_quote_volume_1h"
        ].notna()
        & result[
            "spot_delta_pct_1h"
        ].notna()
    )

    result[
        "spot_flow_valid"
    ] = (
        result[
            "spot_flow_available"
        ]
        & (
            result[
                "spot_quote_volume_1h"
            ] > 0
        )
        & result[
            "spot_delta_pct_1h"
        ].between(
            -1.0,
            1.0,
        )
    )

    # ========================================================
    # SPOT × PERP
    # ========================================================

    result[
        "spot_perp_available"
    ] = (
        result[
            "spot_perp_valid_share_1h"
        ].notna()
    )

    result[
        "spot_perp_valid"
    ] = (
        result[
            "spot_perp_available"
        ]
        & (
            result[
                "spot_perp_valid_share_1h"
            ]
            >= spot_perp_min_valid_share
        )
        & result[
            "spot_flow_valid"
        ]
        & result[
            "taker_flow_valid"
        ]
    )

    # ========================================================
    # LIQUIDATION
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
            ] >= 0
        )
        & (
            result[
                "short_liquidation_notional_1h"
            ] >= 0
        )
        & (
            result[
                "total_liquidation_notional_1h"
            ] >= 0
        )
    )

    result[
        "liquidation_valid"
    ] = (
        result[
            "liquidation_values_valid"
        ]
        & result[
            "liquidation_observed"
        ]
    )

    # ========================================================
    # CORE
    # ========================================================

    result[
        "derivatives_core_valid"
    ] = (
        result["oi_valid"]
        & result["funding_valid"]
        & result["basis_valid"]
        & result["taker_flow_valid"]
    )

    evidence_columns = [
        "oi_valid",
        "funding_valid",
        "basis_valid",
        "taker_flow_valid",
        "spot_perp_valid",
        "liquidation_valid",
    ]

    result[
        "derivatives_regime_evidence_count"
    ] = (
        result[
            evidence_columns
        ]
        .astype("int64")
        .sum(axis=1)
    )

    result[
        "derivatives_regime_evidence_total"
    ] = len(
        evidence_columns
    )

    result[
        "derivatives_quality_score"
    ] = (
        result["oi_valid"].astype(float)
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
    )

    result[
        "derivatives_quality_score"
    ] = (
        result[
            "derivatives_quality_score"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    quality_tier = pd.Series(
        ["insufficient"] * len(result),
        index=result.index,
        dtype="string",
    )

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ] >= 0.50
    ] = "partial"

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ] >= 0.75
    ] = "good"

    quality_tier.loc[
        result[
            "derivatives_quality_score"
        ] >= 0.95
    ] = "complete"

    result[
        "derivatives_quality_tier"
    ] = quality_tier

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
    )

    result.loc[
        ~result[
            "derivatives_regime_quality_valid"
        ],
        "derivatives_regime_confidence",
    ] = np.nan

    return result
