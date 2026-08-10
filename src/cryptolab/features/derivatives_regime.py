from __future__ import annotations

import numpy as np
import pandas as pd


class DerivativesRegimeError(ValueError):
    """Raised when derivatives regime generation fails."""


DEFAULT_FLOW_THRESHOLD = 0.10
DEFAULT_LIQUIDATION_IMBALANCE_THRESHOLD = 0.50

LIQUIDATION_LOOKBACK_HOURS = 24
LIQUIDATION_QUANTILE = 0.90


def aggregate_spot_perp_to_1h(
    spot_perp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Spot × Perp execution flow from 5m to 1h.

    open_time:
        start of the hourly bucket.

    as_of_time:
        first instant at which the full hourly bucket is known.
    """

    columns = [
        "open_time",
        "as_of_time",
        "spot_quote_volume_1h",
        "spot_quote_delta_1h",
        "spot_delta_pct_1h",
        "perp_total_volume_1h",
        "perp_taker_delta_1h",
        "perp_delta_pct_1h",
        "spot_perp_valid_share_1h",
    ]

    if spot_perp_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    required = [
        "timestamp",
        "spot_quote_volume",
        "spot_quote_delta",
        "perp_total_volume",
        "perp_taker_delta",
        "spot_perp_flow_valid",
    ]

    missing = [
        column
        for column in required
        if column not in spot_perp_df.columns
    ]

    if missing:
        raise DerivativesRegimeError(
            "Missing Spot × Perp columns: "
            f"{missing}"
        )

    work = (
        spot_perp_df[
            required
        ]
        .copy()
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    )

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        utc=True,
        errors="coerce",
    )

    if work["timestamp"].isna().any():
        raise DerivativesRegimeError(
            "Invalid Spot × Perp timestamp"
        )

    work[
        "spot_perp_flow_valid"
    ] = (
        work[
            "spot_perp_flow_valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    result = (
        work
        .set_index("timestamp")
        .resample(
            "1h",
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            spot_quote_volume_1h=(
                "spot_quote_volume",
                "sum",
            ),
            spot_quote_delta_1h=(
                "spot_quote_delta",
                "sum",
            ),
            perp_total_volume_1h=(
                "perp_total_volume",
                "sum",
            ),
            perp_taker_delta_1h=(
                "perp_taker_delta",
                "sum",
            ),
            spot_perp_valid_share_1h=(
                "spot_perp_flow_valid",
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "timestamp": "open_time",
            }
        )
    )

    result["as_of_time"] = (
        result["open_time"]
        + pd.Timedelta(hours=1)
    )

    result[
        "spot_delta_pct_1h"
    ] = np.where(
        result[
            "spot_quote_volume_1h"
        ] > 0,
        result[
            "spot_quote_delta_1h"
        ]
        / result[
            "spot_quote_volume_1h"
        ],
        np.nan,
    )

    result[
        "perp_delta_pct_1h"
    ] = np.where(
        result[
            "perp_total_volume_1h"
        ] > 0,
        result[
            "perp_taker_delta_1h"
        ]
        / result[
            "perp_total_volume_1h"
        ],
        np.nan,
    )

    return result[
        columns
    ]


def aggregate_derivatives_to_1h(
    derivatives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate canonical derivatives features from 5m to 1h.

    Stock/state variables:
        last()

    Flow variables:
        sum()

    funding_time and funding_rate remain paired.

    as_of_time:
        open_time + 1h
    """

    columns = [
        "open_time",
        "as_of_time",
        "open_interest_quote",
        "funding_time",
        "funding_rate",
        "funding_rate_bps",
        "basis_rate",
        "basis_bps",
        "basis_zscore_24h",
        "long_liquidation_notional_1h",
        "short_liquidation_notional_1h",
        "total_liquidation_notional_1h",
        "liquidation_delta_1h",
        "liquidation_imbalance_1h",
    ]

    if derivatives_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    required = [
        "timestamp",
        "open_interest_quote",
        "funding_time",
        "funding_rate",
        "funding_rate_bps",
        "basis_rate",
        "basis_bps",
        "basis_zscore_24h",
        "long_liquidation_notional",
        "short_liquidation_notional",
    ]

    missing = [
        column
        for column in required
        if column not in derivatives_df.columns
    ]

    if missing:
        raise DerivativesRegimeError(
            "Missing derivatives feature columns: "
            f"{missing}"
        )

    work = (
        derivatives_df[
            required
        ]
        .copy()
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    )

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        utc=True,
        errors="coerce",
    )

    work["funding_time"] = pd.to_datetime(
        work["funding_time"],
        utc=True,
        errors="coerce",
    )

    if work["timestamp"].isna().any():
        raise DerivativesRegimeError(
            "Invalid derivatives timestamp"
        )

    result = (
        work
        .set_index("timestamp")
        .resample(
            "1h",
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            open_interest_quote=(
                "open_interest_quote",
                "last",
            ),
            funding_time=(
                "funding_time",
                "last",
            ),
            funding_rate=(
                "funding_rate",
                "last",
            ),
            funding_rate_bps=(
                "funding_rate_bps",
                "last",
            ),
            basis_rate=(
                "basis_rate",
                "last",
            ),
            basis_bps=(
                "basis_bps",
                "last",
            ),
            basis_zscore_24h=(
                "basis_zscore_24h",
                "last",
            ),
            long_liquidation_notional_1h=(
                "long_liquidation_notional",
                "sum",
            ),
            short_liquidation_notional_1h=(
                "short_liquidation_notional",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "timestamp": "open_time",
            }
        )
    )

    result["as_of_time"] = (
        result["open_time"]
        + pd.Timedelta(hours=1)
    )

    result[
        "total_liquidation_notional_1h"
    ] = (
        result[
            "long_liquidation_notional_1h"
        ]
        + result[
            "short_liquidation_notional_1h"
        ]
    )

    result[
        "liquidation_delta_1h"
    ] = (
        result[
            "short_liquidation_notional_1h"
        ]
        - result[
            "long_liquidation_notional_1h"
        ]
    )

    result[
        "liquidation_imbalance_1h"
    ] = np.where(
        result[
            "total_liquidation_notional_1h"
        ] > 0,
        result[
            "liquidation_delta_1h"
        ]
        / result[
            "total_liquidation_notional_1h"
        ],
        0.0,
    )

    return result[
        columns
    ]


def build_derivatives_regime(
    price_oi_df: pd.DataFrame,
    spot_perp_df: pd.DataFrame,
    derivatives_df: pd.DataFrame,
    flow_threshold: float = DEFAULT_FLOW_THRESHOLD,
    liquidation_imbalance_threshold: float = (
        DEFAULT_LIQUIDATION_IMBALANCE_THRESHOLD
    ),
) -> pd.DataFrame:
    """
    Build canonical 1h derivatives regime.

    open_time:
        hour label.

    as_of_time:
        open_time + 1h; timestamp at which all full-hour
        features may be considered known.

    Regimes describe state, not trading signals.
    """

    if flow_threshold < 0:
        raise DerivativesRegimeError(
            "flow_threshold cannot be negative"
        )

    if not (
        0
        <= liquidation_imbalance_threshold
        <= 1
    ):
        raise DerivativesRegimeError(
            "liquidation_imbalance_threshold "
            "must be between 0 and 1"
        )

    if price_oi_df.empty:
        raise DerivativesRegimeError(
            "Price × OI dataframe is empty"
        )

    required_price_oi = [
        "open_time",
        "close",
        "price_return_1h",
        "oi_quote_change_pct_1h",
        "price_oi_state",
        "price_oi_state_valid",
    ]

    missing = [
        column
        for column in required_price_oi
        if column not in price_oi_df.columns
    ]

    if missing:
        raise DerivativesRegimeError(
            "Missing Price × OI columns: "
            f"{missing}"
        )

    result = (
        price_oi_df[
            required_price_oi
        ]
        .copy()
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    result["open_time"] = pd.to_datetime(
        result["open_time"],
        utc=True,
        errors="coerce",
    )

    if result["open_time"].isna().any():
        raise DerivativesRegimeError(
            "Invalid Price × OI open_time"
        )

    result["as_of_time"] = (
        result["open_time"]
        + pd.Timedelta(hours=1)
    )

    flow_1h = aggregate_spot_perp_to_1h(
        spot_perp_df
    )

    flow_1h = flow_1h.drop(
        columns=["as_of_time"],
        errors="ignore",
    )

    result = result.merge(
        flow_1h,
        on="open_time",
        how="left",
    )

    derivatives_1h = (
        aggregate_derivatives_to_1h(
            derivatives_df
        )
    )

    derivatives_1h = derivatives_1h.drop(
        columns=["as_of_time"],
        errors="ignore",
    )

    result = result.merge(
        derivatives_1h,
        on="open_time",
        how="left",
    )

    historical_liquidations = (
        result[
            "total_liquidation_notional_1h"
        ]
        .shift(1)
    )

    result[
        "liquidation_threshold_24h"
    ] = (
        historical_liquidations
        .rolling(
            window=LIQUIDATION_LOOKBACK_HOURS,
            min_periods=6,
        )
        .quantile(
            LIQUIDATION_QUANTILE
        )
    )

    result[
        "liquidation_extreme"
    ] = (
        result[
            "liquidation_threshold_24h"
        ].notna()
        & (
            result[
                "liquidation_threshold_24h"
            ] > 0
        )
        & (
            result[
                "total_liquidation_notional_1h"
            ]
            > result[
                "liquidation_threshold_24h"
            ]
        )
    )

    result[
        "derivatives_regime_valid"
    ] = (
        result[
            "price_oi_state_valid"
        ]
        .fillna(False)
        .astype(bool)
        & result[
            "perp_delta_pct_1h"
        ].notna()
    )

    valid = result[
        "derivatives_regime_valid"
    ]

    perp_buy = (
        result[
            "perp_delta_pct_1h"
        ]
        >= flow_threshold
    )

    perp_sell = (
        result[
            "perp_delta_pct_1h"
        ]
        <= -flow_threshold
    )

    spot_buy = (
        result[
            "spot_delta_pct_1h"
        ]
        >= flow_threshold
    )

    spot_sell = (
        result[
            "spot_delta_pct_1h"
        ]
        <= -flow_threshold
    )

    funding_positive = (
        result["funding_rate"]
        >= 0
    )

    funding_negative = (
        result["funding_rate"]
        < 0
    )

    basis_positive = (
        result["basis_rate"]
        >= 0
    )

    basis_negative = (
        result["basis_rate"]
        < 0
    )

    short_liquidation_dominant = (
        result[
            "liquidation_imbalance_1h"
        ]
        >= liquidation_imbalance_threshold
    )

    long_liquidation_dominant = (
        result[
            "liquidation_imbalance_1h"
        ]
        <= -liquidation_imbalance_threshold
    )

    long_support_count = (
        perp_buy.fillna(False).astype(int)
        + spot_buy.fillna(False).astype(int)
        + funding_positive.fillna(False).astype(int)
        + basis_positive.fillna(False).astype(int)
    )

    short_support_count = (
        perp_sell.fillna(False).astype(int)
        + spot_sell.fillna(False).astype(int)
        + funding_negative.fillna(False).astype(int)
        + basis_negative.fillna(False).astype(int)
    )

    regime = pd.Series(
        ["unknown"] * len(result),
        index=result.index,
        dtype="string",
    )

    regime.loc[
        valid
    ] = "mixed"

    short_squeeze = (
        valid
        & result[
            "price_oi_state"
        ].eq(
            "short_covering_candidate"
        )
        & result[
            "liquidation_extreme"
        ]
        & short_liquidation_dominant
    )

    regime.loc[
        short_squeeze
    ] = "short_squeeze"

    long_flush = (
        valid
        & result[
            "price_oi_state"
        ].eq(
            "long_deleveraging_candidate"
        )
        & result[
            "liquidation_extreme"
        ]
        & long_liquidation_dominant
    )

    regime.loc[
        long_flush
    ] = "long_flush"

    unresolved = regime.eq(
        "mixed"
    )

    leveraged_long = (
        unresolved
        & result[
            "price_oi_state"
        ].eq(
            "position_build_up"
        )
        & perp_buy.fillna(False)
        & (
            long_support_count
            >= 2
        )
    )

    regime.loc[
        leveraged_long
    ] = "leveraged_long_build"

    unresolved = regime.eq(
        "mixed"
    )

    leveraged_short = (
        unresolved
        & result[
            "price_oi_state"
        ].eq(
            "short_build_up_candidate"
        )
        & perp_sell.fillna(False)
        & (
            short_support_count
            >= 2
        )
    )

    regime.loc[
        leveraged_short
    ] = "leveraged_short_build"

    unresolved = regime.eq(
        "mixed"
    )

    regime.loc[
        unresolved
        & result[
            "price_oi_state"
        ].eq(
            "short_covering_candidate"
        )
    ] = "short_covering"

    unresolved = regime.eq(
        "mixed"
    )

    regime.loc[
        unresolved
        & result[
            "price_oi_state"
        ].eq(
            "long_deleveraging_candidate"
        )
    ] = "long_deleveraging"

    unresolved = regime.eq(
        "mixed"
    )

    weak_spot = (
        result[
            "spot_delta_pct_1h"
        ].abs()
        < flow_threshold
    )

    weak_perp = (
        result[
            "perp_delta_pct_1h"
        ].abs()
        < flow_threshold
    )

    balanced = (
        unresolved
        & result[
            "price_oi_state"
        ].eq("neutral")
        & weak_spot.fillna(False)
        & weak_perp.fillna(False)
    )

    regime.loc[
        balanced
    ] = "balanced"

    result[
        "derivatives_regime"
    ] = regime

    confidence = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
    )

    long_mask = regime.eq(
        "leveraged_long_build"
    )

    confidence.loc[
        long_mask
    ] = (
        long_support_count.loc[
            long_mask
        ]
        / 4.0
    )

    short_mask = regime.eq(
        "leveraged_short_build"
    )

    confidence.loc[
        short_mask
    ] = (
        short_support_count.loc[
            short_mask
        ]
        / 4.0
    )

    confidence.loc[
        regime.eq(
            "short_squeeze"
        )
    ] = 1.0

    confidence.loc[
        regime.eq(
            "long_flush"
        )
    ] = 1.0

    short_covering_mask = (
        regime.eq(
            "short_covering"
        )
    )

    confidence.loc[
        short_covering_mask
    ] = np.where(
        perp_buy.loc[
            short_covering_mask
        ].fillna(False),
        0.75,
        0.50,
    )

    long_deleveraging_mask = (
        regime.eq(
            "long_deleveraging"
        )
    )

    confidence.loc[
        long_deleveraging_mask
    ] = np.where(
        perp_sell.loc[
            long_deleveraging_mask
        ].fillna(False),
        0.75,
        0.50,
    )

    confidence.loc[
        regime.eq(
            "balanced"
        )
    ] = 1.0

    confidence.loc[
        regime.eq(
            "mixed"
        )
    ] = 0.25

    result[
        "derivatives_regime_confidence"
    ] = confidence

    result.loc[
        ~valid,
        "derivatives_regime_confidence",
    ] = np.nan

    result[
        "long_support_count"
    ] = long_support_count

    result[
        "short_support_count"
    ] = short_support_count

    return (
        result
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )
