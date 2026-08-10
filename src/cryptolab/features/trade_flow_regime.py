from __future__ import annotations

import numpy as np
import pandas as pd


class TradeFlowRegimeError(ValueError):
    """Raised when trade-flow regime generation fails."""


def add_trade_flow_regime_features(
    flow_df: pd.DataFrame,
    large_flow_df: pd.DataFrame | None = None,
    rolling_window: int = 60,
    delta_threshold: float = 0.15,
    count_threshold: float = 0.10,
    balanced_threshold: float = 0.10,
    large_share_threshold: float = 0.20,
    large_delta_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Add descriptive Trade Flow Regime features.

    Intended primarily for 1-minute trade-flow bars.

    Inputs
    ------
    flow_df:
        Output of:
            aggregate_aggtrades()
            -> add_cvd_features()

    large_flow_df:
        Optional output of:
            aggregate_large_trade_features()

        Joined on open_time.

    Regimes
    -------
    large_buy_pressure
        Large trades are material and strongly buy-dominant.

    large_sell_pressure
        Large trades are material and strongly sell-dominant.

    buy_dominant
        Positive bucket delta,
        positive trade-count imbalance,
        positive rolling delta.

    sell_dominant
        Negative bucket delta,
        negative trade-count imbalance,
        negative rolling delta.

    balanced
        Bucket and rolling imbalance are all small.

    mixed
        Evidence exists but does not align strongly.

    unknown
        Insufficient data.

    Notes
    -----
    These are descriptive labels, not trading signals.

    Thresholds are explicit parameters so they can later
    be empirically tested and calibrated.
    """

    if flow_df.empty:
        return flow_df.copy()

    if rolling_window <= 0:
        raise TradeFlowRegimeError(
            "rolling_window must be greater than zero"
        )

    if not 0 <= balanced_threshold <= 1:
        raise TradeFlowRegimeError(
            "balanced_threshold must be between 0 and 1"
        )

    if not 0 <= delta_threshold <= 1:
        raise TradeFlowRegimeError(
            "delta_threshold must be between 0 and 1"
        )

    if not 0 <= count_threshold <= 1:
        raise TradeFlowRegimeError(
            "count_threshold must be between 0 and 1"
        )

    if not 0 <= large_share_threshold <= 1:
        raise TradeFlowRegimeError(
            "large_share_threshold must be between 0 and 1"
        )

    if not 0 <= large_delta_threshold <= 1:
        raise TradeFlowRegimeError(
            "large_delta_threshold must be between 0 and 1"
        )

    required_columns = [
        "open_time",
        "quote_volume",
        "quote_delta",
        "quote_delta_pct",
        "trade_count_imbalance",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in flow_df.columns
    ]

    if missing_columns:
        raise TradeFlowRegimeError(
            f"Missing flow columns: {missing_columns}"
        )

    result = (
        flow_df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # ========================================================
    # MERGE LARGE-TRADE FLOW
    # ========================================================

    if (
        large_flow_df is not None
        and not large_flow_df.empty
    ):
        required_large = [
            "open_time",
            "large_trade_quote",
            "large_quote_delta",
            "large_trade_quote_share",
        ]

        missing_large = [
            column
            for column in required_large
            if column not in large_flow_df.columns
        ]

        if missing_large:
            raise TradeFlowRegimeError(
                f"Missing large-flow columns: {missing_large}"
            )

        large = (
            large_flow_df[
                required_large
            ]
            .copy()
            .sort_values("open_time")
            .drop_duplicates(
                subset=["open_time"],
                keep="last",
            )
        )

        result = result.merge(
            large,
            on="open_time",
            how="left",
        )

    else:
        result["large_trade_quote"] = np.nan
        result["large_quote_delta"] = np.nan
        result["large_trade_quote_share"] = np.nan

    # ========================================================
    # ROLLING FLOW IMBALANCE
    # ========================================================

    result[
        "rolling_quote_delta_60"
    ] = (
        result["quote_delta"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .sum()
    )

    result[
        "rolling_quote_volume_60"
    ] = (
        result["quote_volume"]
        .rolling(
            window=rolling_window,
            min_periods=rolling_window,
        )
        .sum()
    )

    result[
        "rolling_quote_imbalance_60"
    ] = np.where(
        result[
            "rolling_quote_volume_60"
        ] > 0,
        result[
            "rolling_quote_delta_60"
        ]
        / result[
            "rolling_quote_volume_60"
        ],
        np.nan,
    )

    # ========================================================
    # LARGE-TRADE IMBALANCE
    # ========================================================

    result[
        "large_quote_delta_pct"
    ] = np.where(
        result[
            "large_trade_quote"
        ] > 0,
        result[
            "large_quote_delta"
        ]
        / result[
            "large_trade_quote"
        ],
        np.nan,
    )

    # ========================================================
    # OUTPUT INITIALIZATION
    # ========================================================

    result[
        "trade_flow_regime"
    ] = pd.Series(
        ["unknown"] * len(result),
        dtype="string",
    )

    quote_delta_pct = (
        result["quote_delta_pct"]
    )

    count_imbalance = (
        result[
            "trade_count_imbalance"
        ]
    )

    rolling_imbalance = (
        result[
            "rolling_quote_imbalance_60"
        ]
    )

    large_share = (
        result[
            "large_trade_quote_share"
        ]
    )

    large_delta_pct = (
        result[
            "large_quote_delta_pct"
        ]
    )

    # ========================================================
    # LARGE-TRADE PRESSURE
    #
    # Highest priority because it represents a distinct
    # participant-size condition.
    # ========================================================

    large_buy_pressure = (
        large_share.notna()
        & large_delta_pct.notna()
        & (
            large_share
            >= large_share_threshold
        )
        & (
            large_delta_pct
            >= large_delta_threshold
        )
    )

    large_sell_pressure = (
        large_share.notna()
        & large_delta_pct.notna()
        & (
            large_share
            >= large_share_threshold
        )
        & (
            large_delta_pct
            <= -large_delta_threshold
        )
    )

    result.loc[
        large_buy_pressure,
        "trade_flow_regime",
    ] = "large_buy_pressure"

    result.loc[
        large_sell_pressure,
        "trade_flow_regime",
    ] = "large_sell_pressure"

    # ========================================================
    # DIRECTIONAL DOMINANCE
    # ========================================================

    unresolved = (
        result[
            "trade_flow_regime"
        ].eq("unknown")
    )

    buy_dominant = (
        unresolved
        & quote_delta_pct.notna()
        & count_imbalance.notna()
        & rolling_imbalance.notna()
        & (
            quote_delta_pct
            >= delta_threshold
        )
        & (
            count_imbalance
            >= count_threshold
        )
        & (
            rolling_imbalance
            > 0
        )
    )

    sell_dominant = (
        unresolved
        & quote_delta_pct.notna()
        & count_imbalance.notna()
        & rolling_imbalance.notna()
        & (
            quote_delta_pct
            <= -delta_threshold
        )
        & (
            count_imbalance
            <= -count_threshold
        )
        & (
            rolling_imbalance
            < 0
        )
    )

    result.loc[
        buy_dominant,
        "trade_flow_regime",
    ] = "buy_dominant"

    result.loc[
        sell_dominant,
        "trade_flow_regime",
    ] = "sell_dominant"

    # ========================================================
    # BALANCED FLOW
    # ========================================================

    unresolved = (
        result[
            "trade_flow_regime"
        ].eq("unknown")
    )

    balanced = (
        unresolved
        & quote_delta_pct.notna()
        & count_imbalance.notna()
        & rolling_imbalance.notna()
        & (
            quote_delta_pct.abs()
            <= balanced_threshold
        )
        & (
            count_imbalance.abs()
            <= balanced_threshold
        )
        & (
            rolling_imbalance.abs()
            <= balanced_threshold
        )
    )

    result.loc[
        balanced,
        "trade_flow_regime",
    ] = "balanced"

    # ========================================================
    # MIXED
    #
    # Enough core evidence exists, but not enough alignment
    # for any stronger label.
    # ========================================================

    core_available = (
        quote_delta_pct.notna()
        & count_imbalance.notna()
        & rolling_imbalance.notna()
    )

    mixed = (
        result[
            "trade_flow_regime"
        ].eq("unknown")
        & core_available
    )

    result.loc[
        mixed,
        "trade_flow_regime",
    ] = "mixed"

    # ========================================================
    # CONFIDENCE / VALIDITY
    #
    # Confidence means evidence availability, not predictive
    # probability.
    # ========================================================

    core_delta_available = (
        quote_delta_pct.notna()
    ).astype("float64")

    count_available = (
        count_imbalance.notna()
    ).astype("float64")

    rolling_available = (
        rolling_imbalance.notna()
    ).astype("float64")

    large_available = (
        large_share.notna()
        & large_delta_pct.notna()
    ).astype("float64")

    result[
        "flow_regime_confidence"
    ] = (
        core_delta_available
        + count_available
        + rolling_available
        + large_available
    ) / 4.0

    result[
        "trade_flow_regime_valid"
    ] = (
        ~result[
            "trade_flow_regime"
        ].eq("unknown")
        & (
            result[
                "flow_regime_confidence"
            ]
            >= 0.75
        )
    )

    return result
