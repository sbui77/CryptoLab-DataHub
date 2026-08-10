from __future__ import annotations

import numpy as np
import pandas as pd


class SpotPerpFlowError(ValueError):
    """Raised when Spot × Perp flow feature generation fails."""


DEFAULT_DIRECTIONAL_THRESHOLD = 0.10
DEFAULT_LEADERSHIP_RATIO = 1.50


def resample_spot_flow_to_5m(
    spot_flow: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resample canonical Spot Trade Flow from 1m to 5m.

    Spot flow is treated as flow data, therefore:
        quote_volume -> sum
        quote_delta  -> sum

    Normalized delta is recomputed after aggregation:

        spot_delta_pct
        =
        spot_quote_delta / spot_quote_volume

    Output
    ------
    timestamp
    spot_quote_volume
    spot_quote_delta
    spot_delta_pct
    """

    columns = [
        "timestamp",
        "spot_quote_volume",
        "spot_quote_delta",
        "spot_delta_pct",
    ]

    if spot_flow.empty:
        return pd.DataFrame(
            columns=columns
        )

    required = [
        "open_time",
        "quote_volume",
        "quote_delta",
    ]

    missing = [
        column
        for column in required
        if column not in spot_flow.columns
    ]

    if missing:
        raise SpotPerpFlowError(
            "Missing Spot Trade Flow columns: "
            f"{missing}"
        )

    work = (
        spot_flow[
            required
        ]
        .copy()
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
    )

    work["open_time"] = pd.to_datetime(
        work["open_time"],
        utc=True,
        errors="coerce",
    )

    if work["open_time"].isna().any():
        raise SpotPerpFlowError(
            "Invalid Spot open_time"
        )

    result = (
        work
        .set_index("open_time")
        .resample(
            "5min",
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            spot_quote_volume=(
                "quote_volume",
                "sum",
            ),
            spot_quote_delta=(
                "quote_delta",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "open_time": "timestamp",
            }
        )
    )

    result[
        "spot_delta_pct"
    ] = np.where(
        result[
            "spot_quote_volume"
        ] > 0,
        result[
            "spot_quote_delta"
        ]
        / result[
            "spot_quote_volume"
        ],
        np.nan,
    )

    return result[
        columns
    ]


def build_spot_perp_flow_state(
    spot_flow: pd.DataFrame,
    derivatives_df: pd.DataFrame,
    directional_threshold: float = DEFAULT_DIRECTIONAL_THRESHOLD,
    leadership_ratio: float = DEFAULT_LEADERSHIP_RATIO,
) -> pd.DataFrame:
    """
    Build canonical 5m Spot × Perp flow interaction state.

    Parameters
    ----------
    spot_flow:
        Spot Trade Flow Layer, normally curated 1m data.

    derivatives_df:
        Canonical derivatives 5m feature dataset.

    directional_threshold:
        Minimum absolute normalized delta required to
        classify one market as directionally active.

        Default:
            0.10 = 10%

    leadership_ratio:
        Required magnitude ratio for one market to be
        considered clearly dominant when the other market
        is weak or similarly directed.

        Default:
            1.50

    States
    ------
    confirmed_buy
    confirmed_sell

    spot_led_buy
    spot_led_sell

    perp_led_buy
    perp_led_sell

    spot_buy_perp_sell
    spot_sell_perp_buy

    neutral
    unknown

    These are descriptive cross-market states,
    not trading signals.
    """

    if directional_threshold < 0:
        raise SpotPerpFlowError(
            "directional_threshold cannot be negative"
        )

    if leadership_ratio <= 1.0:
        raise SpotPerpFlowError(
            "leadership_ratio must exceed 1.0"
        )

    if spot_flow.empty:
        raise SpotPerpFlowError(
            "Spot Trade Flow dataset is empty"
        )

    if derivatives_df.empty:
        raise SpotPerpFlowError(
            "Derivatives dataset is empty"
        )

    required_derivatives = [
        "timestamp",
        "futures_buy_volume",
        "futures_sell_volume",
        "futures_total_volume",
        "futures_taker_delta",
        "futures_taker_delta_pct",
    ]

    missing_derivatives = [
        column
        for column in required_derivatives
        if column not in derivatives_df.columns
    ]

    if missing_derivatives:
        raise SpotPerpFlowError(
            "Missing derivatives columns: "
            f"{missing_derivatives}"
        )

    # ========================================================
    # SPOT 1m -> 5m
    # ========================================================

    spot_5m = resample_spot_flow_to_5m(
        spot_flow
    )

    # ========================================================
    # PERP 5m
    # ========================================================

    perp = (
        derivatives_df[
            required_derivatives
        ]
        .copy()
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    )

    perp["timestamp"] = pd.to_datetime(
        perp["timestamp"],
        utc=True,
        errors="coerce",
    )

    if perp["timestamp"].isna().any():
        raise SpotPerpFlowError(
            "Invalid derivatives timestamp"
        )

    perp = perp.rename(
        columns={
            "futures_buy_volume":
                "perp_buy_volume",

            "futures_sell_volume":
                "perp_sell_volume",

            "futures_total_volume":
                "perp_total_volume",

            "futures_taker_delta":
                "perp_taker_delta",

            "futures_taker_delta_pct":
                "perp_delta_pct",
        }
    )

    # ========================================================
    # INNER CLOCK
    #
    # Use derivatives/OI 5m time spine and left-join spot.
    # ========================================================

    result = perp.merge(
        spot_5m,
        on="timestamp",
        how="left",
    )

    # ========================================================
    # VALIDITY
    # ========================================================

    result[
        "spot_perp_flow_valid"
    ] = (
        result[
            "spot_delta_pct"
        ].notna()
        & result[
            "perp_delta_pct"
        ].notna()
    )

    valid = result[
        "spot_perp_flow_valid"
    ]

    spot = result[
        "spot_delta_pct"
    ]

    perp_delta = result[
        "perp_delta_pct"
    ]

    threshold = (
        directional_threshold
    )

    # ========================================================
    # MARKET DIRECTION
    # ========================================================

    spot_buy = (
        spot
        >= threshold
    )

    spot_sell = (
        spot
        <= -threshold
    )

    perp_buy = (
        perp_delta
        >= threshold
    )

    perp_sell = (
        perp_delta
        <= -threshold
    )

    spot_active = (
        spot.abs()
        >= threshold
    )

    perp_active = (
        perp_delta.abs()
        >= threshold
    )

    # ========================================================
    # LEADERSHIP MAGNITUDE
    # ========================================================

    abs_spot = (
        spot.abs()
    )

    abs_perp = (
        perp_delta.abs()
    )

    spot_leads = (
        abs_spot
        >= (
            abs_perp
            * leadership_ratio
        )
    )

    perp_leads = (
        abs_perp
        >= (
            abs_spot
            * leadership_ratio
        )
    )

    # ========================================================
    # STATE INITIALIZATION
    # ========================================================

    state = pd.Series(
        ["unknown"] * len(result),
        index=result.index,
        dtype="string",
    )

    state.loc[
        valid
    ] = "neutral"

    # ========================================================
    # DIVERGENCE
    #
    # Highest priority.
    # ========================================================

    state.loc[
        valid
        & spot_buy
        & perp_sell
    ] = "spot_buy_perp_sell"

    state.loc[
        valid
        & spot_sell
        & perp_buy
    ] = "spot_sell_perp_buy"

    # ========================================================
    # CONFIRMED
    # ========================================================

    unresolved = state.eq(
        "neutral"
    )

    state.loc[
        unresolved
        & spot_buy
        & perp_buy
        & ~spot_leads
        & ~perp_leads
    ] = "confirmed_buy"

    state.loc[
        unresolved
        & spot_sell
        & perp_sell
        & ~spot_leads
        & ~perp_leads
    ] = "confirmed_sell"

    # ========================================================
    # SPOT-LED
    # ========================================================

    unresolved = state.eq(
        "neutral"
    )

    state.loc[
        unresolved
        & spot_buy
        & (
            ~perp_active
            | (
                perp_buy
                & spot_leads
            )
        )
    ] = "spot_led_buy"

    state.loc[
        unresolved
        & spot_sell
        & (
            ~perp_active
            | (
                perp_sell
                & spot_leads
            )
        )
    ] = "spot_led_sell"

    # ========================================================
    # PERP-LED
    # ========================================================

    unresolved = state.eq(
        "neutral"
    )

    state.loc[
        unresolved
        & perp_buy
        & (
            ~spot_active
            | (
                spot_buy
                & perp_leads
            )
        )
    ] = "perp_led_buy"

    state.loc[
        unresolved
        & perp_sell
        & (
            ~spot_active
            | (
                spot_sell
                & perp_leads
            )
        )
    ] = "perp_led_sell"

    result[
        "spot_perp_flow_state"
    ] = state

    # ========================================================
    # NUMERIC CROSS-MARKET FEATURES
    # ========================================================

    result[
        "spot_perp_delta_spread"
    ] = (
        result[
            "spot_delta_pct"
        ]
        - result[
            "perp_delta_pct"
        ]
    )

    result[
        "spot_perp_delta_agreement"
    ] = np.where(
        valid,
        np.sign(
            result[
                "spot_delta_pct"
            ]
        )
        == np.sign(
            result[
                "perp_delta_pct"
            ]
        ),
        False,
    )

    result[
        "spot_flow_strength"
    ] = result[
        "spot_delta_pct"
    ].abs()

    result[
        "perp_flow_strength"
    ] = result[
        "perp_delta_pct"
    ].abs()

    result[
        "spot_perp_combined_strength"
    ] = (
        (
            result[
                "spot_flow_strength"
            ]
            + result[
                "perp_flow_strength"
            ]
        )
        / 2.0
    )

    result.loc[
        ~valid,
        [
            "spot_perp_delta_spread",
            "spot_flow_strength",
            "perp_flow_strength",
            "spot_perp_combined_strength",
        ],
    ] = np.nan

    return (
        result
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
