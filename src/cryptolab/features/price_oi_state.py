from __future__ import annotations

import numpy as np
import pandas as pd


class PriceOIStateError(ValueError):
    """Raised when Price × OI state generation fails."""


DEFAULT_PRICE_THRESHOLD = 0.001
DEFAULT_OI_THRESHOLD = 0.002


def resample_open_interest_to_1h(
    derivatives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resample canonical 5m derivatives data to 1h.

    Open Interest is a stock variable, not a flow variable.
    Therefore the hourly observation uses the LAST available
    OI value in each hour.

    Output
    ------
    open_time
    open_interest_base
    open_interest_quote
    """

    if derivatives_df.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open_interest_base",
                "open_interest_quote",
            ]
        )

    required_columns = [
        "timestamp",
        "open_interest_base",
        "open_interest_quote",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in derivatives_df.columns
    ]

    if missing_columns:
        raise PriceOIStateError(
            "Missing derivatives columns: "
            f"{missing_columns}"
        )

    work = (
        derivatives_df[
            required_columns
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
        raise PriceOIStateError(
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
            open_interest_base=(
                "open_interest_base",
                "last",
            ),
            open_interest_quote=(
                "open_interest_quote",
                "last",
            ),
        )
        .dropna(
            subset=[
                "open_interest_base",
                "open_interest_quote",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "timestamp": "open_time",
            }
        )
    )

    return result


def build_price_oi_state(
    price_df: pd.DataFrame,
    derivatives_df: pd.DataFrame,
    price_threshold: float = DEFAULT_PRICE_THRESHOLD,
    oi_threshold: float = DEFAULT_OI_THRESHOLD,
) -> pd.DataFrame:
    """
    Build causal 1h Price × Open Interest state.

    Parameters
    ----------
    price_df:
        Price Layer dataframe containing at least:
            open_time
            close

    derivatives_df:
        Canonical 5m derivatives feature dataframe.

    price_threshold:
        Minimum absolute 1h price return required before
        directional classification.

        Default:
            0.001 = 0.10%

    oi_threshold:
        Minimum absolute 1h OI change required before
        directional classification.

        Default:
            0.002 = 0.20%

    States
    ------
    position_build_up
        price positive + OI positive

    short_covering_candidate
        price positive + OI negative

    short_build_up_candidate
        price negative + OI positive

    long_deleveraging_candidate
        price negative + OI negative

    neutral
        valid data but one or both changes are below threshold

    unknown
        insufficient data

    Important
    ---------
    These are descriptive candidate states, not trading signals.
    """

    if price_threshold < 0:
        raise PriceOIStateError(
            "price_threshold cannot be negative"
        )

    if oi_threshold < 0:
        raise PriceOIStateError(
            "oi_threshold cannot be negative"
        )

    if price_df.empty:
        raise PriceOIStateError(
            "Price dataframe is empty"
        )

    if derivatives_df.empty:
        raise PriceOIStateError(
            "Derivatives dataframe is empty"
        )

    required_price_columns = [
        "open_time",
        "close",
    ]

    missing_price_columns = [
        column
        for column in required_price_columns
        if column not in price_df.columns
    ]

    if missing_price_columns:
        raise PriceOIStateError(
            "Missing price columns: "
            f"{missing_price_columns}"
        )

    # ========================================================
    # PRICE
    # ========================================================

    price = (
        price_df[
            required_price_columns
        ]
        .copy()
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    price["open_time"] = pd.to_datetime(
        price["open_time"],
        utc=True,
        errors="coerce",
    )

    if price["open_time"].isna().any():
        raise PriceOIStateError(
            "Invalid price open_time"
        )

    price["price_return_1h"] = (
        price["close"]
        .pct_change(
            fill_method=None
        )
    )

    # ========================================================
    # OPEN INTEREST
    # ========================================================

    oi = resample_open_interest_to_1h(
        derivatives_df
    )

    oi["oi_base_change_1h"] = (
        oi["open_interest_base"]
        .diff()
    )

    oi["oi_quote_change_1h"] = (
        oi["open_interest_quote"]
        .diff()
    )

    oi["oi_base_change_pct_1h"] = (
        oi["open_interest_base"]
        .pct_change(
            fill_method=None
        )
    )

    oi["oi_quote_change_pct_1h"] = (
        oi["open_interest_quote"]
        .pct_change(
            fill_method=None
        )
    )

    # ========================================================
    # JOIN
    # ========================================================

    result = price.merge(
        oi,
        on="open_time",
        how="left",
    )

    # ========================================================
    # VALIDITY
    # ========================================================

    result[
        "price_oi_state_valid"
    ] = (
        result["price_return_1h"]
        .notna()
        & result[
            "oi_quote_change_pct_1h"
        ]
        .notna()
    )

    # ========================================================
    # SIGNIFICANT MOVEMENT
    # ========================================================

    price_up = (
        result[
            "price_return_1h"
        ]
        >= price_threshold
    )

    price_down = (
        result[
            "price_return_1h"
        ]
        <= -price_threshold
    )

    oi_up = (
        result[
            "oi_quote_change_pct_1h"
        ]
        >= oi_threshold
    )

    oi_down = (
        result[
            "oi_quote_change_pct_1h"
        ]
        <= -oi_threshold
    )

    valid = result[
        "price_oi_state_valid"
    ]

    # ========================================================
    # STATE
    # ========================================================

    state = pd.Series(
        ["unknown"] * len(result),
        index=result.index,
        dtype="string",
    )

    state.loc[
        valid
    ] = "neutral"

    state.loc[
        valid
        & price_up
        & oi_up
    ] = "position_build_up"

    state.loc[
        valid
        & price_up
        & oi_down
    ] = "short_covering_candidate"

    state.loc[
        valid
        & price_down
        & oi_up
    ] = "short_build_up_candidate"

    state.loc[
        valid
        & price_down
        & oi_down
    ] = "long_deleveraging_candidate"

    result[
        "price_oi_state"
    ] = state

    # ========================================================
    # DIRECTION COMPONENTS
    # ========================================================

    result[
        "price_direction"
    ] = pd.Series(
        np.select(
            [
                price_up,
                price_down,
            ],
            [
                "up",
                "down",
            ],
            default="flat",
        ),
        index=result.index,
        dtype="string",
    )

    result[
        "oi_direction"
    ] = pd.Series(
        np.select(
            [
                oi_up,
                oi_down,
            ],
            [
                "up",
                "down",
            ],
            default="flat",
        ),
        index=result.index,
        dtype="string",
    )

    result.loc[
        ~valid,
        "price_direction",
    ] = "unknown"

    result.loc[
        ~valid,
        "oi_direction",
    ] = "unknown"

    # ========================================================
    # MAGNITUDE / CONFIDENCE
    #
    # This is evidence strength, NOT predictive probability.
    # ========================================================

    price_strength = (
        result[
            "price_return_1h"
        ]
        .abs()
        / max(
            price_threshold,
            1e-12,
        )
    )

    oi_strength = (
        result[
            "oi_quote_change_pct_1h"
        ]
        .abs()
        / max(
            oi_threshold,
            1e-12,
        )
    )

    result[
        "price_oi_strength"
    ] = (
        (
            price_strength
            + oi_strength
        )
        / 2.0
    )

    result[
        "price_oi_strength"
    ] = (
        result[
            "price_oi_strength"
        ]
        .clip(
            lower=0.0,
            upper=5.0,
        )
    )

    result.loc[
        ~valid,
        "price_oi_strength",
    ] = np.nan

    return (
        result
        .sort_values("open_time")
        .reset_index(drop=True)
    )
