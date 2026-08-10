from __future__ import annotations
import numpy as np
import pandas as pd
class RegimeError(ValueError):
    """Raised when price-regime generation fails."""
def add_price_regime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add descriptive price-regime features.
    This layer combines already-derived evidence:
        structure_state
        range_state
        rv_30d_percentile_1y
        range_position_720
        structural_range_position
    It does NOT generate trading signals.
    ----------------------------------------------------------
    Volatility regime
    ----------------------------------------------------------
    Based on trailing one-year percentile of 30-day
    realized volatility:
        <= 0.20  very_low
        <= 0.40  low
        <= 0.60  normal
        <= 0.80  high
        >  0.80  very_high
    ----------------------------------------------------------
    Location regime
    ----------------------------------------------------------
    Based on 30-day rolling range position:
        <= 0.33  lower_range
        <  0.67  mid_range
        >= 0.67  upper_range
    ----------------------------------------------------------
    Price regime
    ----------------------------------------------------------
    bullish_expansion:
        structure_state == bullish
        AND price is above active resistance
    bearish_expansion:
        structure_state == bearish
        AND price is below active support
    trending_up:
        bullish structure
        AND not above resistance
        AND location is upper half / upper region
    trending_down:
        bearish structure
        AND not below support
        AND location is lower half / lower region
    volatile_range:
        inside structural range
        AND volatility regime is high / very_high
    range:
        inside structural range
        AND volatility is not high
    unknown:
        insufficient or contradictory evidence
    ----------------------------------------------------------
    Regime confidence
    ----------------------------------------------------------
    A descriptive score from 0 to 1 based on how many
    independent evidence components are available:
        structural state
        structural range state
        volatility regime
        rolling location
    This is NOT a probability of future return.
    """
    if df.empty:
        raise RegimeError(
            "Input dataframe is empty"
        )
    required_columns = [
        "open_time",
        "close",
        "structure_state",
        "range_state",
        "rv_30d_percentile_1y",
        "range_position_720",
        "structural_range_position",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise RegimeError(
            f"Missing columns: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    row_count = len(result)
    # ========================================================
    # VOLATILITY REGIME
    # ========================================================
    result["volatility_regime"] = pd.Series(
        ["unknown"] * row_count,
        dtype="string",
    )
    rv_percentile = (
        result[
            "rv_30d_percentile_1y"
        ]
    )
    valid_rv = (
        rv_percentile.notna()
        & (
            rv_percentile >= 0
        )
        & (
            rv_percentile <= 1
        )
    )
    result.loc[
        valid_rv
        & (
            rv_percentile <= 0.20
        ),
        "volatility_regime",
    ] = "very_low"
    result.loc[
        valid_rv
        & (
            rv_percentile > 0.20
        )
        & (
            rv_percentile <= 0.40
        ),
        "volatility_regime",
    ] = "low"
    result.loc[
        valid_rv
        & (
            rv_percentile > 0.40
        )
        & (
            rv_percentile <= 0.60
        ),
        "volatility_regime",
    ] = "normal"
    result.loc[
        valid_rv
        & (
            rv_percentile > 0.60
        )
        & (
            rv_percentile <= 0.80
        ),
        "volatility_regime",
    ] = "high"
    result.loc[
        valid_rv
        & (
            rv_percentile > 0.80
        ),
        "volatility_regime",
    ] = "very_high"
    # ========================================================
    # LOCATION REGIME
    # ========================================================
    result["location_regime"] = pd.Series(
        ["unknown"] * row_count,
        dtype="string",
    )
    location = (
        result[
            "range_position_720"
        ]
    )
    valid_location = (
        location.notna()
    )
    result.loc[
        valid_location
        & (
            location <= 0.33
        ),
        "location_regime",
    ] = "lower_range"
    result.loc[
        valid_location
        & (
            location > 0.33
        )
        & (
            location < 0.67
        ),
        "location_regime",
    ] = "mid_range"
    result.loc[
        valid_location
        & (
            location >= 0.67
        ),
        "location_regime",
    ] = "upper_range"
    # ========================================================
    # PRICE REGIME
    # ========================================================
    result["price_regime"] = pd.Series(
        ["unknown"] * row_count,
        dtype="string",
    )
    structure_state = (
        result[
            "structure_state"
        ]
    )
    range_state = (
        result[
            "range_state"
        ]
    )
    volatility_regime = (
        result[
            "volatility_regime"
        ]
    )
    location_regime = (
        result[
            "location_regime"
        ]
    )
    # --------------------------------------------------------
    # Expansion regimes
    # --------------------------------------------------------
    bullish_expansion = (
        structure_state.eq("bullish")
        & range_state.eq(
            "above_resistance"
        )
    )
    bearish_expansion = (
        structure_state.eq("bearish")
        & range_state.eq(
            "below_support"
        )
    )
    result.loc[
        bullish_expansion,
        "price_regime",
    ] = "bullish_expansion"
    result.loc[
        bearish_expansion,
        "price_regime",
    ] = "bearish_expansion"
    # --------------------------------------------------------
    # Trending regimes
    # --------------------------------------------------------
    trending_up = (
        structure_state.eq("bullish")
        & ~bullish_expansion
        & location_regime.eq(
            "upper_range"
        )
    )
    trending_down = (
        structure_state.eq("bearish")
        & ~bearish_expansion
        & location_regime.eq(
            "lower_range"
        )
    )
    result.loc[
        trending_up,
        "price_regime",
    ] = "trending_up"
    result.loc[
        trending_down,
        "price_regime",
    ] = "trending_down"
    # --------------------------------------------------------
    # Range regimes
    # --------------------------------------------------------
    inside_range = (
        range_state.eq(
            "inside_range"
        )
    )
    volatile = (
        volatility_regime.isin(
            [
                "high",
                "very_high",
            ]
        )
    )
    volatile_range = (
        inside_range
        & volatile
        & result[
            "price_regime"
        ].eq(
            "unknown"
        )
    )
    normal_range = (
        inside_range
        & ~volatile
        & result[
            "price_regime"
        ].eq(
            "unknown"
        )
    )
    result.loc[
        volatile_range,
        "price_regime",
    ] = "volatile_range"
    result.loc[
        normal_range,
        "price_regime",
    ] = "range"
    # ========================================================
    # REGIME CONFIDENCE
    #
    # Evidence availability score.
    # Not a predictive probability.
    # ========================================================
    structural_state_available = (
        structure_state.isin(
            [
                "bullish",
                "bearish",
            ]
        )
    ).astype(
        "float64"
    )
    range_state_available = (
        range_state.isin(
            [
                "inside_range",
                "above_resistance",
                "below_support",
            ]
        )
    ).astype(
        "float64"
    )
    volatility_available = (
        ~volatility_regime.eq(
            "unknown"
        )
    ).astype(
        "float64"
    )
    location_available = (
        ~location_regime.eq(
            "unknown"
        )
    ).astype(
        "float64"
    )
    evidence_count = (
        structural_state_available
        + range_state_available
        + volatility_available
        + location_available
    )
    result[
        "regime_confidence"
    ] = (
        evidence_count
        / 4.0
    )
    # ========================================================
    # REGIME VALIDITY
    # ========================================================
    result[
        "price_regime_valid"
    ] = (
        ~result[
            "price_regime"
        ].eq(
            "unknown"
        )
        & (
            result[
                "regime_confidence"
            ]
            >= 0.75
        )
    )
    return result
