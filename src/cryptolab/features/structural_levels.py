from __future__ import annotations
import numpy as np
import pandas as pd
class StructuralLevelsError(ValueError):
    """Raised when structural-level generation fails."""
def add_structural_level_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add active structural levels and structural-range features.
    Inputs
    ------
    Requires confirmed swing state produced by
    market_structure.py:
        last_swing_high
        last_swing_low
    Definitions
    -----------
    active_resistance:
        Most recently confirmed swing high.
    active_support:
        Most recently confirmed swing low.
    structural_range:
        active_resistance - active_support
    structural_range_position:
        Position of current close inside the active
        structural range.
        0.0 = at support
        0.5 = middle of range
        1.0 = at resistance
        Values may be below 0 or above 1 when price
        has moved outside the active range.
    range_state:
        incomplete
            One or both structural levels unavailable.
        inverted
            active_resistance <= active_support.
            Indicates an invalid or temporarily ambiguous
            structural range.
        inside_range
            support <= close <= resistance
        above_resistance
            close > resistance
        below_support
            close < support
    nearest_structure_side:
        "support" or "resistance", whichever is closer
        to the current close in percentage terms.
    Notes
    -----
    This function does not redefine BOS/CHOCH.
    It only represents the active structural geometry
    using already-confirmed swing information.
    """
    if df.empty:
        raise StructuralLevelsError(
            "Input dataframe is empty"
        )
    required_columns = [
        "open_time",
        "close",
        "last_swing_high",
        "last_swing_low",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise StructuralLevelsError(
            f"Missing columns: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    # ========================================================
    # ACTIVE LEVELS
    # ========================================================
    result["active_resistance"] = (
        result["last_swing_high"]
    )
    result["active_support"] = (
        result["last_swing_low"]
    )
    resistance = (
        result["active_resistance"]
    )
    support = (
        result["active_support"]
    )
    close = (
        result["close"]
    )
    # ========================================================
    # DISTANCE TO STRUCTURE
    # ========================================================
    result[
        "distance_to_resistance"
    ] = (
        resistance
        - close
    )
    result[
        "distance_to_support"
    ] = (
        close
        - support
    )
    result[
        "distance_to_resistance_pct"
    ] = np.where(
        resistance > 0,
        close / resistance - 1.0,
        np.nan,
    )
    result[
        "distance_to_support_pct"
    ] = np.where(
        support > 0,
        close / support - 1.0,
        np.nan,
    )
    # ========================================================
    # STRUCTURAL RANGE
    # ========================================================
    result[
        "structural_range"
    ] = (
        resistance
        - support
    )
    result[
        "structural_range_pct"
    ] = np.where(
        support > 0,
        result[
            "structural_range"
        ]
        / support,
        np.nan,
    )
    valid_range = (
        resistance.notna()
        & support.notna()
        & (
            resistance
            > support
        )
    )
    result[
        "structural_range_position"
    ] = np.where(
        valid_range,
        (
            close
            - support
        )
        / (
            resistance
            - support
        ),
        np.nan,
    )
    # ========================================================
    # RANGE STATE
    # ========================================================
    result[
        "range_state"
    ] = pd.Series(
        ["incomplete"] * len(result),
        dtype="string",
    )
    both_levels = (
        resistance.notna()
        & support.notna()
    )
    inverted = (
        both_levels
        & (
            resistance
            <= support
        )
    )
    result.loc[
        inverted,
        "range_state",
    ] = "inverted"
    inside_range = (
        valid_range
        & (
            close
            >= support
        )
        & (
            close
            <= resistance
        )
    )
    above_resistance = (
        valid_range
        & (
            close
            > resistance
        )
    )
    below_support = (
        valid_range
        & (
            close
            < support
        )
    )
    result.loc[
        inside_range,
        "range_state",
    ] = "inside_range"
    result.loc[
        above_resistance,
        "range_state",
    ] = "above_resistance"
    result.loc[
        below_support,
        "range_state",
    ] = "below_support"
    # ========================================================
    # NEAREST STRUCTURAL SIDE
    #
    # Use absolute percentage distance for comparability.
    # ========================================================
    resistance_distance_abs = (
        result[
            "distance_to_resistance_pct"
        ]
        .abs()
    )
    support_distance_abs = (
        result[
            "distance_to_support_pct"
        ]
        .abs()
    )
    result[
        "nearest_structure_side"
    ] = pd.Series(
        [pd.NA] * len(result),
        dtype="string",
    )
    result[
        "nearest_structure_distance_pct"
    ] = np.nan
    comparable = (
        valid_range
        & resistance_distance_abs.notna()
        & support_distance_abs.notna()
    )
    resistance_nearer = (
        comparable
        & (
            resistance_distance_abs
            <= support_distance_abs
        )
    )
    support_nearer = (
        comparable
        & (
            support_distance_abs
            < resistance_distance_abs
        )
    )
    result.loc[
        resistance_nearer,
        "nearest_structure_side",
    ] = "resistance"
    result.loc[
        resistance_nearer,
        "nearest_structure_distance_pct",
    ] = resistance_distance_abs[
        resistance_nearer
    ]
    result.loc[
        support_nearer,
        "nearest_structure_side",
    ] = "support"
    result.loc[
        support_nearer,
        "nearest_structure_distance_pct",
    ] = support_distance_abs[
        support_nearer
    ]
    # ========================================================
    # SIMPLE VALIDITY FLAG
    # ========================================================
    result[
        "structural_range_valid"
    ] = (
        valid_range
    )
    return result
