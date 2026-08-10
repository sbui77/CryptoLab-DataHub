from __future__ import annotations
import numpy as np
import pandas as pd
class StructuralStateError(ValueError):
    """Raised when structural-state generation fails."""
def add_structural_state_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add BOS, CHOCH and market structural-state features.
    Requires confirmed swing state from market_structure.py.
    Core logic
    ----------
    Bullish structural break:
        close > previously confirmed swing high
    Bearish structural break:
        close < previously confirmed swing low
    State machine
    -------------
    neutral + bullish break
        -> bullish state
        -> initial structural break
    neutral + bearish break
        -> bearish state
        -> initial structural break
    bullish + bullish break
        -> bullish BOS
    bullish + bearish break
        -> bearish CHOCH
        -> state changes to bearish
    bearish + bearish break
        -> bearish BOS
    bearish + bullish break
        -> bullish CHOCH
        -> state changes to bullish
    Important
    ---------
    The reference swing high/low is shifted by one row.
    Therefore a newly confirmed swing cannot be broken on the
    exact same row in which it first becomes known.
    This preserves a conservative causal interpretation.
    Generated columns
    -----------------
    break_reference_high
    break_reference_low
    structural_break
    break_direction
    break_level
    bos
    bos_direction
    choch
    choch_direction
    initial_structure_break
    structure_state
    state_changed
    """
    if df.empty:
        raise StructuralStateError(
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
        raise StructuralStateError(
            f"Missing columns: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    row_count = len(result)
    # ========================================================
    # BREAK REFERENCE LEVELS
    #
    # Shift by one row so a swing must already have been
    # confirmed before it can be used as a break level.
    # ========================================================
    result["break_reference_high"] = (
        result["last_swing_high"]
        .shift(1)
    )
    result["break_reference_low"] = (
        result["last_swing_low"]
        .shift(1)
    )
    # ========================================================
    # OUTPUT INITIALIZATION
    # ========================================================
    result["structural_break"] = False
    result["break_direction"] = pd.Series(
        [pd.NA] * row_count,
        dtype="string",
    )
    result["break_level"] = np.nan
    result["bos"] = False
    result["bos_direction"] = pd.Series(
        [pd.NA] * row_count,
        dtype="string",
    )
    result["choch"] = False
    result["choch_direction"] = pd.Series(
        [pd.NA] * row_count,
        dtype="string",
    )
    result["initial_structure_break"] = False
    result["structure_state"] = pd.Series(
        ["neutral"] * row_count,
        dtype="string",
    )
    result["state_changed"] = False
    # ========================================================
    # STATE MACHINE
    # ========================================================
    current_state = "neutral"
    active_high: float | None = None
    active_low: float | None = None
    high_broken = False
    low_broken = False
    previous_reference_high: float | None = None
    previous_reference_low: float | None = None
    for index in range(row_count):
        reference_high = result.at[
            index,
            "break_reference_high",
        ]
        reference_low = result.at[
            index,
            "break_reference_low",
        ]
        close = result.at[
            index,
            "close",
        ]
        # ----------------------------------------------------
        # New confirmed swing high becomes active resistance.
        # Reset its breached state.
        # ----------------------------------------------------
        if pd.notna(reference_high):
            reference_high = float(
                reference_high
            )
            if (
                previous_reference_high is None
                or not np.isclose(
                    reference_high,
                    previous_reference_high,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                active_high = reference_high
                high_broken = False
            previous_reference_high = (
                reference_high
            )
        # ----------------------------------------------------
        # New confirmed swing low becomes active support.
        # ----------------------------------------------------
        if pd.notna(reference_low):
            reference_low = float(
                reference_low
            )
            if (
                previous_reference_low is None
                or not np.isclose(
                    reference_low,
                    previous_reference_low,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                active_low = reference_low
                low_broken = False
            previous_reference_low = (
                reference_low
            )
        bullish_break = False
        bearish_break = False
        # ----------------------------------------------------
        # Break must occur only once per active swing level.
        # ----------------------------------------------------
        if (
            active_high is not None
            and not high_broken
            and float(close) > active_high
        ):
            bullish_break = True
        if (
            active_low is not None
            and not low_broken
            and float(close) < active_low
        ):
            bearish_break = True
        # ----------------------------------------------------
        # Extremely rare case:
        # one close cannot logically be both above resistance
        # and below support when levels are ordered normally.
        #
        # If corrupted/inverted levels create such condition,
        # skip classification rather than invent structure.
        # ----------------------------------------------------
        if (
            bullish_break
            and bearish_break
        ):
            result.at[
                index,
                "structure_state",
            ] = current_state
            continue
        previous_state = current_state
        # ====================================================
        # BULLISH BREAK
        # ====================================================
        if bullish_break:
            high_broken = True
            result.at[
                index,
                "structural_break",
            ] = True
            result.at[
                index,
                "break_direction",
            ] = "bullish"
            result.at[
                index,
                "break_level",
            ] = active_high
            # Neutral -> first directional state.
            if current_state == "neutral":
                result.at[
                    index,
                    "initial_structure_break",
                ] = True
                current_state = "bullish"
            # Bullish continuation.
            elif current_state == "bullish":
                result.at[
                    index,
                    "bos",
                ] = True
                result.at[
                    index,
                    "bos_direction",
                ] = "bullish"
            # Bearish -> bullish reversal.
            elif current_state == "bearish":
                result.at[
                    index,
                    "choch",
                ] = True
                result.at[
                    index,
                    "choch_direction",
                ] = "bullish"
                current_state = "bullish"
        # ====================================================
        # BEARISH BREAK
        # ====================================================
        elif bearish_break:
            low_broken = True
            result.at[
                index,
                "structural_break",
            ] = True
            result.at[
                index,
                "break_direction",
            ] = "bearish"
            result.at[
                index,
                "break_level",
            ] = active_low
            # Neutral -> first directional state.
            if current_state == "neutral":
                result.at[
                    index,
                    "initial_structure_break",
                ] = True
                current_state = "bearish"
            # Bearish continuation.
            elif current_state == "bearish":
                result.at[
                    index,
                    "bos",
                ] = True
                result.at[
                    index,
                    "bos_direction",
                ] = "bearish"
            # Bullish -> bearish reversal.
            elif current_state == "bullish":
                result.at[
                    index,
                    "choch",
                ] = True
                result.at[
                    index,
                    "choch_direction",
                ] = "bearish"
                current_state = "bearish"
        # ====================================================
        # STORE STATE
        # ====================================================
        result.at[
            index,
            "structure_state",
        ] = current_state
        result.at[
            index,
            "state_changed",
        ] = (
            current_state
            != previous_state
        )
    return result
