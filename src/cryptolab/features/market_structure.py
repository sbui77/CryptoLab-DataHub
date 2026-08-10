from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
class MarketStructureError(ValueError):
    """Raised when market-structure generation fails."""
def add_market_structure_features(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    expected_interval_hours: int = 1,
) -> pd.DataFrame:
    """
    Add confirmed swing and basic market-structure features.
    This implementation is causal/backtest-safe and uses
    vectorized pivot detection.
    A pivot at index P is only emitted as a confirmed swing at:
        P + right_bars
    Therefore the swing cannot be known until enough bars
    to the right have occurred.
    Swing High
    ----------
    Pivot high must be strictly greater than all highs in:
        - left_bars before the pivot
        - right_bars after the pivot
    Swing Low
    ---------
    Pivot low must be strictly lower than all lows in:
        - left_bars before the pivot
        - right_bars after the pivot
    Structure labels
    ----------------
    High swings:
        HH = Higher High
        LH = Lower High
        EH = Equal High
    Low swings:
        HL = Higher Low
        LL = Lower Low
        EL = Equal Low
    Data quality
    ------------
    A swing is accepted only when the entire detection window
    is continuous at expected_interval_hours.
    Output columns
    --------------
    confirmed_swing_high
    confirmed_swing_low
    swing_event
    swing_kind
    swing_price
    swing_time
    swing_confirmation_time
    swing_window_valid
    structure_label
    last_swing_high
    last_swing_low
    last_swing_high_time
    last_swing_low_time
    hours_since_swing_high
    hours_since_swing_low
    """
    if df.empty:
        raise MarketStructureError(
            "Input dataframe is empty"
        )
    if left_bars <= 0:
        raise MarketStructureError(
            "left_bars must be greater than zero"
        )
    if right_bars <= 0:
        raise MarketStructureError(
            "right_bars must be greater than zero"
        )
    if expected_interval_hours <= 0:
        raise MarketStructureError(
            "expected_interval_hours must be greater than zero"
        )
    required_columns = [
        "open_time",
        "high",
        "low",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise MarketStructureError(
            f"Missing columns: {missing_columns}"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if not pd.api.types.is_datetime64_any_dtype(
        result["open_time"]
    ):
        raise MarketStructureError(
            "open_time must be datetime dtype"
        )
    row_count = len(result)
    # ========================================================
    # OUTPUT COLUMN INITIALIZATION
    # ========================================================
    result["confirmed_swing_high"] = False
    result["confirmed_swing_low"] = False
    result["swing_event"] = False
    result["swing_kind"] = pd.Series(
        [pd.NA] * row_count,
        dtype="string",
    )
    result["swing_price"] = np.nan
    datetime_dtype = result[
        "open_time"
    ].dtype
    result["swing_time"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype=datetime_dtype,
    )
    result["swing_confirmation_time"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype=datetime_dtype,
    )
    result["swing_window_valid"] = False
    result["structure_label"] = pd.Series(
        [pd.NA] * row_count,
        dtype="string",
    )
    # ========================================================
    # WINDOW CONFIGURATION
    # ========================================================
    full_window_size = (
        left_bars
        + 1
        + right_bars
    )
    transition_count = (
        full_window_size
        - 1
    )
    if row_count < full_window_size:
        return _finalize_market_structure_state(
            result
        )
    # ========================================================
    # VECTORIZE PRICE WINDOWS
    # ========================================================
    highs = (
        result["high"]
        .to_numpy(
            dtype="float64"
        )
    )
    lows = (
        result["low"]
        .to_numpy(
            dtype="float64"
        )
    )
    high_windows = sliding_window_view(
        highs,
        window_shape=full_window_size,
    )
    low_windows = sliding_window_view(
        lows,
        window_shape=full_window_size,
    )
    pivot_highs = (
        high_windows[
            :,
            left_bars,
        ]
    )
    pivot_lows = (
        low_windows[
            :,
            left_bars,
        ]
    )
    left_high_max = (
        high_windows[
            :,
            :left_bars,
        ]
        .max(
            axis=1
        )
    )
    right_high_max = (
        high_windows[
            :,
            left_bars + 1:,
        ]
        .max(
            axis=1
        )
    )
    left_low_min = (
        low_windows[
            :,
            :left_bars,
        ]
        .min(
            axis=1
        )
    )
    right_low_min = (
        low_windows[
            :,
            left_bars + 1:,
        ]
        .min(
            axis=1
        )
    )
    candidate_swing_high = (
        (pivot_highs > left_high_max)
        & (pivot_highs > right_high_max)
    )
    candidate_swing_low = (
        (pivot_lows < left_low_min)
        & (pivot_lows < right_low_min)
    )
    # ========================================================
    # VECTORIZE TIME-CONTINUITY VALIDATION
    #
    # Do NOT convert datetime to raw integers here.
    # Pandas may use ns/us resolution depending on version.
    # Instead calculate actual elapsed hours explicitly.
    # ========================================================
    delta_hours = (
        result["open_time"]
        .diff()
        .dt.total_seconds()
        .to_numpy(
            dtype="float64"
        )
        / 3600.0
    )
    # First element is NaN because there is no previous row.
    transition_hours = (
        delta_hours[1:]
    )
    transition_valid = np.isclose(
        transition_hours,
        float(
            expected_interval_hours
        ),
        rtol=0.0,
        atol=1e-9,
    )
    transition_windows = sliding_window_view(
        transition_valid,
        window_shape=transition_count,
    )
    window_valid = (
        transition_windows
        .all(
            axis=1
        )
    )
    if (
        len(candidate_swing_high)
        != len(window_valid)
    ):
        raise MarketStructureError(
            "Internal market-structure "
            "window alignment error"
        )
    swing_high_mask = (
        candidate_swing_high
        & window_valid
    )
    swing_low_mask = (
        candidate_swing_low
        & window_valid
    )
    # ========================================================
    # MAP WINDOW POSITIONS TO PIVOT / CONFIRMATION ROWS
    # ========================================================
    window_start_indices = np.arange(
        len(window_valid),
        dtype="int64",
    )
    pivot_indices = (
        window_start_indices
        + left_bars
    )
    confirmation_indices = (
        pivot_indices
        + right_bars
    )
    high_positions = np.flatnonzero(
        swing_high_mask
    )
    low_positions = np.flatnonzero(
        swing_low_mask
    )
    high_pivot_indices = (
        pivot_indices[
            high_positions
        ]
    )
    high_confirmation_indices = (
        confirmation_indices[
            high_positions
        ]
    )
    low_pivot_indices = (
        pivot_indices[
            low_positions
        ]
    )
    low_confirmation_indices = (
        confirmation_indices[
            low_positions
        ]
    )
    # ========================================================
    # DUAL HIGH + LOW PIVOTS
    #
    # Rare outside-bar situation.
    # ========================================================
    dual_positions = np.flatnonzero(
        swing_high_mask
        & swing_low_mask
    )
    dual_confirmation_indices = (
        confirmation_indices[
            dual_positions
        ]
    )
    dual_confirmation_set = set(
        int(value)
        for value in (
            dual_confirmation_indices
        )
    )
    # ========================================================
    # BOOLEAN EVENT FLAGS
    # ========================================================
    if (
        len(
            high_confirmation_indices
        )
        > 0
    ):
        result.loc[
            high_confirmation_indices,
            "confirmed_swing_high",
        ] = True
    if (
        len(
            low_confirmation_indices
        )
        > 0
    ):
        result.loc[
            low_confirmation_indices,
            "confirmed_swing_low",
        ] = True
    all_event_indices = np.union1d(
        high_confirmation_indices,
        low_confirmation_indices,
    )
    if len(all_event_indices) > 0:
        result.loc[
            all_event_indices,
            "swing_event",
        ] = True
        result.loc[
            all_event_indices,
            "swing_window_valid",
        ] = True
        result.loc[
            all_event_indices,
            "swing_confirmation_time",
        ] = result.loc[
            all_event_indices,
            "open_time",
        ].to_numpy()
    # ========================================================
    # HIGH SWING EVENTS
    # ========================================================
    normal_high_mask = np.array(
        [
            int(confirm_index)
            not in dual_confirmation_set
            for confirm_index
            in high_confirmation_indices
        ],
        dtype=bool,
    )
    normal_high_pivots = (
        high_pivot_indices[
            normal_high_mask
        ]
    )
    normal_high_confirmations = (
        high_confirmation_indices[
            normal_high_mask
        ]
    )
    if (
        len(
            normal_high_confirmations
        )
        > 0
    ):
        result.loc[
            normal_high_confirmations,
            "swing_kind",
        ] = "high"
        result.loc[
            normal_high_confirmations,
            "swing_price",
        ] = result.loc[
            normal_high_pivots,
            "high",
        ].to_numpy()
        result.loc[
            normal_high_confirmations,
            "swing_time",
        ] = result.loc[
            normal_high_pivots,
            "open_time",
        ].to_numpy()
    # ========================================================
    # LOW SWING EVENTS
    # ========================================================
    normal_low_mask = np.array(
        [
            int(confirm_index)
            not in dual_confirmation_set
            for confirm_index
            in low_confirmation_indices
        ],
        dtype=bool,
    )
    normal_low_pivots = (
        low_pivot_indices[
            normal_low_mask
        ]
    )
    normal_low_confirmations = (
        low_confirmation_indices[
            normal_low_mask
        ]
    )
    if (
        len(
            normal_low_confirmations
        )
        > 0
    ):
        result.loc[
            normal_low_confirmations,
            "swing_kind",
        ] = "low"
        result.loc[
            normal_low_confirmations,
            "swing_price",
        ] = result.loc[
            normal_low_pivots,
            "low",
        ].to_numpy()
        result.loc[
            normal_low_confirmations,
            "swing_time",
        ] = result.loc[
            normal_low_pivots,
            "open_time",
        ].to_numpy()
    # ========================================================
    # HH / LH / HL / LL CLASSIFICATION
    #
    # This loop runs only over confirmed scalar swing events.
    # ========================================================
    previous_high: float | None = None
    previous_low: float | None = None
    scalar_event_mask = (
        result[
            "swing_event"
        ]
        & result[
            "swing_kind"
        ]
        .notna()
    )
    scalar_event_indices = (
        result.index[
            scalar_event_mask
        ]
    )
    for index in scalar_event_indices:
        swing_kind = result.at[
            index,
            "swing_kind",
        ]
        swing_price = result.at[
            index,
            "swing_price",
        ]
        if pd.isna(
            swing_price
        ):
            continue
        swing_price = float(
            swing_price
        )
        # ----------------------------------------------------
        # High structure
        # ----------------------------------------------------
        if swing_kind == "high":
            if (
                previous_high
                is not None
            ):
                if (
                    swing_price
                    > previous_high
                ):
                    label = "HH"
                elif (
                    swing_price
                    < previous_high
                ):
                    label = "LH"
                else:
                    label = "EH"
                result.at[
                    index,
                    "structure_label",
                ] = label
            previous_high = (
                swing_price
            )
        # ----------------------------------------------------
        # Low structure
        # ----------------------------------------------------
        elif swing_kind == "low":
            if (
                previous_low
                is not None
            ):
                if (
                    swing_price
                    > previous_low
                ):
                    label = "HL"
                elif (
                    swing_price
                    < previous_low
                ):
                    label = "LL"
                else:
                    label = "EL"
                result.at[
                    index,
                    "structure_label",
                ] = label
            previous_low = (
                swing_price
            )
    return _finalize_market_structure_state(
        result
    )
def _finalize_market_structure_state(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill latest confirmed swing state.
    Nullable string masks are normalized to strict boolean
    Series before use.
    """
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    # ========================================================
    # SAFE BOOLEAN MASKS
    # ========================================================
    high_kind_mask = (
        result[
            "swing_kind"
        ]
        .eq("high")
        .fillna(False)
        .astype(bool)
    )
    low_kind_mask = (
        result[
            "swing_kind"
        ]
        .eq("low")
        .fillna(False)
        .astype(bool)
    )
    confirmed_high_mask = (
        result[
            "confirmed_swing_high"
        ]
        .fillna(False)
        .astype(bool)
    )
    confirmed_low_mask = (
        result[
            "confirmed_swing_low"
        ]
        .fillna(False)
        .astype(bool)
    )
    high_event_mask = (
        confirmed_high_mask
        & high_kind_mask
    )
    low_event_mask = (
        confirmed_low_mask
        & low_kind_mask
    )
    # ========================================================
    # LAST CONFIRMED HIGH
    # ========================================================
    result[
        "last_swing_high"
    ] = (
        result[
            "swing_price"
        ]
        .where(
            high_event_mask
        )
        .ffill()
    )
    result[
        "last_swing_high_time"
    ] = (
        result[
            "swing_time"
        ]
        .where(
            high_event_mask
        )
        .ffill()
    )
    # ========================================================
    # LAST CONFIRMED LOW
    # ========================================================
    result[
        "last_swing_low"
    ] = (
        result[
            "swing_price"
        ]
        .where(
            low_event_mask
        )
        .ffill()
    )
    result[
        "last_swing_low_time"
    ] = (
        result[
            "swing_time"
        ]
        .where(
            low_event_mask
        )
        .ffill()
    )
    # ========================================================
    # TIME SINCE ACTUAL PIVOT
    # ========================================================
    result[
        "hours_since_swing_high"
    ] = (
        (
            result[
                "open_time"
            ]
            - result[
                "last_swing_high_time"
            ]
        )
        .dt.total_seconds()
        / 3600.0
    )
    result[
        "hours_since_swing_low"
    ] = (
        (
            result[
                "open_time"
            ]
            - result[
                "last_swing_low_time"
            ]
        )
        .dt.total_seconds()
        / 3600.0
    )
    return result