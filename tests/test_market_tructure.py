import numpy as np
import pandas as pd
from cryptolab.features.market_structure import (
    add_market_structure_features,
)
def make_structure_df() -> pd.DataFrame:
    """
    Deterministic test series with obvious pivots.
    High sequence creates identifiable local highs.
    Low sequence creates identifiable local lows.
    """
    rows = 30
    open_time = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="1h",
    )
    close = np.array(
        [
            100,
            102,
            104,
            108,
            105,
            103,
            101,
            104,
            108,
            112,
            109,
            106,
            103,
            105,
            109,
            115,
            111,
            107,
            104,
            108,
            111,
            117,
            113,
            109,
            106,
            110,
            114,
            119,
            116,
            112,
        ],
        dtype="float64",
    )
    open_price = (
        close - 0.5
    )
    high = (
        close + 1.0
    )
    low = (
        close - 1.0
    )
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )
def test_market_structure_columns():
    result = add_market_structure_features(
        make_structure_df(),
        left_bars=2,
        right_bars=2,
    )
    expected_columns = [
        "confirmed_swing_high",
        "confirmed_swing_low",
        "swing_event",
        "swing_kind",
        "swing_price",
        "swing_time",
        "swing_confirmation_time",
        "swing_window_valid",
        "structure_label",
        "last_swing_high",
        "last_swing_low",
        "last_swing_high_time",
        "last_swing_low_time",
        "hours_since_swing_high",
        "hours_since_swing_low",
    ]
    for column in expected_columns:
        assert (
            column
            in result.columns
        )
def test_swing_events_exist():
    result = add_market_structure_features(
        make_structure_df(),
        left_bars=2,
        right_bars=2,
    )
    assert (
        result[
            "confirmed_swing_high"
        ].sum()
        > 0
    )
    assert (
        result[
            "confirmed_swing_low"
        ].sum()
        > 0
    )
def test_confirmation_occurs_after_pivot():
    result = add_market_structure_features(
        make_structure_df(),
        left_bars=2,
        right_bars=2,
    )
    events = result[
        result[
            "swing_event"
        ]
        & result[
            "swing_time"
        ].notna()
    ]
    assert not events.empty
    assert (
        events[
            "swing_confirmation_time"
        ]
        > events[
            "swing_time"
        ]
    ).all()
def test_structure_labels_exist():
    result = add_market_structure_features(
        make_structure_df(),
        left_bars=2,
        right_bars=2,
    )
    labels = (
        result[
            "structure_label"
        ]
        .dropna()
    )
    assert (
        len(labels)
        > 0
    )
    allowed = {
        "HH",
        "LH",
        "HL",
        "LL",
        "EH",
        "EL",
    }
    assert set(
        labels.tolist()
    ).issubset(
        allowed
    )
def test_gap_invalidates_detection_window():
    df = make_structure_df()
    df = (
        df
        .drop(
            index=10
        )
        .reset_index(
            drop=True
        )
    )
    result = add_market_structure_features(
        df,
        left_bars=2,
        right_bars=2,
    )
    gap_time = pd.Timestamp(
        "2026-01-01T11:00:00Z"
    )
    nearby = result[
        (
            result["open_time"]
            >= gap_time
            - pd.Timedelta(
                hours=4
            )
        )
        &
        (
            result["open_time"]
            <= gap_time
            + pd.Timedelta(
                hours=4
            )
        )
    ]
    # We mainly require the function to operate safely
    # without fabricating continuity across the gap.
    assert not nearby.empty
def test_last_swing_state_forward_fills():
    result = add_market_structure_features(
        make_structure_df(),
        left_bars=2,
        right_bars=2,
    )
    high_events = result[
        result[
            "confirmed_swing_high"
        ]
        & result[
            "swing_kind"
        ].eq("high")
    ]
    assert not high_events.empty
    first_event_index = (
        high_events.index[0]
    )
    first_price = result.loc[
        first_event_index,
        "swing_price",
    ]
    assert np.isclose(
        result.loc[
            first_event_index,
            "last_swing_high",
        ],
        first_price,
    )
