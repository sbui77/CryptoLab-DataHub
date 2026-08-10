import pandas as pd
from cryptolab.features.structural_state import (
    add_structural_state_features,
)
def make_structural_df() -> pd.DataFrame:
    """
    Synthetic confirmed-swing state designed to produce:
        initial bullish break
        bullish BOS
        bearish CHOCH
        bearish BOS
    """
    open_time = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=12,
        freq="1h",
    )
    close = [
        100.0,
        101.0,
        106.0,
        107.0,
        108.0,
        112.0,
        111.0,
        104.0,
        93.0,
        92.0,
        88.0,
        87.0,
    ]
    # These represent already-confirmed swing states.
    last_swing_high = [
        None,
        105.0,
        105.0,
        105.0,
        110.0,
        110.0,
        110.0,
        110.0,
        110.0,
        110.0,
        110.0,
        110.0,
    ]
    last_swing_low = [
        None,
        95.0,
        95.0,
        95.0,
        95.0,
        95.0,
        95.0,
        94.0,
        94.0,
        94.0,
        90.0,
        90.0,
    ]
    return pd.DataFrame(
        {
            "open_time": open_time,
            "close": close,
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,
        }
    )
def test_structural_state_columns():
    result = add_structural_state_features(
        make_structural_df()
    )
    expected_columns = [
        "break_reference_high",
        "break_reference_low",
        "structural_break",
        "break_direction",
        "break_level",
        "bos",
        "bos_direction",
        "choch",
        "choch_direction",
        "initial_structure_break",
        "structure_state",
        "state_changed",
    ]
    for column in expected_columns:
        assert column in result.columns
def test_initial_break_exists():
    result = add_structural_state_features(
        make_structural_df()
    )
    assert (
        result[
            "initial_structure_break"
        ].sum()
        > 0
    )
def test_bos_exists():
    result = add_structural_state_features(
        make_structural_df()
    )
    assert (
        result["bos"].sum()
        > 0
    )
def test_choch_exists():
    result = add_structural_state_features(
        make_structural_df()
    )
    assert (
        result["choch"].sum()
        > 0
    )
def test_structure_state_values():
    result = add_structural_state_features(
        make_structural_df()
    )
    allowed = {
        "neutral",
        "bullish",
        "bearish",
    }
    assert set(
        result[
            "structure_state"
        ].dropna()
    ).issubset(
        allowed
    )
def test_choch_changes_state():
    result = add_structural_state_features(
        make_structural_df()
    )
    choch_rows = result[
        result["choch"]
    ]
    assert not choch_rows.empty
    for _, row in choch_rows.iterrows():
        if (
            row[
                "choch_direction"
            ]
            == "bullish"
        ):
            assert (
                row[
                    "structure_state"
                ]
                == "bullish"
            )
        elif (
            row[
                "choch_direction"
            ]
            == "bearish"
        ):
            assert (
                row[
                    "structure_state"
                ]
                == "bearish"
            )
