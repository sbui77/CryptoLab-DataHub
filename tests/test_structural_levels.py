import numpy as np
import pandas as pd
from cryptolab.features.structural_levels import (
    add_structural_level_features,
)
def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=6,
                freq="1h",
            ),
            "close": [
                100.0,
                105.0,
                110.0,
                115.0,
                121.0,
                89.0,
            ],
            "last_swing_high": [
                np.nan,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
            ],
            "last_swing_low": [
                np.nan,
                90.0,
                90.0,
                90.0,
                90.0,
                90.0,
            ],
        }
    )
def test_structural_level_columns():
    result = add_structural_level_features(
        make_df()
    )
    expected = [
        "active_resistance",
        "active_support",
        "distance_to_resistance",
        "distance_to_support",
        "distance_to_resistance_pct",
        "distance_to_support_pct",
        "structural_range",
        "structural_range_pct",
        "structural_range_position",
        "range_state",
        "nearest_structure_side",
        "nearest_structure_distance_pct",
        "structural_range_valid",
    ]
    for column in expected:
        assert column in result.columns
def test_structural_range():
    result = add_structural_level_features(
        make_df()
    )
    assert np.isclose(
        result.loc[
            1,
            "structural_range",
        ],
        30.0,
    )
def test_inside_range_position():
    result = add_structural_level_features(
        make_df()
    )
    position = result.loc[
        2,
        "structural_range_position",
    ]
    expected = (
        110.0 - 90.0
    ) / (
        120.0 - 90.0
    )
    assert np.isclose(
        position,
        expected,
    )
    assert (
        result.loc[
            2,
            "range_state",
        ]
        == "inside_range"
    )
def test_above_resistance():
    result = add_structural_level_features(
        make_df()
    )
    assert (
        result.loc[
            4,
            "range_state",
        ]
        == "above_resistance"
    )
    assert (
        result.loc[
            4,
            "structural_range_position",
        ]
        > 1.0
    )
def test_below_support():
    result = add_structural_level_features(
        make_df()
    )
    assert (
        result.loc[
            5,
            "range_state",
        ]
        == "below_support"
    )
    assert (
        result.loc[
            5,
            "structural_range_position",
        ]
        < 0.0
    )
def test_nearest_structure_side():
    result = add_structural_level_features(
        make_df()
    )
    # Close=115 is closer to resistance=120
    # than support=90.
    assert (
        result.loc[
            3,
            "nearest_structure_side",
        ]
        == "resistance"
    )
    assert (
        result.loc[
            3,
            "nearest_structure_distance_pct",
        ]
        >= 0
    )
