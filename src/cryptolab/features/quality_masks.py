from __future__ import annotations
import numpy as np
import pandas as pd
class QualityMaskError(ValueError):
    """Raised when feature quality-mask generation fails."""
# Hour-based windows used by the Price Layer.
QUALITY_HORIZONS = {
    "24h": 24,
    "7d": 168,
    "30d": 720,
    "90d": 2160,
    "1y": 8760,
}
def add_price_quality_masks(
    df: pd.DataFrame,
    expected_interval_hours: int = 1,
) -> pd.DataFrame:
    """
    Add continuity and feature-quality masks to hourly price data.
    This function does NOT modify or fabricate missing candles.
    Instead, it records whether each rolling feature window
    crosses a source-data gap.
    Generated core fields
    ---------------------
    source_interval_hours:
        Actual elapsed hours since the previous stored candle.
    source_gap_before:
        True when the elapsed interval is greater than the
        expected 1-hour interval.
    source_irregular_before:
        True whenever the timestamp interval differs from
        exactly one hour.
    missing_hours_before:
        Approximate number of missing hourly candles between
        the previous stored candle and the current candle.
    For each horizon:
        gap_count_<horizon>
        elapsed_hours_<horizon>
        quality_<horizon>
    Example:
        quality_30d == True
    means the 30-day feature window is based on an uninterrupted
    hourly series rather than merely 720 stored observations.
    Feature-specific validity fields are also generated when the
    corresponding feature exists.
    """
    if df.empty:
        raise QualityMaskError(
            "Input dataframe is empty"
        )
    required_columns = [
        "open_time",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise QualityMaskError(
            f"Missing columns: {missing_columns}"
        )
    if expected_interval_hours <= 0:
        raise QualityMaskError(
            "expected_interval_hours must be positive"
        )
    result = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if not pd.api.types.is_datetime64_any_dtype(
        result["open_time"]
    ):
        raise QualityMaskError(
            "open_time must be datetime dtype"
        )
    # --------------------------------------------------------
    # SOURCE-LEVEL CONTINUITY
    # --------------------------------------------------------
    delta = (
        result["open_time"]
        .diff()
    )
    result["source_interval_hours"] = (
        delta
        .dt.total_seconds()
        / 3600.0
    )
    interval = (
        result["source_interval_hours"]
    )
    result["source_gap_before"] = (
        interval
        > expected_interval_hours
    )
    result["source_irregular_before"] = (
        interval.notna()
        & ~np.isclose(
            interval,
            expected_interval_hours,
            rtol=0.0,
            atol=1e-9,
        )
    )
    missing_hours = (
        interval
        - expected_interval_hours
    )
    result["missing_hours_before"] = (
        np.where(
            missing_hours > 0,
            np.rint(
                missing_hours
            ),
            0,
        )
        .astype("int64")
    )
    # First observation has no previous observation.
    result.loc[
        0,
        "source_gap_before",
    ] = False
    result.loc[
        0,
        "source_irregular_before",
    ] = False
    result.loc[
        0,
        "missing_hours_before",
    ] = 0
    # Convert booleans to integers for rolling sums.
    gap_indicator = (
        result["source_gap_before"]
        .astype("int64")
    )
    # --------------------------------------------------------
    # GENERIC HORIZON QUALITY
    # --------------------------------------------------------
    for name, hours in (
        QUALITY_HORIZONS.items()
    ):
        gap_count_column = (
            f"gap_count_{name}"
        )
        elapsed_column = (
            f"elapsed_hours_{name}"
        )
        quality_column = (
            f"quality_{name}"
        )
        # The last H transitions correspond to the feature
        # horizon used by shift(H) or rolling(H).
        result[
            gap_count_column
        ] = (
            gap_indicator
            .rolling(
                window=hours,
                min_periods=hours,
            )
            .sum()
        )
        shifted_time = (
            result["open_time"]
            .shift(hours)
        )
        result[
            elapsed_column
        ] = (
            (
                result["open_time"]
                - shifted_time
            )
            .dt.total_seconds()
            / 3600.0
        )
        result[
            quality_column
        ] = (
            result[
                gap_count_column
            ].eq(0)
            & np.isclose(
                result[
                    elapsed_column
                ],
                hours,
                rtol=0.0,
                atol=1e-9,
                equal_nan=False,
            )
        )
    # --------------------------------------------------------
    # FEATURE-SPECIFIC QUALITY MASKS
    # --------------------------------------------------------
    feature_quality_map = {
        "return_24h": "quality_24h",
        "return_7d": "quality_7d",
        "return_30d": "quality_30d",
        "rolling_high_24": "quality_24h",
        "rolling_low_24": "quality_24h",
        "rolling_range_24": "quality_24h",
        "range_position_24": "quality_24h",
        "rolling_high_168": "quality_7d",
        "rolling_low_168": "quality_7d",
        "rolling_range_168": "quality_7d",
        "range_position_168": "quality_7d",
        "rolling_high_720": "quality_30d",
        "rolling_low_720": "quality_30d",
        "rolling_range_720": "quality_30d",
        "range_position_720": "quality_30d",
        "rv_24h": "quality_24h",
        "rv_7d": "quality_7d",
        "rv_30d": "quality_30d",
        "rv_90d": "quality_90d",
    }
    for feature, quality_column in (
        feature_quality_map.items()
    ):
        if feature not in result.columns:
            continue
        validity_column = (
            f"{feature}_valid"
        )
        result[
            validity_column
        ] = (
            result[feature].notna()
            & result[
                quality_column
            ]
        )
    # --------------------------------------------------------
    # ATR QUALITY
    #
    # ATR14 on hourly data requires 14 uninterrupted
    # hourly transitions.
    # --------------------------------------------------------
    if "atr_14" in result.columns:
        atr_gap_count = (
            gap_indicator
            .rolling(
                window=14,
                min_periods=14,
            )
            .sum()
        )
        result[
            "atr_14_valid"
        ] = (
            result["atr_14"].notna()
            & atr_gap_count.eq(0)
        )
    if "atr_14_pct" in result.columns:
        result[
            "atr_14_pct_valid"
        ] = (
            result["atr_14_pct"].notna()
            & result.get(
                "atr_14_valid",
                False,
            )
        )
    # --------------------------------------------------------
    # ONE-YEAR RV DISTRIBUTION QUALITY
    #
    # The percentile and z-score use a trailing one-year
    # sequence of rv_30d values.
    #
    # Rather than merely checking one-year timestamp
    # continuity, require every rv_30d observation used
    # in the distribution window itself to be valid.
    # --------------------------------------------------------
    if "rv_30d_valid" in result.columns:
        valid_rv30_count = (
            result[
                "rv_30d_valid"
            ]
            .astype("int64")
            .rolling(
                window=QUALITY_HORIZONS[
                    "1y"
                ],
                min_periods=QUALITY_HORIZONS[
                    "1y"
                ],
            )
            .sum()
        )
        full_valid_rv30_window = (
            valid_rv30_count
            == QUALITY_HORIZONS[
                "1y"
            ]
        )
        if (
            "rv_30d_percentile_1y"
            in result.columns
        ):
            result[
                "rv_30d_percentile_1y_valid"
            ] = (
                result[
                    "rv_30d_percentile_1y"
                ].notna()
                & full_valid_rv30_window
            )
        if (
            "rv_30d_zscore_1y"
            in result.columns
        ):
            result[
                "rv_30d_zscore_1y_valid"
            ] = (
                result[
                    "rv_30d_zscore_1y"
                ].notna()
                & full_valid_rv30_window
            )
    # --------------------------------------------------------
    # VWAP QUALITY
    #
    # Session VWAP is cumulative from the session start.
    # Therefore its validity depends on whether any source
    # gap occurred inside the current session before the
    # current observation.
    # --------------------------------------------------------
    result["_daily_session"] = (
        result["open_time"]
        .dt.floor("D")
    )
    weekday = (
        result["open_time"]
        .dt.weekday
    )
    result["_weekly_session"] = (
        result["open_time"]
        .dt.floor("D")
        - pd.to_timedelta(
            weekday,
            unit="D",
        )
    )
    result["_monthly_session"] = (
        pd.to_datetime(
            {
                "year": (
                    result["open_time"]
                    .dt.year
                ),
                "month": (
                    result["open_time"]
                    .dt.month
                ),
                "day": 1,
            },
            utc=True,
        )
    )
    session_map = {
        "daily": "_daily_session",
        "weekly": "_weekly_session",
        "monthly": "_monthly_session",
    }
    for session, key_column in (
        session_map.items()
    ):
        session_gap_count = (
            result
            .groupby(
                key_column,
                sort=False,
            )["source_gap_before"]
            .cumsum()
        )
        quality_column = (
            f"{session}_vwap_valid"
        )
        vwap_column = (
            f"{session}_vwap"
        )
        if vwap_column in result.columns:
            result[
                quality_column
            ] = (
                result[
                    vwap_column
                ].notna()
                & session_gap_count.eq(0)
            )
        distance_column = (
            f"distance_to_{session}_vwap"
        )
        if distance_column in result.columns:
            result[
                f"{distance_column}_valid"
            ] = (
                result[
                    distance_column
                ].notna()
                & result.get(
                    quality_column,
                    False,
                )
            )
        distance_pct_column = (
            f"distance_to_{session}_vwap_pct"
        )
        if (
            distance_pct_column
            in result.columns
        ):
            result[
                f"{distance_pct_column}_valid"
            ] = (
                result[
                    distance_pct_column
                ].notna()
                & result.get(
                    quality_column,
                    False,
                )
            )
    result = result.drop(
        columns=[
            "_daily_session",
            "_weekly_session",
            "_monthly_session",
        ]
    )
    return result
