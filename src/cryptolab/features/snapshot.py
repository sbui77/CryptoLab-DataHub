from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
class PriceSnapshotError(ValueError):
    """Raised when Price Layer snapshot generation fails."""
@dataclass(frozen=True)
class PriceSnapshot:
    """
    Lightweight representation of the latest Price Layer state.
    """
    exchange: str
    symbol: str
    timeframe: str
    open_time: pd.Timestamp
    close: float
    simple_return: float | None
    return_24h: float | None
    return_7d: float | None
    return_30d: float | None
    atr_14_pct: float | None
    rv_24h: float | None
    rv_7d: float | None
    rv_30d: float | None
    rv_90d: float | None
    rv_30d_percentile_1y: float | None
    rv_30d_zscore_1y: float | None
    daily_vwap: float | None
    weekly_vwap: float | None
    monthly_vwap: float | None
    distance_to_daily_vwap_pct: float | None
    distance_to_weekly_vwap_pct: float | None
    distance_to_monthly_vwap_pct: float | None
    structure_state: str | None
    last_swing_high: float | None
    last_swing_low: float | None
    active_resistance: float | None
    active_support: float | None
    structural_range: float | None
    structural_range_pct: float | None
    structural_range_position: float | None
    range_state: str | None
    nearest_structure_side: str | None
    nearest_structure_distance_pct: float | None
    price_regime: str | None
    volatility_regime: str | None
    location_regime: str | None
    regime_confidence: float | None
    price_regime_valid: bool
    quality_24h: bool
    quality_7d: bool
    quality_30d: bool
    quality_90d: bool
    latest_bos_direction: str | None
    latest_bos_time: pd.Timestamp | None
    latest_choch_direction: str | None
    latest_choch_time: pd.Timestamp | None
def _optional_float(
    value: Any,
) -> float | None:
    """
    Convert pandas / numpy scalar to Python float.
    Missing values become None.
    """
    if pd.isna(value):
        return None
    return float(value)
def _optional_string(
    value: Any,
) -> str | None:
    """
    Convert pandas nullable string to Python string.
    """
    if pd.isna(value):
        return None
    return str(value)
def _optional_timestamp(
    value: Any,
) -> pd.Timestamp | None:
    """
    Convert timestamp-like value into pd.Timestamp.
    """
    if pd.isna(value):
        return None
    return pd.Timestamp(value)
def _latest_event(
    df: pd.DataFrame,
    flag_column: str,
    direction_column: str,
) -> tuple[str | None, pd.Timestamp | None]:
    """
    Return the latest event direction and event time.
    """
    if (
        flag_column not in df.columns
        or direction_column not in df.columns
    ):
        return None, None
    mask = (
        df[flag_column]
        .fillna(False)
        .astype(bool)
    )
    events = df.loc[
        mask,
        [
            "open_time",
            direction_column,
        ],
    ]
    if events.empty:
        return None, None
    row = events.iloc[-1]
    return (
        _optional_string(
            row[direction_column]
        ),
        _optional_timestamp(
            row["open_time"]
        ),
    )
def build_price_snapshot(
    df: pd.DataFrame,
) -> PriceSnapshot:
    """
    Build the latest Price Layer state snapshot.
    The input dataframe is expected to be the complete
    curated Price Layer after Steps 6.1–6.9.
    The snapshot deliberately contains descriptive state
    only. It does not produce BUY / SELL decisions.
    """
    if df.empty:
        raise PriceSnapshotError(
            "Input dataframe is empty"
        )
    required_columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "close",
        "simple_return",
        "return_24h",
        "return_7d",
        "return_30d",
        "atr_14_pct",
        "rv_24h",
        "rv_7d",
        "rv_30d",
        "rv_90d",
        "rv_30d_percentile_1y",
        "rv_30d_zscore_1y",
        "daily_vwap",
        "weekly_vwap",
        "monthly_vwap",
        "distance_to_daily_vwap_pct",
        "distance_to_weekly_vwap_pct",
        "distance_to_monthly_vwap_pct",
        "structure_state",
        "last_swing_high",
        "last_swing_low",
        "active_resistance",
        "active_support",
        "structural_range",
        "structural_range_pct",
        "structural_range_position",
        "range_state",
        "nearest_structure_side",
        "nearest_structure_distance_pct",
        "price_regime",
        "volatility_regime",
        "location_regime",
        "regime_confidence",
        "price_regime_valid",
        "quality_24h",
        "quality_7d",
        "quality_30d",
        "quality_90d",
        "bos",
        "bos_direction",
        "choch",
        "choch_direction",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise PriceSnapshotError(
            f"Missing columns: {missing_columns}"
        )
    work = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    latest = work.iloc[-1]
    latest_bos_direction, latest_bos_time = (
        _latest_event(
            work,
            flag_column="bos",
            direction_column="bos_direction",
        )
    )
    latest_choch_direction, latest_choch_time = (
        _latest_event(
            work,
            flag_column="choch",
            direction_column="choch_direction",
        )
    )
    return PriceSnapshot(
        exchange=str(
            latest["exchange"]
        ),
        symbol=str(
            latest["symbol"]
        ),
        timeframe=str(
            latest["timeframe"]
        ),
        open_time=pd.Timestamp(
            latest["open_time"]
        ),
        close=float(
            latest["close"]
        ),
        simple_return=_optional_float(
            latest["simple_return"]
        ),
        return_24h=_optional_float(
            latest["return_24h"]
        ),
        return_7d=_optional_float(
            latest["return_7d"]
        ),
        return_30d=_optional_float(
            latest["return_30d"]
        ),
        atr_14_pct=_optional_float(
            latest["atr_14_pct"]
        ),
        rv_24h=_optional_float(
            latest["rv_24h"]
        ),
        rv_7d=_optional_float(
            latest["rv_7d"]
        ),
        rv_30d=_optional_float(
            latest["rv_30d"]
        ),
        rv_90d=_optional_float(
            latest["rv_90d"]
        ),
        rv_30d_percentile_1y=_optional_float(
            latest[
                "rv_30d_percentile_1y"
            ]
        ),
        rv_30d_zscore_1y=_optional_float(
            latest[
                "rv_30d_zscore_1y"
            ]
        ),
        daily_vwap=_optional_float(
            latest["daily_vwap"]
        ),
        weekly_vwap=_optional_float(
            latest["weekly_vwap"]
        ),
        monthly_vwap=_optional_float(
            latest["monthly_vwap"]
        ),
        distance_to_daily_vwap_pct=_optional_float(
            latest[
                "distance_to_daily_vwap_pct"
            ]
        ),
        distance_to_weekly_vwap_pct=_optional_float(
            latest[
                "distance_to_weekly_vwap_pct"
            ]
        ),
        distance_to_monthly_vwap_pct=_optional_float(
            latest[
                "distance_to_monthly_vwap_pct"
            ]
        ),
        structure_state=_optional_string(
            latest[
                "structure_state"
            ]
        ),
        last_swing_high=_optional_float(
            latest[
                "last_swing_high"
            ]
        ),
        last_swing_low=_optional_float(
            latest[
                "last_swing_low"
            ]
        ),
        active_resistance=_optional_float(
            latest[
                "active_resistance"
            ]
        ),
        active_support=_optional_float(
            latest[
                "active_support"
            ]
        ),
        structural_range=_optional_float(
            latest[
                "structural_range"
            ]
        ),
        structural_range_pct=_optional_float(
            latest[
                "structural_range_pct"
            ]
        ),
        structural_range_position=_optional_float(
            latest[
                "structural_range_position"
            ]
        ),
        range_state=_optional_string(
            latest[
                "range_state"
            ]
        ),
        nearest_structure_side=_optional_string(
            latest[
                "nearest_structure_side"
            ]
        ),
        nearest_structure_distance_pct=_optional_float(
            latest[
                "nearest_structure_distance_pct"
            ]
        ),
        price_regime=_optional_string(
            latest[
                "price_regime"
            ]
        ),
        volatility_regime=_optional_string(
            latest[
                "volatility_regime"
            ]
        ),
        location_regime=_optional_string(
            latest[
                "location_regime"
            ]
        ),
        regime_confidence=_optional_float(
            latest[
                "regime_confidence"
            ]
        ),
        price_regime_valid=bool(
            latest[
                "price_regime_valid"
            ]
        ),
        quality_24h=bool(
            latest["quality_24h"]
        ),
        quality_7d=bool(
            latest["quality_7d"]
        ),
        quality_30d=bool(
            latest["quality_30d"]
        ),
        quality_90d=bool(
            latest["quality_90d"]
        ),
        latest_bos_direction=(
            latest_bos_direction
        ),
        latest_bos_time=(
            latest_bos_time
        ),
        latest_choch_direction=(
            latest_choch_direction
        ),
        latest_choch_time=(
            latest_choch_time
        ),
    )
def price_snapshot_to_dataframe(
    snapshot: PriceSnapshot,
) -> pd.DataFrame:
    """
    Convert PriceSnapshot into a single-row dataframe.
    """
    data = {
        field: getattr(
            snapshot,
            field,
        )
        for field in (
            snapshot.__dataclass_fields__
        )
    }
    return pd.DataFrame(
        [data]
    )
