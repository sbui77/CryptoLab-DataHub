from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
class TradeFlowSnapshotError(ValueError):
    """Raised when Trade Flow snapshot generation fails."""
@dataclass(frozen=True)
class TradeFlowSnapshot:
    """
    Lightweight latest Trade Flow state.
    """
    exchange: str
    symbol: str
    timeframe: str
    open_time: pd.Timestamp
    close: float
    trade_count: int
    buy_trade_ratio: float | None
    sell_trade_ratio: float | None
    trade_count_imbalance: float | None
    buy_quote_ratio: float | None
    sell_quote_ratio: float | None
    quote_delta: float | None
    quote_delta_pct: float | None
    base_delta: float | None
    base_delta_pct: float | None
    rolling_quote_imbalance_60: float | None
    base_cvd: float | None
    quote_cvd: float | None
    daily_base_cvd: float | None
    daily_quote_cvd: float | None
    rolling_base_cvd_60: float | None
    rolling_quote_cvd_60: float | None
    rolling_base_cvd_1440: float | None
    rolling_quote_cvd_1440: float | None
    large_trade_quote_share: float | None
    large_quote_delta: float | None
    large_quote_delta_pct: float | None
    trade_flow_regime: str | None
    flow_regime_confidence: float | None
    trade_flow_regime_valid: bool
def _optional_float(
    value: Any,
) -> float | None:
    """
    Convert nullable scalar to Python float.
    """
    if pd.isna(value):
        return None
    return float(value)
def _optional_string(
    value: Any,
) -> str | None:
    """
    Convert nullable scalar to Python string.
    """
    if pd.isna(value):
        return None
    return str(value)
def build_trade_flow_snapshot(
    df: pd.DataFrame,
) -> TradeFlowSnapshot:
    """
    Build the latest Trade Flow snapshot.
    Expects final 1m Trade Flow dataset after:
        aggregation
        imbalance
        CVD
        large trades
        trade-flow regime
    This is descriptive state only.
    It does not generate trading signals.
    """
    if df.empty:
        raise TradeFlowSnapshotError(
            "Input dataframe is empty"
        )
    required_columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "close",
        "trade_count",
        "buy_trade_ratio",
        "sell_trade_ratio",
        "trade_count_imbalance",
        "buy_quote_ratio",
        "sell_quote_ratio",
        "quote_delta",
        "quote_delta_pct",
        "base_delta",
        "base_delta_pct",
        "rolling_quote_imbalance_60",
        "base_cvd",
        "quote_cvd",
        "daily_base_cvd",
        "daily_quote_cvd",
        "rolling_base_cvd_60",
        "rolling_quote_cvd_60",
        "rolling_base_cvd_1440",
        "rolling_quote_cvd_1440",
        "large_trade_quote_share",
        "large_quote_delta",
        "large_quote_delta_pct",
        "trade_flow_regime",
        "flow_regime_confidence",
        "trade_flow_regime_valid",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise TradeFlowSnapshotError(
            f"Missing columns: {missing_columns}"
        )
    work = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    latest = work.iloc[-1]
    return TradeFlowSnapshot(
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
        trade_count=int(
            latest["trade_count"]
        ),
        buy_trade_ratio=_optional_float(
            latest["buy_trade_ratio"]
        ),
        sell_trade_ratio=_optional_float(
            latest["sell_trade_ratio"]
        ),
        trade_count_imbalance=_optional_float(
            latest["trade_count_imbalance"]
        ),
        buy_quote_ratio=_optional_float(
            latest["buy_quote_ratio"]
        ),
        sell_quote_ratio=_optional_float(
            latest["sell_quote_ratio"]
        ),
        quote_delta=_optional_float(
            latest["quote_delta"]
        ),
        quote_delta_pct=_optional_float(
            latest["quote_delta_pct"]
        ),
        base_delta=_optional_float(
            latest["base_delta"]
        ),
        base_delta_pct=_optional_float(
            latest["base_delta_pct"]
        ),
        rolling_quote_imbalance_60=_optional_float(
            latest[
                "rolling_quote_imbalance_60"
            ]
        ),
        base_cvd=_optional_float(
            latest["base_cvd"]
        ),
        quote_cvd=_optional_float(
            latest["quote_cvd"]
        ),
        daily_base_cvd=_optional_float(
            latest["daily_base_cvd"]
        ),
        daily_quote_cvd=_optional_float(
            latest["daily_quote_cvd"]
        ),
        rolling_base_cvd_60=_optional_float(
            latest["rolling_base_cvd_60"]
        ),
        rolling_quote_cvd_60=_optional_float(
            latest["rolling_quote_cvd_60"]
        ),
        rolling_base_cvd_1440=_optional_float(
            latest["rolling_base_cvd_1440"]
        ),
        rolling_quote_cvd_1440=_optional_float(
            latest["rolling_quote_cvd_1440"]
        ),
        large_trade_quote_share=_optional_float(
            latest[
                "large_trade_quote_share"
            ]
        ),
        large_quote_delta=_optional_float(
            latest["large_quote_delta"]
        ),
        large_quote_delta_pct=_optional_float(
            latest[
                "large_quote_delta_pct"
            ]
        ),
        trade_flow_regime=_optional_string(
            latest[
                "trade_flow_regime"
            ]
        ),
        flow_regime_confidence=_optional_float(
            latest[
                "flow_regime_confidence"
            ]
        ),
        trade_flow_regime_valid=bool(
            latest[
                "trade_flow_regime_valid"
            ]
        ),
    )
def trade_flow_snapshot_to_dataframe(
    snapshot: TradeFlowSnapshot,
) -> pd.DataFrame:
    """
    Convert snapshot into a one-row dataframe.
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
