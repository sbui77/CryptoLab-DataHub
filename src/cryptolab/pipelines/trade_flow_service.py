from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from cryptolab.features.large_trades import (
    add_large_trade_flags,
    aggregate_large_trade_features,
)
from cryptolab.features.trade_flow import (
    add_cvd_features,
    aggregate_aggtrades,
)
from cryptolab.features.trade_flow_regime import (
    add_trade_flow_regime_features,
)
from cryptolab.features.trade_flow_snapshot import (
    TradeFlowSnapshot,
    build_trade_flow_snapshot,
)
from cryptolab.pipelines.aggtrades_storage import (
    read_aggtrades,
)
from cryptolab.pipelines.aggtrades_update import (
    AggTradesUpdateResult,
    update_aggtrades,
)
from cryptolab.pipelines.trade_flow_snapshot import (
    save_trade_flow_snapshot,
)
from cryptolab.pipelines.trade_flow_storage import (
    save_trade_flow_features,
)
from cryptolab.quality.aggtrades import (
    AggTradesAuditResult,
    audit_aggtrades,
)
class TradeFlowServiceError(RuntimeError):
    """Raised when Trade Flow production pipeline fails."""
@dataclass(frozen=True)
class TradeFlowBuildResult:
    """
    Summary of one complete Trade Flow production run.
    """
    exchange: str
    symbol: str
    timeframe: str
    raw_rows: int
    flow_rows: int
    update_result: AggTradesUpdateResult
    audit_result: AggTradesAuditResult
    snapshot: TradeFlowSnapshot
    structurally_valid: bool
    continuous: bool
    valid: bool
def build_trade_flow_features(
    trades: pd.DataFrame,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """
    Build complete curated Trade Flow features.
    Pipeline:
        aggTrades
            ↓
        aggregation
            ↓
        imbalance
            ↓
        CVD
            ↓
        large-trade classification
            ↓
        Trade Flow regime
    """
    if trades.empty:
        raise TradeFlowServiceError(
            "aggTrades input is empty"
        )
    flow = aggregate_aggtrades(
        trades,
        timeframe=timeframe,
    )
    if flow.empty:
        raise TradeFlowServiceError(
            "Trade Flow aggregation returned no rows"
        )
    flow = add_cvd_features(
        flow
    )
    trade_count = len(trades)
    min_periods = min(
        1_000,
        max(
            10,
            trade_count // 2,
        ),
    )
    rolling_window = max(
        min_periods,
        min(
            10_000,
            trade_count,
        ),
    )
    flagged = add_large_trade_flags(
        trades,
        rolling_window=rolling_window,
        min_periods=min_periods,
        large_quantile=0.95,
        very_large_quantile=0.99,
    )
    large_flow = (
        aggregate_large_trade_features(
            flagged,
            timeframe=timeframe,
        )
    )
    flow = add_trade_flow_regime_features(
        flow,
        large_flow_df=large_flow,
        rolling_window=60,
    )
    return (
        flow
        .sort_values("open_time")
        .reset_index(drop=True)
    )
def run_trade_flow_pipeline(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    update_raw: bool = True,
) -> TradeFlowBuildResult:
    """
    Run complete Trade Flow production pipeline.
    Steps
    -----
    1. Incrementally update raw aggTrades.
    2. Read complete local raw dataset.
    3. Audit raw data.
    4. Build curated Trade Flow features.
    5. Persist daily curated partitions.
    6. Build latest snapshot.
    7. Persist latest snapshot.
    """
    exchange = exchange.lower()
    symbol = symbol.upper()
    if update_raw:
        update_result = update_aggtrades(
            exchange=exchange,
            symbol=symbol,
            page_limit=1000,
            pause_seconds=0.05,
        )
    else:
        from cryptolab.pipelines.aggtrades_storage import (
            get_latest_agg_trade_id,
        )
        latest_id = (
            get_latest_agg_trade_id(
                exchange=exchange,
                symbol=symbol,
            )
        )
        if latest_id is None:
            raise TradeFlowServiceError(
                "Cannot run without raw aggTrades"
            )
        update_result = AggTradesUpdateResult(
            exchange=exchange,
            symbol=symbol,
            start_id=latest_id + 1,
            target_id=latest_id,
            rows_fetched=0,
            pages_fetched=0,
            first_fetched_id=None,
            last_fetched_id=None,
            up_to_date=True,
        )
    trades = read_aggtrades(
        exchange=exchange,
        symbol=symbol,
    )
    if trades.empty:
        raise TradeFlowServiceError(
            "No aggTrades available after update"
        )
    audit_result = audit_aggtrades(
        trades
    )
    if not audit_result.structurally_valid:
        raise TradeFlowServiceError(
            "aggTrades structural audit failed"
        )
    if not audit_result.continuous:
        raise TradeFlowServiceError(
            "aggTrades ID continuity audit failed"
        )
    flow = build_trade_flow_features(
        trades,
        timeframe=timeframe,
    )
    save_trade_flow_features(
        flow,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    snapshot = build_trade_flow_snapshot(
        flow
    )
    save_trade_flow_snapshot(
        snapshot
    )
    valid = (
        audit_result.structurally_valid
        and audit_result.continuous
        and not flow.empty
    )
    return TradeFlowBuildResult(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        raw_rows=len(trades),
        flow_rows=len(flow),
        update_result=update_result,
        audit_result=audit_result,
        snapshot=snapshot,
        structurally_valid=(
            audit_result.structurally_valid
        ),
        continuous=(
            audit_result.continuous
        ),
        valid=valid,
    )
