from __future__ import annotations
from pathlib import Path
import pandas as pd
from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
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
    trade_flow_snapshot_to_dataframe,
)
from cryptolab.pipelines.aggtrades_storage import (
    read_aggtrades,
)
def build_trade_flow_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> pd.DataFrame:
    """
    Build current Trade Flow feature dataset from raw aggTrades.
    """
    trades = read_aggtrades(
        exchange=exchange,
        symbol=symbol,
    )
    if trades.empty:
        raise RuntimeError(
            "No aggTrades data found"
        )
    flow = aggregate_aggtrades(
        trades,
        timeframe=timeframe,
    )
    flow = add_cvd_features(
        flow
    )
    min_periods = min(
        1_000,
        max(
            10,
            len(trades) // 2,
        ),
    )
    flagged = add_large_trade_flags(
        trades,
        rolling_window=10_000,
        min_periods=min_periods,
        large_quantile=0.95,
        very_large_quantile=0.99,
    )
    large_flow = aggregate_large_trade_features(
        flagged,
        timeframe=timeframe,
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
def save_trade_flow_snapshot(
    snapshot: TradeFlowSnapshot,
) -> Path:
    """
    Save latest Trade Flow snapshot to curated storage.
    """
    config = load_config()
    curated_root = resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )
    directory = (
        curated_root
        / "trade_flow"
        / "snapshot"
        / f"exchange={snapshot.exchange}"
        / f"symbol={snapshot.symbol}"
        / f"timeframe={snapshot.timeframe}"
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = (
        directory
        / "latest.parquet"
    )
    df = trade_flow_snapshot_to_dataframe(
        snapshot
    )
    df.to_parquet(
        output,
        index=False,
        compression="snappy",
    )
    return output
def build_and_save_trade_flow_snapshot(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> TradeFlowSnapshot:
    """
    Build Trade Flow dataset, create latest snapshot,
    and persist it.
    """
    flow = build_trade_flow_dataset(
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
    return snapshot
