from __future__ import annotations
from pathlib import Path
import pandas as pd
from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
class TradeFlowStorageError(RuntimeError):
    """Raised when curated Trade Flow storage fails."""
def _get_curated_root() -> Path:
    """
    Resolve curated data root from CryptoLab config.
    """
    config = load_config()
    return resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )
def _trade_flow_root(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Return canonical curated Trade Flow storage root.
    """
    return (
        _get_curated_root()
        / "trade_flow"
        / "features"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"timeframe={timeframe}"
    )
def save_trade_flow_features(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    """
    Save curated Trade Flow bars into daily Parquet partitions.
    Trade Flow is substantially denser than Price Layer,
    therefore daily partitioning is used.
    Existing daily partitions are overwritten using the
    supplied dataframe subset for that day.
    """
    if df.empty:
        return []
    required_columns = [
        "open_time",
    ]
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing:
        raise TradeFlowStorageError(
            f"Missing columns: {missing}"
        )
    work = (
        df.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    work["year"] = (
        work["open_time"]
        .dt.year
    )
    work["month"] = (
        work["open_time"]
        .dt.month
    )
    work["day"] = (
        work["open_time"]
        .dt.day
    )
    root = _trade_flow_root(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    saved_files: list[Path] = []
    grouped = work.groupby(
        [
            "year",
            "month",
            "day",
        ],
        sort=True,
    )
    for (
        year,
        month,
        day,
    ), partition in grouped:
        directory = (
            root
            / f"year={int(year)}"
            / f"month={int(month):02d}"
            / f"day={int(day):02d}"
        )
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        output = (
            directory
            / "data.parquet"
        )
        partition = (
            partition
            .drop(
                columns=[
                    "year",
                    "month",
                    "day",
                ]
            )
            .sort_values("open_time")
            .drop_duplicates(
                subset=["open_time"],
                keep="last",
            )
            .reset_index(drop=True)
        )
        partition.to_parquet(
            output,
            index=False,
            compression="snappy",
        )
        saved_files.append(
            output
        )
    return saved_files
def list_trade_flow_files(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    """
    List all curated Trade Flow partitions.
    """
    root = _trade_flow_root(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not root.exists():
        return []
    return sorted(
        root.glob(
            "year=*/month=*/day=*/data.parquet"
        )
    )
def read_trade_flow_features(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Read all curated Trade Flow bars.
    """
    files = list_trade_flow_files(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not files:
        return pd.DataFrame()
    frames = [
        pd.read_parquet(file)
        for file in files
    ]
    result = pd.concat(
        frames,
        ignore_index=True,
    )
    return (
        result
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )
