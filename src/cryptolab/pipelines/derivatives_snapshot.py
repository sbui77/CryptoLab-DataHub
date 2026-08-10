from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.derivatives_snapshot import (
    DerivativesSnapshot,
    build_derivatives_snapshot,
    derivatives_snapshot_to_dataframe,
)
from cryptolab.pipelines.derivatives_quality import (
    build_derivatives_quality_dataset,
)


class DerivativesSnapshotPipelineError(
    RuntimeError
):
    """Raised when derivatives snapshot pipeline fails."""


def build_derivatives_snapshot_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
) -> tuple[
    DerivativesSnapshot,
    pd.DataFrame,
]:
    """
    Build current derivatives snapshot from quality dataset.
    """

    quality = build_derivatives_quality_dataset(
        exchange=exchange,
        symbol=symbol,
        price_timeframe=price_timeframe,
        derivatives_period=derivatives_period,
    )

    snapshot = build_derivatives_snapshot(
        quality,
        exchange=exchange,
        symbol=symbol,
        timeframe=price_timeframe,
    )

    frame = derivatives_snapshot_to_dataframe(
        snapshot
    )

    return (
        snapshot,
        frame,
    )


def _get_curated_root() -> Path:
    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )


def save_derivatives_snapshot(
    snapshot: DerivativesSnapshot,
) -> Path:
    """
    Persist latest derivatives snapshot.

    Snapshot is intentionally a single overwrite file.
    Historical state remains in the quality/regime datasets.
    """

    directory = (
        _get_curated_root()
        / "derivatives"
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

    frame = derivatives_snapshot_to_dataframe(
        snapshot
    )

    frame.to_parquet(
        output,
        index=False,
        compression="snappy",
    )

    return output


def build_and_save_derivatives_snapshot(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    price_timeframe: str = "1h",
    derivatives_period: str = "5m",
) -> tuple[
    DerivativesSnapshot,
    Path,
]:
    snapshot, _ = (
        build_derivatives_snapshot_dataset(
            exchange=exchange,
            symbol=symbol,
            price_timeframe=price_timeframe,
            derivatives_period=derivatives_period,
        )
    )

    output = save_derivatives_snapshot(
        snapshot
    )

    return (
        snapshot,
        output,
    )
