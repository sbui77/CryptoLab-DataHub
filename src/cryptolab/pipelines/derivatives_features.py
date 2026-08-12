from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)
from cryptolab.features.availability import (
    AVAILABLE_AT,
)
from cryptolab.features.derivatives import (
    build_derivatives_features,
)
from cryptolab.pipelines.basis_storage import (
    read_basis,
)
from cryptolab.pipelines.funding_rate_storage import (
    read_funding_rate,
)
from cryptolab.pipelines.liquidation_storage import (
    read_liquidations,
)
from cryptolab.pipelines.open_interest_storage import (
    read_open_interest,
)
from cryptolab.pipelines.taker_flow_storage import (
    read_taker_flow,
)


class DerivativesFeaturePipelineError(
    RuntimeError
):
    """Raised when derivatives feature pipeline fails."""


def build_derivatives_feature_dataset(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    period: str = "5m",
    require_availability: bool = True,
) -> pd.DataFrame:
    """
    Build complete derivatives feature dataset
    from locally stored raw datasets.

    require_availability
        Raw storage always carries the point-in-time contract,
        so production builds require it.

        Consumers that join this dataset with a layer which has
        not yet been migrated must pass False. Emitting a
        derivatives-only available_at into a combined artifact
        would understate that artifact's true availability.
    """

    open_interest = read_open_interest(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    if open_interest.empty:
        raise DerivativesFeaturePipelineError(
            "Open Interest dataset is empty"
        )

    funding = read_funding_rate(
        exchange=exchange,
        symbol=symbol,
    )

    basis = read_basis(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    taker_flow = read_taker_flow(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    liquidations = read_liquidations(
        exchange=exchange,
        symbol=symbol,
    )

    return build_derivatives_features(
        open_interest=open_interest,
        funding_rate=funding,
        basis=basis,
        taker_flow=taker_flow,
        liquidations=liquidations,
        require_availability=(
            require_availability
        ),
    )


def _get_curated_root() -> Path:
    """
    Resolve curated root.
    """

    config = load_config()

    return resolve_project_path(
        get_config_value(
            config,
            "paths.curated",
            "data/curated",
        )
    )


def save_derivatives_features(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    period: str,
) -> Path:
    """
    Save current complete derivatives feature dataset.

    At this stage one consolidated parquet is sufficient.
    Production partitioning is handled later in Step 8.12.
    """

    if df.empty:
        raise DerivativesFeaturePipelineError(
            "Cannot save empty derivatives dataset"
        )

    if AVAILABLE_AT not in df.columns:
        raise DerivativesFeaturePipelineError(
            "Derivatives feature artifact is missing "
            "available_at; the point-in-time contract "
            "must be persisted with the data"
        )

    directory = (
        _get_curated_root()
        / "derivatives"
        / "features"
        / f"exchange={exchange.lower()}"
        / f"symbol={symbol.upper()}"
        / f"period={period}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        directory
        / "features.parquet"
    )

    (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .to_parquet(
            output,
            index=False,
            compression="snappy",
        )
    )

    return output


def build_and_save_derivatives_features(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    period: str = "5m",
) -> tuple[
    pd.DataFrame,
    Path,
]:
    """
    Build and persist derivatives features.
    """

    df = build_derivatives_feature_dataset(
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    output = save_derivatives_features(
        df,
        exchange=exchange,
        symbol=symbol,
        period=period,
    )

    return (
        df,
        output,
    )
