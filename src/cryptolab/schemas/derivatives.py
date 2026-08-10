from __future__ import annotations

from dataclasses import dataclass


class DerivativesSchemaError(ValueError):
    """Raised when derivatives data violates canonical schema."""


# ============================================================
# DATASET NAMES
# ============================================================

OPEN_INTEREST_DATASET = "open_interest"
FUNDING_RATE_DATASET = "funding_rate"
PREMIUM_BASIS_DATASET = "premium_basis"
TAKER_FLOW_DATASET = "taker_flow"
LIQUIDATIONS_DATASET = "liquidations"


DERIVATIVES_DATASETS = {
    OPEN_INTEREST_DATASET,
    FUNDING_RATE_DATASET,
    PREMIUM_BASIS_DATASET,
    TAKER_FLOW_DATASET,
    LIQUIDATIONS_DATASET,
}


# ============================================================
# OPEN INTEREST
# ============================================================

OPEN_INTEREST_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "open_interest_base",
    "open_interest_quote",
]


# ============================================================
# FUNDING RATE
# ============================================================

FUNDING_RATE_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "funding_time",
    "funding_rate",
    "mark_price",
    "rate_type",
]


# ============================================================
# PREMIUM / BASIS
# ============================================================

PREMIUM_BASIS_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "index_price",
    "futures_price",
    "basis",
    "basis_rate",
    "annualized_basis_rate",
]


# ============================================================
# FUTURES TAKER FLOW
# ============================================================

TAKER_FLOW_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "period",
    "timestamp",
    "buy_volume",
    "sell_volume",
    "buy_sell_ratio",
]


# ============================================================
# LIQUIDATIONS
# ============================================================

LIQUIDATION_COLUMNS = [
    "exchange",
    "market",
    "symbol",
    "event_time",
    "order_time",
    "side",
    "order_type",
    "time_in_force",
    "original_quantity",
    "price",
    "average_price",
    "order_status",
    "last_filled_quantity",
    "filled_quantity",
    "liquidation_notional",
]


# ============================================================
# COMMON METADATA
# ============================================================

SUPPORTED_EXCHANGES = {
    "binance",
}

SUPPORTED_MARKETS = {
    "usdm_perpetual",
}

SUPPORTED_PERIODS = {
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
}


@dataclass(frozen=True)
class DerivativesDatasetContract:
    """
    Metadata describing one canonical derivatives dataset.
    """

    dataset: str

    exchange: str

    market: str

    symbol: str

    timestamp_column: str

    native_period: str | None

    event_based: bool


def validate_contract(
    contract: DerivativesDatasetContract,
) -> None:
    """
    Validate dataset-level metadata.

    This validates the contract itself, not dataframe rows.
    """

    if (
        contract.dataset
        not in DERIVATIVES_DATASETS
    ):
        raise DerivativesSchemaError(
            f"Unsupported derivatives dataset: "
            f"{contract.dataset}"
        )

    if (
        contract.exchange
        not in SUPPORTED_EXCHANGES
    ):
        raise DerivativesSchemaError(
            f"Unsupported exchange: "
            f"{contract.exchange}"
        )

    if (
        contract.market
        not in SUPPORTED_MARKETS
    ):
        raise DerivativesSchemaError(
            f"Unsupported derivatives market: "
            f"{contract.market}"
        )

    if not contract.symbol:
        raise DerivativesSchemaError(
            "symbol cannot be empty"
        )

    if not contract.timestamp_column:
        raise DerivativesSchemaError(
            "timestamp_column cannot be empty"
        )

    if (
        contract.native_period is not None
        and contract.native_period
        not in SUPPORTED_PERIODS
    ):
        raise DerivativesSchemaError(
            f"Unsupported period: "
            f"{contract.native_period}"
        )

    if (
        contract.event_based
        and contract.native_period
        is not None
    ):
        raise DerivativesSchemaError(
            "Event-based dataset cannot have "
            "native_period"
        )


# ============================================================
# CANONICAL CONTRACTS — BTCUSDT
# ============================================================


def btcusdt_open_interest_contract(
) -> DerivativesDatasetContract:
    return DerivativesDatasetContract(
        dataset=OPEN_INTEREST_DATASET,
        exchange="binance",
        market="usdm_perpetual",
        symbol="BTCUSDT",
        timestamp_column="timestamp",
        native_period="5m",
        event_based=False,
    )


def btcusdt_funding_rate_contract(
) -> DerivativesDatasetContract:
    return DerivativesDatasetContract(
        dataset=FUNDING_RATE_DATASET,
        exchange="binance",
        market="usdm_perpetual",
        symbol="BTCUSDT",
        timestamp_column="funding_time",
        native_period=None,
        event_based=True,
    )


def btcusdt_premium_basis_contract(
) -> DerivativesDatasetContract:
    return DerivativesDatasetContract(
        dataset=PREMIUM_BASIS_DATASET,
        exchange="binance",
        market="usdm_perpetual",
        symbol="BTCUSDT",
        timestamp_column="timestamp",
        native_period="5m",
        event_based=False,
    )


def btcusdt_taker_flow_contract(
) -> DerivativesDatasetContract:
    return DerivativesDatasetContract(
        dataset=TAKER_FLOW_DATASET,
        exchange="binance",
        market="usdm_perpetual",
        symbol="BTCUSDT",
        timestamp_column="timestamp",
        native_period="5m",
        event_based=False,
    )


def btcusdt_liquidation_contract(
) -> DerivativesDatasetContract:
    return DerivativesDatasetContract(
        dataset=LIQUIDATIONS_DATASET,
        exchange="binance",
        market="usdm_perpetual",
        symbol="BTCUSDT",
        timestamp_column="event_time",
        native_period=None,
        event_based=True,
    )
