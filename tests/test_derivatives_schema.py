import pytest

from cryptolab.schemas.derivatives import (
    DERIVATIVES_DATASETS,
    DerivativesDatasetContract,
    DerivativesSchemaError,
    btcusdt_funding_rate_contract,
    btcusdt_liquidation_contract,
    btcusdt_open_interest_contract,
    btcusdt_premium_basis_contract,
    btcusdt_taker_flow_contract,
    validate_contract,
)


def test_all_btcusdt_contracts_valid():
    contracts = [
        btcusdt_open_interest_contract(),
        btcusdt_funding_rate_contract(),
        btcusdt_premium_basis_contract(),
        btcusdt_taker_flow_contract(),
        btcusdt_liquidation_contract(),
    ]

    for contract in contracts:
        validate_contract(
            contract
        )


def test_all_datasets_present():
    expected = {
        "open_interest",
        "funding_rate",
        "premium_basis",
        "taker_flow",
        "liquidations",
    }

    assert (
        DERIVATIVES_DATASETS
        == expected
    )


def test_open_interest_contract():
    contract = (
        btcusdt_open_interest_contract()
    )

    assert (
        contract.native_period
        == "5m"
    )

    assert (
        contract.event_based
        is False
    )


def test_funding_is_event_based():
    contract = (
        btcusdt_funding_rate_contract()
    )

    assert (
        contract.native_period
        is None
    )

    assert (
        contract.event_based
        is True
    )


def test_liquidation_is_event_based():
    contract = (
        btcusdt_liquidation_contract()
    )

    assert (
        contract.event_based
        is True
    )


def test_invalid_dataset_rejected():
    contract = (
        DerivativesDatasetContract(
            dataset="invalid",
            exchange="binance",
            market="usdm_perpetual",
            symbol="BTCUSDT",
            timestamp_column="timestamp",
            native_period="5m",
            event_based=False,
        )
    )

    with pytest.raises(
        DerivativesSchemaError
    ):
        validate_contract(
            contract
        )


def test_event_dataset_cannot_have_period():
    contract = (
        DerivativesDatasetContract(
            dataset="funding_rate",
            exchange="binance",
            market="usdm_perpetual",
            symbol="BTCUSDT",
            timestamp_column="funding_time",
            native_period="5m",
            event_based=True,
        )
    )

    with pytest.raises(
        DerivativesSchemaError
    ):
        validate_contract(
            contract
        )
