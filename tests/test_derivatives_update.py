from __future__ import annotations

from dataclasses import dataclass

import cryptolab.pipelines.derivatives_update as module


@dataclass(frozen=True)
class FakeOIResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    start_time: str = "2026-08-10T10:00:00Z"
    end_time: str = "2026-08-10T11:00:00Z"
    rows_fetched: int = 12
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    bootstrap: bool = False
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeFundingResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    bootstrap: bool = False
    rows_fetched: int = 1
    pages_fetched: int = 1
    first_funding_time: str = "2026-08-10T08:00:00Z"
    last_funding_time: str = "2026-08-10T08:00:00Z"
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeBasisResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    contract_type: str = "PERPETUAL"
    bootstrap: bool = False
    rows_fetched: int = 12
    pages_fetched: int = 1
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    target_end_time: str = "2026-08-10T10:55:00Z"
    up_to_date: bool = True


@dataclass(frozen=True)
class FakeTakerResult:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    bootstrap: bool = False
    rows_fetched: int = 12
    pages_fetched: int = 1
    first_timestamp: str = "2026-08-10T10:00:00Z"
    last_timestamp: str = "2026-08-10T10:55:00Z"
    target_end_time: str = "2026-08-10T10:55:00Z"
    up_to_date: bool = True


def install_success_mocks(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "update_open_interest",
        lambda **kwargs: FakeOIResult(),
    )

    monkeypatch.setattr(
        module,
        "update_funding_rate",
        lambda **kwargs: FakeFundingResult(),
    )

    monkeypatch.setattr(
        module,
        "update_basis",
        lambda **kwargs: FakeBasisResult(),
    )

    monkeypatch.setattr(
        module,
        "update_taker_flow",
        lambda **kwargs: FakeTakerResult(),
    )


def test_all_streams_success(
    monkeypatch,
):
    install_success_mocks(
        monkeypatch
    )

    result = module.update_derivatives()

    assert result.success is True
    assert result.failure_count == 0
    assert result.failures == ()

    assert (
        result.open_interest.success
        is True
    )

    assert (
        result.funding_rate.success
        is True
    )

    assert (
        result.basis.success
        is True
    )

    assert (
        result.taker_flow.success
        is True
    )


def test_normalizes_funding_timestamp(
    monkeypatch,
):
    install_success_mocks(
        monkeypatch
    )

    result = module.update_derivatives()

    assert (
        result.funding_rate.first_timestamp
        == "2026-08-10T08:00:00Z"
    )

    assert (
        result.funding_rate.last_timestamp
        == "2026-08-10T08:00:00Z"
    )


def test_rows_fetched_normalized(
    monkeypatch,
):
    install_success_mocks(
        monkeypatch
    )

    result = module.update_derivatives()

    assert (
        result.open_interest.rows_fetched
        == 12
    )

    assert (
        result.funding_rate.rows_fetched
        == 1
    )


def test_continue_after_one_failure(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        module,
        "update_open_interest",
        lambda **kwargs: FakeOIResult(),
    )

    def funding_failure(
        **kwargs,
    ):
        calls.append(
            "funding"
        )

        raise RuntimeError(
            "funding failed"
        )

    monkeypatch.setattr(
        module,
        "update_funding_rate",
        funding_failure,
    )

    def basis_success(
        **kwargs,
    ):
        calls.append(
            "basis"
        )

        return FakeBasisResult()

    monkeypatch.setattr(
        module,
        "update_basis",
        basis_success,
    )

    def taker_success(
        **kwargs,
    ):
        calls.append(
            "taker"
        )

        return FakeTakerResult()

    monkeypatch.setattr(
        module,
        "update_taker_flow",
        taker_success,
    )

    result = module.update_derivatives(
        continue_on_error=True,
    )

    assert result.success is False

    assert result.failure_count == 1

    assert result.failures == (
        "funding_rate",
    )

    assert (
        result.funding_rate.success
        is False
    )

    assert (
        result.basis.success
        is True
    )

    assert (
        result.taker_flow.success
        is True
    )

    assert calls == [
        "funding",
        "basis",
        "taker",
    ]


def test_stop_after_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "update_open_interest",
        lambda **kwargs: FakeOIResult(),
    )

    monkeypatch.setattr(
        module,
        "update_funding_rate",
        lambda **kwargs: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "funding failed"
                )
            )
        ),
    )

    monkeypatch.setattr(
        module,
        "update_basis",
        lambda **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "basis should not run"
                )
            )
        ),
    )

    monkeypatch.setattr(
        module,
        "update_taker_flow",
        lambda **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "taker should not run"
                )
            )
        ),
    )

    result = module.update_derivatives(
        continue_on_error=False,
    )

    assert result.success is False

    assert result.failure_count == 3

    assert (
        result.funding_rate.error_type
        == "RuntimeError"
    )

    assert (
        result.basis.error_type
        == "Skipped"
    )

    assert (
        result.taker_flow.error_type
        == "Skipped"
    )


def test_error_message_preserved(
    monkeypatch,
):
    install_success_mocks(
        monkeypatch
    )

    def failure(
        **kwargs,
    ):
        raise ValueError(
            "test failure"
        )

    monkeypatch.setattr(
        module,
        "update_basis",
        failure,
    )

    result = module.update_derivatives()

    assert (
        result.basis.error_type
        == "ValueError"
    )

    assert (
        result.basis.error_message
        == "test failure"
    )


def test_exchange_normalized(
    monkeypatch,
):
    install_success_mocks(
        monkeypatch
    )

    result = module.update_derivatives(
        exchange="BINANCE",
        symbol="btcusdt",
    )

    assert (
        result.exchange
        == "binance"
    )

    assert (
        result.symbol
        == "BTCUSDT"
    )


def test_unsupported_exchange():
    try:
        module.update_derivatives(
            exchange="other"
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
