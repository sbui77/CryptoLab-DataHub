from __future__ import annotations

from pathlib import Path

import pandas as pd

import cryptolab.pipelines.basis_storage as storage


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
                "binance",
            ],
            "market": [
                "usdm_perpetual",
            ] * 3,
            "symbol": [
                "BTCUSDT",
            ] * 3,
            "contract_type": [
                "PERPETUAL",
            ] * 3,
            "period": [
                "5m",
            ] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10T23:55:00Z",
                    "2026-08-11T00:00:00Z",
                    "2026-08-11T00:05:00Z",
                ],
                utc=True,
            ),
            "index_price": [
                60000.0,
                60100.0,
                60200.0,
            ],
            "futures_price": [
                60006.0,
                60106.01,
                60193.98,
            ],
            "basis": [
                6.0,
                6.01,
                -6.02,
            ],
            "basis_rate": [
                0.0001,
                0.0001,
                -0.0001,
            ],
            "annualized_basis_rate": [
                float("nan"),
                float("nan"),
                float("nan"),
            ],
        }
    )


def test_save_daily_partitions(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    files = storage.save_basis(
        make_df()
    )

    assert len(files) == 2


def test_read_basis(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_basis(
        make_df()
    )

    result = storage.read_basis(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    assert len(result) == 3

    assert (
        result["timestamp"]
        .is_monotonic_increasing
    )


def test_idempotent_save(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_basis(
        make_df()
    )

    storage.save_basis(
        make_df()
    )

    result = storage.read_basis(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    assert len(result) == 3

    assert (
        result["timestamp"]
        .duplicated()
        .sum()
        == 0
    )


def test_latest_timestamp(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_basis(
        make_df()
    )

    latest = (
        storage
        .get_latest_basis_timestamp(
            exchange="binance",
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert latest == pd.Timestamp(
        "2026-08-11T00:05:00Z"
    )
