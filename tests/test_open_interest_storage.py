from __future__ import annotations

from pathlib import Path

import pandas as pd

import cryptolab.pipelines.open_interest_storage as storage


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
            "open_interest_base": [
                1000.0,
                1001.0,
                1002.0,
            ],
            "open_interest_quote": [
                60000000.0,
                60100000.0,
                60200000.0,
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

    files = storage.save_open_interest(
        make_df()
    )

    assert len(files) == 2


def test_read_open_interest(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_open_interest(
        make_df()
    )

    result = storage.read_open_interest(
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

    storage.save_open_interest(
        make_df()
    )

    storage.save_open_interest(
        make_df()
    )

    result = storage.read_open_interest(
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

    storage.save_open_interest(
        make_df()
    )

    latest = (
        storage
        .get_latest_open_interest_timestamp(
            exchange="binance",
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert latest == pd.Timestamp(
        "2026-08-11T00:05:00Z"
    )
