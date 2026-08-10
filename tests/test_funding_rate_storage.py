from __future__ import annotations

from pathlib import Path

import pandas as pd

import cryptolab.pipelines.funding_rate_storage as storage


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

            "funding_time": (
                pd.to_datetime(
                    [
                        "2026-07-31T16:00:00Z",
                        "2026-08-01T00:00:00Z",
                        "2026-08-01T08:00:00Z",
                    ],
                    utc=True,
                )
            ),

            "funding_rate": [
                0.0001,
                0.0002,
                -0.0001,
            ],

            "mark_price": [
                60000.0,
                60100.0,
                60200.0,
            ],

            "rate_type": [
                "Regular",
                "Regular",
                "Regular",
            ],
        }
    )


def test_save_monthly_partitions(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    files = storage.save_funding_rate(
        make_df()
    )

    assert len(files) == 2


def test_read_funding_rate(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_funding_rate(
        make_df()
    )

    result = storage.read_funding_rate(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert len(result) == 3

    assert (
        result["funding_time"]
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

    storage.save_funding_rate(
        make_df()
    )

    storage.save_funding_rate(
        make_df()
    )

    result = storage.read_funding_rate(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert len(result) == 3

    assert (
        result[
            "funding_time"
        ]
        .duplicated()
        .sum()
        == 0
    )


def test_latest_funding_time(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_funding_rate(
        make_df()
    )

    latest = (
        storage
        .get_latest_funding_time(
            exchange="binance",
            symbol="BTCUSDT",
        )
    )

    assert latest == pd.Timestamp(
        "2026-08-01T08:00:00Z"
    )
