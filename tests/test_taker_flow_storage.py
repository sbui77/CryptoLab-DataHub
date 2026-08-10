from pathlib import Path

import pandas as pd

import cryptolab.pipelines.taker_flow_storage as storage


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

            "buy_volume": [
                100.0,
                150.0,
                200.0,
            ],

            "sell_volume": [
                80.0,
                150.0,
                100.0,
            ],

            "buy_sell_ratio": [
                1.25,
                1.0,
                2.0,
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

    files = storage.save_taker_flow(
        make_df()
    )

    assert len(files) == 2


def test_idempotent_save(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_taker_flow(
        make_df()
    )

    storage.save_taker_flow(
        make_df()
    )

    result = storage.read_taker_flow(
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

    storage.save_taker_flow(
        make_df()
    )

    latest = (
        storage
        .get_latest_taker_flow_timestamp(
            exchange="binance",
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert latest == pd.Timestamp(
        "2026-08-11T00:05:00Z"
    )
