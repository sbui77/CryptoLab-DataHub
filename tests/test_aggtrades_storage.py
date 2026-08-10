from __future__ import annotations

from pathlib import Path

import pandas as pd

import cryptolab.pipelines.aggtrades_storage as storage


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
                "BTCUSDT",
            ],
            "agg_trade_id": [
                100,
                101,
                102,
            ],
            "price": [
                60000.0,
                60100.0,
                60200.0,
            ],
            "quantity": [
                0.50,
                0.25,
                0.10,
            ],
            "quote_quantity": [
                30000.0,
                15025.0,
                6020.0,
            ],
            "first_trade_id": [
                1000,
                1001,
                1002,
            ],
            "last_trade_id": [
                1000,
                1001,
                1002,
            ],
            "trade_time": pd.to_datetime(
                [
                    "2026-08-10T23:59:58Z",
                    "2026-08-10T23:59:59Z",
                    "2026-08-11T00:00:01Z",
                ],
                utc=True,
            ),
            "buyer_is_maker": [
                False,
                True,
                False,
            ],
            "taker_side": [
                "buy",
                "sell",
                "buy",
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

    files = storage.save_aggtrades(
        make_df()
    )

    assert len(files) == 2

    assert (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=10"
        / "data.parquet"
    ).exists()

    assert (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=11"
        / "data.parquet"
    ).exists()


def test_read_aggtrades(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    original = make_df()

    storage.save_aggtrades(
        original
    )

    result = storage.read_aggtrades(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert len(result) == 3

    assert (
        result["agg_trade_id"]
        .tolist()
        == [
            100,
            101,
            102,
        ]
    )


def test_deduplicate_on_save(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    df = make_df()

    storage.save_aggtrades(
        df
    )

    storage.save_aggtrades(
        df
    )

    result = storage.read_aggtrades(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert len(result) == 3

    assert (
        result[
            "agg_trade_id"
        ]
        .duplicated()
        .sum()
        == 0
    )


def test_latest_trade_id(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    storage.save_aggtrades(
        make_df()
    )

    latest_id = (
        storage.get_latest_agg_trade_id(
            exchange="binance",
            symbol="BTCUSDT",
        )
    )

    assert latest_id == 102


def test_empty_storage(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    result = storage.read_aggtrades(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert result.empty

    assert (
        storage.get_latest_agg_trade_id(
            exchange="binance",
            symbol="BTCUSDT",
        )
        is None
    )
