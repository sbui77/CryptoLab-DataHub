from pathlib import Path

import pandas as pd

import cryptolab.pipelines.liquidation_storage as storage


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
            ],

            "market": [
                "usdm_perpetual",
            ],

            "symbol": [
                "BTCUSDT",
            ],

            "event_time": pd.to_datetime(
                [
                    "2026-08-10T10:00:00Z",
                ],
                utc=True,
            ),

            "order_time": pd.to_datetime(
                [
                    "2026-08-10T10:00:00Z",
                ],
                utc=True,
            ),

            "side": [
                "SELL",
            ],

            "liquidation_side": [
                "long",
            ],

            "order_type": [
                "LIMIT",
            ],

            "time_in_force": [
                "IOC",
            ],

            "original_quantity": [
                1.0,
            ],

            "price": [
                65000.0,
            ],

            "average_price": [
                65000.0,
            ],

            "order_status": [
                "FILLED",
            ],

            "last_filled_quantity": [
                1.0,
            ],

            "filled_quantity": [
                1.0,
            ],

            "liquidation_notional": [
                65000.0,
            ],
        }
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

    storage.save_liquidations(
        make_df()
    )

    storage.save_liquidations(
        make_df()
    )

    result = storage.read_liquidations(
        exchange="binance",
        symbol="BTCUSDT",
    )

    assert len(result) == 1

    assert (
        result.duplicated().sum()
        == 0
    )
