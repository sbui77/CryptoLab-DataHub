from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.basis_storage as storage


def test_read_mixed_schema_partitions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    # --------------------------------------------------------
    # Legacy partition
    # --------------------------------------------------------

    legacy = pd.DataFrame(
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
            "contract_type": [
                "PERPETUAL",
            ],
            "period": [
                "5m",
            ],
            "timestamp": [
                pd.Timestamp(
                    "2026-08-10T23:55:00Z"
                )
            ],
            "index_price": [
                120000.0,
            ],
            "futures_price": [
                120010.0,
            ],
            "basis": [
                10.0,
            ],
            "basis_rate": [
                0.000083,
            ],
            "annualized_basis_rate": [
                0.0075,
            ],
        }
    )

    legacy_path = (
        tmp_path
        / "derivatives"
        / "premium_basis"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "period=5m"
        / "year=2026"
        / "month=08"
        / "day=10"
        / "data.parquet"
    )

    legacy_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy.to_parquet(
        legacy_path,
        index=False,
    )

    # --------------------------------------------------------
    # PIT-aware partition
    # --------------------------------------------------------

    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    runtime = pd.DataFrame(
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
            "contract_type": [
                "PERPETUAL",
            ],
            "period": [
                "5m",
            ],
            "timestamp": [
                event_time,
            ],
            "index_price": [
                120001.0,
            ],
            "futures_price": [
                120012.0,
            ],
            "basis": [
                11.0,
            ],
            "basis_rate": [
                0.000091,
            ],
            "annualized_basis_rate": [
                0.0080,
            ],
            "available_at": [
                event_time,
            ],
            "available_at_quality": [
                "derived",
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T12:00:00Z"
                ),
            ],
            "ingested_at_quality": [
                "exact",
            ],
        }
    )

    runtime_path = (
        tmp_path
        / "derivatives"
        / "premium_basis"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "period=5m"
        / "year=2026"
        / "month=08"
        / "day=11"
        / "data.parquet"
    )

    runtime_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime.to_parquet(
        runtime_path,
        index=False,
    )

    # --------------------------------------------------------
    # Read across schema vintages
    # --------------------------------------------------------

    result = storage.read_basis(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(result) == 2

    assert (
        result[
            "available_at_quality"
        ]
        .eq("derived")
        .all()
    )

    legacy_result = result[
        result[
            "timestamp"
        ]
        == pd.Timestamp(
            "2026-08-10T23:55:00Z"
        )
    ].iloc[0]

    assert (
        legacy_result[
            "ingested_at_quality"
        ]
        == "unknown"
    )

    assert pd.isna(
        legacy_result[
            "ingested_at"
        ]
    )

    runtime_result = result[
        result[
            "timestamp"
        ]
        == pd.Timestamp(
            "2026-08-11T00:00:00Z"
        )
    ].iloc[0]

    assert (
        runtime_result[
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        runtime_result[
            "ingested_at"
        ]
        == pd.Timestamp(
            "2026-08-11T12:00:00Z"
        )
    )
