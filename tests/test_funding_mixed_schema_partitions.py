from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.funding_rate_storage as storage


def _legacy_row(
    funding_time: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": ["binance"],
            "market": ["usdm_perpetual"],
            "symbol": ["BTCUSDT"],
            "funding_time": [
                pd.Timestamp(
                    funding_time
                )
            ],
            "funding_rate": [0.0001],
            "mark_price": [120000.0],
            "rate_type": ["Regular"],
        }
    )


def _runtime_row(
    funding_time: str,
) -> pd.DataFrame:
    event_time = pd.Timestamp(
        funding_time
    )

    return pd.DataFrame(
        {
            "exchange": ["binance"],
            "market": ["usdm_perpetual"],
            "symbol": ["BTCUSDT"],
            "funding_time": [event_time],
            "funding_rate": [0.0002],
            "mark_price": [121000.0],
            "rate_type": ["Regular"],
            "available_at": [
                event_time
            ],
            "available_at_quality": [
                "derived"
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T12:00:00Z"
                )
            ],
            "ingested_at_quality": [
                "exact"
            ],
        }
    )


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
    # Build a genuinely legacy month directly on disk.
    # --------------------------------------------------------

    legacy = _legacy_row(
        "2026-07-31T16:00:00Z"
    )

    legacy_path = (
        tmp_path
        / "derivatives"
        / "funding_rate"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=07"
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
    # Build new PIT-aware August partition.
    # --------------------------------------------------------

    runtime = _runtime_row(
        "2026-08-11T08:00:00Z"
    )

    runtime_path = (
        tmp_path
        / "derivatives"
        / "funding_rate"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
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
    # Read across both schema vintages.
    # --------------------------------------------------------

    result = storage.read_funding_rate(
        "binance",
        "BTCUSDT",
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
            "funding_time"
        ]
        == pd.Timestamp(
            "2026-07-31T16:00:00Z"
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
            "funding_time"
        ]
        == pd.Timestamp(
            "2026-08-11T08:00:00Z"
        )
    ].iloc[0]

    assert (
        runtime_result[
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert pd.notna(
        runtime_result[
            "ingested_at"
        ]
    )
