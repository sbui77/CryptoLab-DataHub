from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.pipelines.taker_flow_storage import (
    TAKER_FLOW_COLUMNS,
    TakerFlowStorageError,
    normalize_taker_flow,
)
import cryptolab.sources.derivatives_taker_flow as taker_source


def make_legacy_taker_flow() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
            ],
            "market": [
                "usdm_perpetual",
                "usdm_perpetual",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
            ],
            "period": [
                "5m",
                "5m",
            ],
            "timestamp": [
                "2026-08-11T00:00:00Z",
                "2026-08-11T00:05:00Z",
            ],
            "buy_volume": [
                1000.0,
                1200.0,
            ],
            "sell_volume": [
                900.0,
                1100.0,
            ],
            "buy_sell_ratio": [
                1.111111,
                1.090909,
            ],
        }
    )


def test_legacy_taker_flow_gets_time_metadata():
    result = normalize_taker_flow(
        make_legacy_taker_flow()
    )

    assert list(
        result.columns
    ) == TAKER_FLOW_COLUMNS

    pd.testing.assert_series_equal(
        result[
            "available_at"
        ],
        result[
            "timestamp"
        ],
        check_names=False,
    )

    assert (
        result[
            "available_at_quality"
        ]
        .eq("derived")
        .all()
    )

    assert (
        result[
            "ingested_at"
        ]
        .isna()
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("unknown")
        .all()
    )


def test_runtime_fetch_attaches_time_metadata(
    monkeypatch,
):
    payload = [
        {
            "buySellRatio": "1.10",
            "buyVol": "1100.0",
            "sellVol": "1000.0",
            "timestamp": 1786406400000,
        },
        {
            "buySellRatio": "0.90",
            "buyVol": "900.0",
            "sellVol": "1000.0",
            "timestamp": 1786406700000,
        },
    ]

    monkeypatch.setattr(
        taker_source,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = (
        taker_source.fetch_taker_flow_history(
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert len(result) == 2

    pd.testing.assert_series_equal(
        result[
            "available_at"
        ],
        result[
            "timestamp"
        ],
        check_names=False,
    )

    assert (
        result[
            "available_at_quality"
        ]
        .eq("derived")
        .all()
    )

    assert (
        result[
            "ingested_at"
        ]
        .notna()
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("exact")
        .all()
    )

    assert (
        result[
            "ingested_at"
        ].nunique()
        == 1
    )


def test_empty_runtime_fetch_has_new_schema(
    monkeypatch,
):
    monkeypatch.setattr(
        taker_source,
        "_request_json",
        lambda *args, **kwargs: [],
    )

    result = (
        taker_source.fetch_taker_flow_history(
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert result.empty

    for column in [
        "available_at",
        "available_at_quality",
        "ingested_at",
        "ingested_at_quality",
    ]:
        assert column in result.columns


def test_partial_time_metadata_rejected():
    source = make_legacy_taker_flow()

    source[
        "available_at"
    ] = pd.to_datetime(
        source[
            "timestamp"
        ],
        utc=True,
    )

    with pytest.raises(
        TakerFlowStorageError
    ):
        normalize_taker_flow(
            source
        )


def test_runtime_metadata_survives_normalization():
    source = make_legacy_taker_flow()

    source[
        "available_at"
    ] = pd.to_datetime(
        source[
            "timestamp"
        ],
        utc=True,
    )

    source[
        "available_at_quality"
    ] = "derived"

    ingestion_time = pd.Timestamp(
        "2026-08-11T08:00:00Z"
    )

    source[
        "ingested_at"
    ] = ingestion_time

    source[
        "ingested_at_quality"
    ] = "exact"

    result = normalize_taker_flow(
        source
    )

    assert (
        result[
            "ingested_at"
        ]
        .eq(
            ingestion_time
        )
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("exact")
        .all()
    )


def test_normalization_is_idempotent():
    first = normalize_taker_flow(
        make_legacy_taker_flow()
    )

    second = normalize_taker_flow(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_unknown_ingestion_is_not_fabricated():
    result = normalize_taker_flow(
        make_legacy_taker_flow()
    )

    assert (
        result[
            "ingested_at"
        ]
        .isna()
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("unknown")
        .all()
    )
