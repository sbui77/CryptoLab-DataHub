from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.pipelines.open_interest_storage import (
    OPEN_INTEREST_COLUMNS,
    OpenInterestStorageError,
    normalize_open_interest,
)
import cryptolab.sources.derivatives_open_interest as oi_source


def make_legacy_oi() -> pd.DataFrame:
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
            "open_interest_base": [
                80000.0,
                80010.0,
            ],
            "open_interest_quote": [
                9_600_000_000.0,
                9_601_200_000.0,
            ],
        }
    )


def test_legacy_oi_gets_explicit_time_metadata():
    result = normalize_open_interest(
        make_legacy_oi()
    )

    assert list(
        result.columns
    ) == OPEN_INTEREST_COLUMNS

    pd.testing.assert_series_equal(
        result["available_at"],
        result["timestamp"],
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


def test_history_fetch_attaches_runtime_metadata(
    monkeypatch,
):
    payload = [
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "80000.0",
            "sumOpenInterestValue": "9600000000.0",
            "timestamp": 1786406400000,
        },
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "80010.0",
            "sumOpenInterestValue": "9601200000.0",
            "timestamp": 1786406700000,
        },
    ]

    monkeypatch.setattr(
        oi_source,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = (
        oi_source.fetch_open_interest_history(
            symbol="BTCUSDT",
            period="5m",
        )
    )

    assert len(result) == 2

    pd.testing.assert_series_equal(
        result["available_at"],
        result["timestamp"],
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


def test_current_fetch_attaches_runtime_metadata(
    monkeypatch,
):
    payload = {
        "symbol": "BTCUSDT",
        "openInterest": "80000.0",
        "time": 1786406400000,
    }

    monkeypatch.setattr(
        oi_source,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = (
        oi_source.fetch_current_open_interest(
            symbol="BTCUSDT"
        )
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "available_at"
        ]
        == result.iloc[0][
            "timestamp"
        ]
    )

    assert (
        result.iloc[0][
            "available_at_quality"
        ]
        == "derived"
    )

    assert pd.notna(
        result.iloc[0][
            "ingested_at"
        ]
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )


def test_empty_history_has_time_contract_schema(
    monkeypatch,
):
    monkeypatch.setattr(
        oi_source,
        "_request_json",
        lambda *args, **kwargs: [],
    )

    result = (
        oi_source.fetch_open_interest_history(
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
    source = make_legacy_oi()

    source["available_at"] = (
        pd.to_datetime(
            source["timestamp"],
            utc=True,
        )
    )

    with pytest.raises(
        OpenInterestStorageError
    ):
        normalize_open_interest(
            source
        )


def test_runtime_oi_normalization_preserves_metadata():
    source = make_legacy_oi()

    source["available_at"] = (
        pd.to_datetime(
            source["timestamp"],
            utc=True,
        )
    )

    source[
        "available_at_quality"
    ] = "derived"

    ingestion_time = pd.Timestamp(
        "2026-08-11T01:00:00Z"
    )

    source["ingested_at"] = (
        ingestion_time
    )

    source[
        "ingested_at_quality"
    ] = "exact"

    result = normalize_open_interest(
        source
    )

    assert (
        result[
            "ingested_at"
        ]
        .eq(ingestion_time)
        .all()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("exact")
        .all()
    )


def test_runtime_oi_normalization_is_idempotent():
    source = make_legacy_oi()

    first = normalize_open_interest(
        source
    )

    second = normalize_open_interest(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )
