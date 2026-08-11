from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.pipelines.funding_rate_storage import (
    FUNDING_RATE_COLUMNS,
    FundingRateStorageError,
    normalize_funding_rate,
)
import cryptolab.sources.derivatives_funding as funding_source


def make_legacy_funding() -> pd.DataFrame:
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
            "funding_time": [
                "2026-08-11T00:00:00Z",
                "2026-08-11T08:00:00Z",
            ],
            "funding_rate": [
                0.0001,
                0.0002,
            ],
            "mark_price": [
                120000.0,
                121000.0,
            ],
            "rate_type": [
                "Regular",
                "Regular",
            ],
        }
    )


def test_legacy_funding_gets_explicit_time_metadata():
    result = normalize_funding_rate(
        make_legacy_funding()
    )

    assert list(
        result.columns
    ) == FUNDING_RATE_COLUMNS

    pd.testing.assert_series_equal(
        result["available_at"],
        result["funding_time"],
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
            "symbol": "BTCUSDT",
            "fundingRate": "0.0001",
            "fundingTime": 1786406400000,
            "markPrice": "120000.0",
        },
        {
            "symbol": "BTCUSDT",
            "fundingRate": "0.0002",
            "fundingTime": 1786435200000,
            "markPrice": "121000.0",
        },
    ]

    monkeypatch.setattr(
        funding_source,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = (
        funding_source.fetch_funding_rate_history(
            symbol="BTCUSDT"
        )
    )

    assert len(result) == 2

    pd.testing.assert_series_equal(
        result["available_at"],
        result["funding_time"],
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
        funding_source,
        "_request_json",
        lambda *args, **kwargs: [],
    )

    result = (
        funding_source.fetch_funding_rate_history(
            symbol="BTCUSDT"
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
    source = make_legacy_funding()

    source["available_at"] = pd.to_datetime(
        source["funding_time"],
        utc=True,
    )

    with pytest.raises(
        FundingRateStorageError
    ):
        normalize_funding_rate(
            source
        )


def test_runtime_metadata_survives_normalization():
    source = make_legacy_funding()

    source["available_at"] = pd.to_datetime(
        source["funding_time"],
        utc=True,
    )

    source[
        "available_at_quality"
    ] = "derived"

    ingestion_time = pd.Timestamp(
        "2026-08-11T09:00:00Z"
    )

    source[
        "ingested_at"
    ] = ingestion_time

    source[
        "ingested_at_quality"
    ] = "exact"

    result = normalize_funding_rate(
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


def test_normalization_is_idempotent():
    first = normalize_funding_rate(
        make_legacy_funding()
    )

    second = normalize_funding_rate(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_unknown_ingestion_is_not_fabricated():
    result = normalize_funding_rate(
        make_legacy_funding()
    )

    assert (
        result["ingested_at"]
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
