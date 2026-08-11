from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.pipelines.basis_storage import (
    BASIS_COLUMNS,
    BasisStorageError,
    normalize_basis,
)
import cryptolab.sources.derivatives_basis as basis_source


def make_legacy_basis() -> pd.DataFrame:
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
            "contract_type": [
                "PERPETUAL",
                "PERPETUAL",
            ],
            "period": [
                "5m",
                "5m",
            ],
            "timestamp": [
                "2026-08-11T00:00:00Z",
                "2026-08-11T00:05:00Z",
            ],
            "index_price": [
                120000.0,
                120100.0,
            ],
            "futures_price": [
                120010.0,
                120110.0,
            ],
            "basis": [
                10.0,
                10.0,
            ],
            "basis_rate": [
                0.000083,
                0.000083,
            ],
            "annualized_basis_rate": [
                0.0075,
                0.0075,
            ],
        }
    )


def test_legacy_basis_gets_explicit_time_metadata():
    result = normalize_basis(
        make_legacy_basis()
    )

    assert list(
        result.columns
    ) == BASIS_COLUMNS

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
            "pair": "BTCUSDT",
            "contractType": "PERPETUAL",
            "indexPrice": "120000.0",
            "futuresPrice": "120010.0",
            "basis": "10.0",
            "basisRate": "0.000083",
            "annualizedBasisRate": "0.0075",
            "timestamp": 1786406400000,
        },
        {
            "pair": "BTCUSDT",
            "contractType": "PERPETUAL",
            "indexPrice": "120100.0",
            "futuresPrice": "120110.0",
            "basis": "10.0",
            "basisRate": "0.000083",
            "annualizedBasisRate": "0.0075",
            "timestamp": 1786406700000,
        },
    ]

    monkeypatch.setattr(
        basis_source,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = (
        basis_source.fetch_basis_history(
            pair="BTCUSDT",
            contract_type="PERPETUAL",
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
        basis_source,
        "_request_json",
        lambda *args, **kwargs: [],
    )

    result = (
        basis_source.fetch_basis_history(
            pair="BTCUSDT",
            contract_type="PERPETUAL",
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
    source = make_legacy_basis()

    source[
        "available_at"
    ] = pd.to_datetime(
        source[
            "timestamp"
        ],
        utc=True,
    )

    with pytest.raises(
        BasisStorageError
    ):
        normalize_basis(
            source
        )


def test_runtime_metadata_survives_normalization():
    source = make_legacy_basis()

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
        "2026-08-11T01:00:00Z"
    )

    source[
        "ingested_at"
    ] = ingestion_time

    source[
        "ingested_at_quality"
    ] = "exact"

    result = normalize_basis(
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
    first = normalize_basis(
        make_legacy_basis()
    )

    second = normalize_basis(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_unknown_ingestion_is_not_fabricated():
    result = normalize_basis(
        make_legacy_basis()
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
