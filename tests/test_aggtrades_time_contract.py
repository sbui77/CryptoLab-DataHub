from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.schemas.aggtrades import (
    AGGTRADE_COLUMNS,
    AggTradeSchemaError,
    normalize_aggtrades,
)
from cryptolab.sources.aggtrades import (
    _parse_aggtrades,
)


def make_legacy_aggtrades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
            ],
            "agg_trade_id": [
                100,
                101,
            ],
            "price": [
                100000.0,
                100001.0,
            ],
            "quantity": [
                0.10,
                0.20,
            ],
            "first_trade_id": [
                200,
                201,
            ],
            "last_trade_id": [
                200,
                201,
            ],
            "trade_time": [
                "2026-08-11T00:00:01Z",
                "2026-08-11T00:00:02Z",
            ],
            "buyer_is_maker": [
                False,
                True,
            ],
        }
    )


def test_legacy_aggtrades_get_explicit_metadata():
    result = normalize_aggtrades(
        make_legacy_aggtrades()
    )

    assert list(
        result.columns
    ) == AGGTRADE_COLUMNS

    pd.testing.assert_series_equal(
        result["available_at"],
        result["trade_time"],
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


def test_runtime_parser_preserves_time_metadata():
    payload = [
        {
            "a": 123,
            "p": "100000.00",
            "q": "0.125",
            "f": 500,
            "l": 500,
            "T": 1786406400000,
            "m": False,
        }
    ]

    result = _parse_aggtrades(
        payload,
        symbol="BTCUSDT",
    )

    assert len(result) == 1

    assert list(
        result.columns
    ) == AGGTRADE_COLUMNS

    assert (
        result.iloc[0][
            "available_at"
        ]
        == result.iloc[0][
            "trade_time"
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


def test_runtime_ingestion_occurs_after_trade():
    payload = [
        {
            "a": 124,
            "p": "100001.00",
            "q": "0.100",
            "f": 501,
            "l": 501,
            "T": 1786406400000,
            "m": True,
        }
    ]

    result = _parse_aggtrades(
        payload,
        symbol="BTCUSDT",
    )

    assert (
        result.iloc[0]["ingested_at"]
        >= result.iloc[0]["available_at"]
    )


def test_empty_runtime_parser_has_new_schema():
    result = _parse_aggtrades(
        [],
        symbol="BTCUSDT",
    )

    assert result.empty

    assert list(
        result.columns
    ) == AGGTRADE_COLUMNS

    assert (
        str(
            result[
                "trade_time"
            ].dtype
        )
        == "datetime64[ns, UTC]"
    )

    assert (
        str(
            result[
                "available_at"
            ].dtype
        )
        == "datetime64[ns, UTC]"
    )

    assert (
        str(
            result[
                "ingested_at"
            ].dtype
        )
        == "datetime64[ns, UTC]"
    )


def test_partial_time_metadata_rejected():
    source = make_legacy_aggtrades()

    source["available_at"] = (
        pd.to_datetime(
            source[
                "trade_time"
            ],
            utc=True,
        )
    )

    with pytest.raises(
        AggTradeSchemaError
    ):
        normalize_aggtrades(
            source
        )


def test_unknown_ingestion_does_not_get_fake_timestamp():
    result = normalize_aggtrades(
        make_legacy_aggtrades()
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .eq("unknown")
        .all()
    )

    assert (
        result[
            "ingested_at"
        ]
        .isna()
        .all()
    )


def test_normalizing_runtime_data_is_idempotent():
    payload = [
        {
            "a": 125,
            "p": "100002.00",
            "q": "0.300",
            "f": 502,
            "l": 502,
            "T": 1786406400000,
            "m": False,
        }
    ]

    first = _parse_aggtrades(
        payload,
        symbol="BTCUSDT",
    )

    second = normalize_aggtrades(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )
