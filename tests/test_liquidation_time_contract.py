from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.pipelines.liquidation_storage import (
    LIQUIDATION_COLUMNS,
    LiquidationStorageError,
    normalize_liquidations,
)


def make_legacy_liquidation() -> pd.DataFrame:
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
            "event_time": [
                pd.Timestamp(
                    "2026-08-11T09:00:00Z"
                ),
            ],
            "order_time": [
                pd.Timestamp(
                    "2026-08-11T09:00:00Z"
                ),
            ],
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
                120000.0,
            ],
            "average_price": [
                119990.0,
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
                119990.0,
            ],
        }
    )


def test_legacy_liquidation_gets_derived_availability():
    result = normalize_liquidations(
        make_legacy_liquidation()
    )

    assert list(
        result.columns
    ) == LIQUIDATION_COLUMNS

    assert (
        result.iloc[0][
            "available_at"
        ]
        == result.iloc[0][
            "event_time"
        ]
    )

    assert (
        result.iloc[0][
            "available_at_quality"
        ]
        == "derived"
    )

    assert pd.isna(
        result.iloc[0][
            "ingested_at"
        ]
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )


def test_exact_ws_runtime_metadata_survives():
    source = make_legacy_liquidation()

    source[
        "available_at"
    ] = pd.Timestamp(
        "2026-08-11T09:00:00.100Z"
    )

    source[
        "available_at_quality"
    ] = "exact"

    source[
        "ingested_at"
    ] = pd.Timestamp(
        "2026-08-11T09:00:00.120Z"
    )

    source[
        "ingested_at_quality"
    ] = "exact"

    result = normalize_liquidations(
        source
    )

    assert (
        result.iloc[0][
            "available_at_quality"
        ]
        == "exact"
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        result.iloc[0][
            "ingested_at"
        ]
        >= result.iloc[0][
            "available_at"
        ]
    )


def test_ingestion_cannot_precede_exact_availability():
    source = make_legacy_liquidation()

    source[
        "available_at"
    ] = pd.Timestamp(
        "2026-08-11T09:00:00.200Z"
    )

    source[
        "available_at_quality"
    ] = "exact"

    source[
        "ingested_at"
    ] = pd.Timestamp(
        "2026-08-11T09:00:00.100Z"
    )

    source[
        "ingested_at_quality"
    ] = "exact"

    with pytest.raises(
        LiquidationStorageError
    ):
        normalize_liquidations(
            source
        )


def test_partial_pit_schema_rejected():
    source = make_legacy_liquidation()

    source[
        "available_at"
    ] = source[
        "event_time"
    ]

    with pytest.raises(
        LiquidationStorageError
    ):
        normalize_liquidations(
            source
        )


def test_normalization_is_idempotent():
    first = normalize_liquidations(
        make_legacy_liquidation()
    )

    second = normalize_liquidations(
        first
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )
