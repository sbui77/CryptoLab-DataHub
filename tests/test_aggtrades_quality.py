from __future__ import annotations

import pandas as pd
import pytest

from cryptolab.quality.aggtrades import (
    AggTradesQualityError,
    audit_aggtrades,
    validate_aggtrades,
)


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
                0.5,
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
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T00:00:01Z",
                    "2026-08-10T00:00:02Z",
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


def test_validate_good_data():
    validate_aggtrades(
        make_df()
    )


def test_duplicate_rejected():
    df = make_df()

    df.loc[
        2,
        "agg_trade_id",
    ] = 101

    with pytest.raises(
        AggTradesQualityError
    ):
        validate_aggtrades(
            df
        )


def test_invalid_maker_mapping_rejected():
    df = make_df()

    df.loc[
        0,
        "taker_side",
    ] = "sell"

    with pytest.raises(
        AggTradesQualityError
    ):
        validate_aggtrades(
            df
        )


def test_audit_continuous():
    result = audit_aggtrades(
        make_df()
    )

    assert result.rows == 3
    assert result.id_gap_count == 0
    assert result.missing_id_count == 0
    assert result.structurally_valid is True
    assert result.continuous is True
    assert result.valid is True


def test_audit_id_gap():
    df = make_df()

    df.loc[
        2,
        "agg_trade_id",
    ] = 105

    result = audit_aggtrades(
        df
    )

    assert result.id_gap_count == 1
    assert result.missing_id_count == 3
    assert result.structurally_valid is True
    assert result.continuous is False
    assert result.valid is True
