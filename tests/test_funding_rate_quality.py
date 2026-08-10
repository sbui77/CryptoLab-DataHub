import pandas as pd

from cryptolab.quality.funding_rate import (
    audit_funding_rate,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
            ] * 3,

            "market": [
                "usdm_perpetual",
            ] * 3,

            "symbol": [
                "BTCUSDT",
            ] * 3,

            "funding_time": (
                pd.to_datetime(
                    [
                        "2026-08-01T00:00:00Z",
                        "2026-08-01T08:00:00Z",
                        "2026-08-01T16:00:00Z",
                    ],
                    utc=True,
                )
            ),

            "funding_rate": [
                0.0001,
                0.0002,
                -0.0001,
            ],

            "mark_price": [
                60000.0,
                60100.0,
                60200.0,
            ],

            "rate_type": [
                "Regular",
                "Regular",
                "Regular",
            ],
        }
    )


def test_good_funding_audit():
    result = audit_funding_rate(
        make_df()
    )

    assert result.rows == 3
    assert result.duplicate_count == 0

    assert (
        result.median_interval_hours
        == 8.0
    )

    assert (
        result.structurally_valid
        is True
    )

    assert result.valid is True


def test_duplicate_detected():
    df = make_df()

    df.loc[
        2,
        "funding_time",
    ] = df.loc[
        1,
        "funding_time",
    ]

    result = audit_funding_rate(
        df
    )

    assert (
        result.duplicate_count
        == 1
    )

    assert (
        result.structurally_valid
        is False
    )


def test_unknown_rate_type():
    df = make_df()

    df.loc[
        0,
        "rate_type",
    ] = "Unknown"

    result = audit_funding_rate(
        df
    )

    assert (
        result.unknown_rate_type_count
        == 1
    )

    assert result.valid is False
