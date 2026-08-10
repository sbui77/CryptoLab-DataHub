import pandas as pd

from cryptolab.quality.taker_flow import (
    audit_taker_flow,
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

            "period": [
                "5m",
            ] * 3,

            "timestamp": pd.to_datetime(
                [
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T00:05:00Z",
                    "2026-08-10T00:10:00Z",
                ],
                utc=True,
            ),

            "buy_volume": [
                100.0,
                150.0,
                200.0,
            ],

            "sell_volume": [
                80.0,
                150.0,
                100.0,
            ],

            "buy_sell_ratio": [
                1.25,
                1.0,
                2.0,
            ],
        }
    )


def test_good_audit():
    result = audit_taker_flow(
        make_df()
    )

    assert result.rows == 3
    assert result.duplicate_count == 0
    assert result.gap_count == 0
    assert (
        result.ratio_identity_mismatch_count
        == 0
    )

    assert result.structurally_valid is True
    assert result.continuous is True
    assert result.valid is True


def test_bad_ratio_detected():
    df = make_df()

    df.loc[
        0,
        "buy_sell_ratio",
    ] = 10.0

    result = audit_taker_flow(
        df
    )

    assert (
        result.ratio_identity_mismatch_count
        == 1
    )

    assert result.valid is False


def test_gap_detected():
    df = make_df()

    df.loc[
        2,
        "timestamp",
    ] = pd.Timestamp(
        "2026-08-10T00:20:00Z"
    )

    result = audit_taker_flow(
        df
    )

    assert result.gap_count == 1
    assert result.continuous is False
