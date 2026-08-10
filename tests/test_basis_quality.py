import pandas as pd

from cryptolab.quality.basis import (
    audit_basis,
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

            "contract_type": [
                "PERPETUAL",
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

            "index_price": [
                60000.0,
                60100.0,
                60200.0,
            ],

            "futures_price": [
                60006.0,
                60106.01,
                60193.98,
            ],

            "basis": [
                6.0,
                6.01,
                -6.02,
            ],

            "basis_rate": [
                0.0001,
                0.0001,
                -0.0001,
            ],

            "annualized_basis_rate": [
                float("nan"),
            ] * 3,
        }
    )


def test_good_basis_audit():
    result = audit_basis(
        make_df()
    )

    assert result.rows == 3
    assert result.duplicate_count == 0
    assert result.gap_count == 0

    assert (
        result.basis_identity_mismatch_count
        == 0
    )

    assert (
        result.basis_rate_identity_mismatch_count
        == 0
    )

    assert result.structurally_valid is True
    assert result.continuous is True
    assert result.valid is True


def test_binance_basis_rate_rounding_is_valid():
    """
    Reproduce observed Binance API precision.

    Exact:
        -30.03565217 / 65047.33565217
        ~= -0.00046175

    Binance reports:
        -0.0005

    This is valid exchange-side rounding and must not be
    classified as corruption.
    """

    df = pd.DataFrame(
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

            "contract_type": [
                "PERPETUAL",
            ],

            "period": [
                "5m",
            ],

            "timestamp": pd.to_datetime(
                [
                    "2026-08-10T00:00:00Z",
                ],
                utc=True,
            ),

            "index_price": [
                65047.33565217,
            ],

            "futures_price": [
                65017.30,
            ],

            "basis": [
                -30.03565217,
            ],

            "basis_rate": [
                -0.0005,
            ],

            "annualized_basis_rate": [
                float("nan"),
            ],
        }
    )

    result = audit_basis(
        df
    )

    assert (
        result.basis_identity_mismatch_count
        == 0
    )

    assert (
        result.basis_rate_identity_mismatch_count
        == 0
    )

    assert result.valid is True


def test_duplicate_detected():
    df = make_df()

    df.loc[
        2,
        "timestamp",
    ] = df.loc[
        1,
        "timestamp",
    ]

    result = audit_basis(
        df
    )

    assert result.duplicate_count == 1
    assert result.valid is False


def test_bad_basis_identity_detected():
    df = make_df()

    df.loc[
        0,
        "basis",
    ] = 999.0

    result = audit_basis(
        df
    )

    assert (
        result.basis_identity_mismatch_count
        == 1
    )

    assert result.valid is False


def test_bad_basis_rate_detected():
    df = make_df()

    df.loc[
        0,
        "basis_rate",
    ] = 0.01

    result = audit_basis(
        df
    )

    assert (
        result.basis_rate_identity_mismatch_count
        == 1
    )

    assert result.valid is False