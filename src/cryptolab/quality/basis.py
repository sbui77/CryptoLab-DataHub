from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasisAuditResult:
    exchange: str
    symbol: str
    period: str
    contract_type: str

    rows: int

    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    duplicate_count: int
    gap_count: int

    invalid_index_price_count: int
    invalid_futures_price_count: int
    invalid_basis_count: int
    invalid_basis_rate_count: int

    basis_identity_mismatch_count: int
    basis_rate_identity_mismatch_count: int

    timestamp_disorder_count: int

    structurally_valid: bool
    continuous: bool
    valid: bool


PERIOD_TO_DELTA = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}


# Binance basisRate is returned with materially lower
# precision than basis/indexPrice in observed API data.
#
# Example:
#
# basis / index_price ~= -0.00046175
# Binance basisRate   = -0.0005
#
# A reported resolution of 0.0001 implies a rounding
# tolerance of approximately half one unit.
BASIS_RATE_ABS_TOLERANCE = 5.1e-5


def audit_basis(
    df: pd.DataFrame,
) -> BasisAuditResult:
    """
    Audit canonical Binance basis history.

    Identity checks
    ---------------
    basis:
        basis ~= futures_price - index_price

    basis_rate:
        basis_rate ~= basis / index_price

    Important
    ---------
    Binance returns basisRate at lower precision than the
    underlying price fields. Therefore basis_rate is checked
    using a tolerance consistent with its observed reporting
    resolution rather than exact floating-point equality.
    """

    if df.empty:
        return BasisAuditResult(
            exchange="unknown",
            symbol="unknown",
            period="unknown",
            contract_type="unknown",
            rows=0,
            first_timestamp=None,
            last_timestamp=None,
            duplicate_count=0,
            gap_count=0,
            invalid_index_price_count=0,
            invalid_futures_price_count=0,
            invalid_basis_count=0,
            invalid_basis_rate_count=0,
            basis_identity_mismatch_count=0,
            basis_rate_identity_mismatch_count=0,
            timestamp_disorder_count=0,
            structurally_valid=False,
            continuous=False,
            valid=False,
        )

    required_columns = [
        "exchange",
        "symbol",
        "period",
        "contract_type",
        "timestamp",
        "index_price",
        "futures_price",
        "basis",
        "basis_rate",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing basis audit columns: "
            f"{missing_columns}"
        )

    work = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    exchange = str(
        work["exchange"].iloc[0]
    )

    symbol = str(
        work["symbol"].iloc[0]
    )

    period = str(
        work["period"].iloc[0]
    )

    contract_type = str(
        work["contract_type"].iloc[0]
    )

    if period not in PERIOD_TO_DELTA:
        raise ValueError(
            f"Unsupported basis period: {period}"
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_count = int(
        work["timestamp"]
        .duplicated()
        .sum()
    )

    # ========================================================
    # BASIC NUMERIC VALIDITY
    # ========================================================

    invalid_index_price_count = int(
        (
            work["index_price"].isna()
            | (
                work["index_price"]
                <= 0
            )
        ).sum()
    )

    invalid_futures_price_count = int(
        (
            work["futures_price"].isna()
            | (
                work["futures_price"]
                <= 0
            )
        ).sum()
    )

    invalid_basis_count = int(
        (
            work["basis"].isna()
            | ~np.isfinite(
                work["basis"]
            )
        ).sum()
    )

    invalid_basis_rate_count = int(
        (
            work["basis_rate"].isna()
            | ~np.isfinite(
                work["basis_rate"]
            )
        ).sum()
    )

    # ========================================================
    # BASIS IDENTITY
    #
    # Binance basis itself retains high precision, so this
    # relationship can be checked tightly.
    # ========================================================

    expected_basis = (
        work["futures_price"]
        - work["index_price"]
    )

    basis_identity_valid = np.isclose(
        work["basis"],
        expected_basis,
        rtol=1e-9,
        atol=1e-7,
        equal_nan=False,
    )

    basis_identity_mismatch_count = int(
        (
            ~basis_identity_valid
        ).sum()
    )

    # ========================================================
    # BASIS RATE IDENTITY
    #
    # Do NOT require exact equality.
    #
    # basisRate from Binance is lower precision than:
    #
    #     basis / index_price
    #
    # so validate against an absolute rounding tolerance.
    # ========================================================

    expected_basis_rate = np.where(
        work["index_price"] > 0,
        work["basis"]
        / work["index_price"],
        np.nan,
    )

    basis_rate_difference = np.abs(
        work["basis_rate"]
        - expected_basis_rate
    )

    basis_rate_identity_valid = (
        np.isfinite(
            work["basis_rate"]
        )
        & np.isfinite(
            expected_basis_rate
        )
        & (
            basis_rate_difference
            <= BASIS_RATE_ABS_TOLERANCE
        )
    )

    basis_rate_identity_mismatch_count = int(
        (
            ~basis_rate_identity_valid
        ).sum()
    )

    # ========================================================
    # TIMESTAMP / CONTINUITY
    # ========================================================

    diff = (
        work["timestamp"]
        .diff()
    )

    timestamp_disorder_count = int(
        (
            diff
            < pd.Timedelta(0)
        ).sum()
    )

    expected_delta = (
        PERIOD_TO_DELTA[
            period
        ]
    )

    gap_count = int(
        (
            diff
            > expected_delta
        ).sum()
    )

    # ========================================================
    # STATUS
    # ========================================================

    structurally_valid = (
        duplicate_count == 0
        and invalid_index_price_count == 0
        and invalid_futures_price_count == 0
        and invalid_basis_count == 0
        and invalid_basis_rate_count == 0
        and basis_identity_mismatch_count == 0
        and basis_rate_identity_mismatch_count == 0
        and timestamp_disorder_count == 0
    )

    continuous = (
        gap_count == 0
    )

    valid = (
        structurally_valid
    )

    return BasisAuditResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        contract_type=contract_type,
        rows=len(work),
        first_timestamp=pd.Timestamp(
            work["timestamp"].iloc[0]
        ),
        last_timestamp=pd.Timestamp(
            work["timestamp"].iloc[-1]
        ),
        duplicate_count=duplicate_count,
        gap_count=gap_count,
        invalid_index_price_count=(
            invalid_index_price_count
        ),
        invalid_futures_price_count=(
            invalid_futures_price_count
        ),
        invalid_basis_count=(
            invalid_basis_count
        ),
        invalid_basis_rate_count=(
            invalid_basis_rate_count
        ),
        basis_identity_mismatch_count=(
            basis_identity_mismatch_count
        ),
        basis_rate_identity_mismatch_count=(
            basis_rate_identity_mismatch_count
        ),
        timestamp_disorder_count=(
            timestamp_disorder_count
        ),
        structurally_valid=(
            structurally_valid
        ),
        continuous=continuous,
        valid=valid,
    )