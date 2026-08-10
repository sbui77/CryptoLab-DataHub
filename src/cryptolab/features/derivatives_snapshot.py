from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


class DerivativesSnapshotError(ValueError):
    """Raised when derivatives snapshot generation fails."""


@dataclass(frozen=True)
class DerivativesSnapshot:
    exchange: str
    symbol: str
    timeframe: str

    open_time: pd.Timestamp
    as_of_time: pd.Timestamp

    close: float

    price_return_1h: float | None
    oi_quote_change_pct_1h: float | None

    spot_delta_pct_1h: float | None
    perp_delta_pct_1h: float | None

    funding_time: pd.Timestamp | None
    funding_rate: float | None
    basis_rate: float | None

    long_liquidation_notional_1h: float | None
    short_liquidation_notional_1h: float | None
    liquidation_imbalance_1h: float | None

    price_oi_state: str

    derivatives_regime: str
    derivatives_regime_confidence: float

    derivatives_quality_score: float
    derivatives_quality_tier: str
    derivatives_regime_evidence_count: int

    oi_valid: bool
    funding_valid: bool
    basis_valid: bool
    taker_flow_valid: bool
    spot_perp_valid: bool
    liquidation_valid: bool

    derivatives_core_valid: bool
    derivatives_regime_quality_valid: bool


def _optional_float(
    value,
) -> float | None:
    if pd.isna(value):
        return None

    return float(value)


def _optional_timestamp(
    value,
) -> pd.Timestamp | None:
    if pd.isna(value):
        return None

    return pd.Timestamp(value)


def build_derivatives_snapshot(
    df: pd.DataFrame,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
) -> DerivativesSnapshot:
    """
    Build latest quality-valid derivatives snapshot.

    Selection rule
    --------------
    Use the latest row where:

        derivatives_regime_quality_valid == True

    This deliberately avoids exposing a newer but
    quality-invalid regime as the current trusted snapshot.
    """

    if df.empty:
        raise DerivativesSnapshotError(
            "Derivatives quality dataframe is empty"
        )

    required = [
        "open_time",
        "as_of_time",
        "close",
        "price_return_1h",
        "oi_quote_change_pct_1h",
        "spot_delta_pct_1h",
        "perp_delta_pct_1h",
        "funding_time",
        "funding_rate",
        "basis_rate",
        "long_liquidation_notional_1h",
        "short_liquidation_notional_1h",
        "liquidation_imbalance_1h",
        "price_oi_state",
        "derivatives_regime",
        "derivatives_regime_confidence",
        "derivatives_quality_score",
        "derivatives_quality_tier",
        "derivatives_regime_evidence_count",
        "oi_valid",
        "funding_valid",
        "basis_valid",
        "taker_flow_valid",
        "spot_perp_valid",
        "liquidation_valid",
        "derivatives_core_valid",
        "derivatives_regime_quality_valid",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise DerivativesSnapshotError(
            "Missing derivatives snapshot columns: "
            f"{missing}"
        )

    work = df.copy()

    work["open_time"] = pd.to_datetime(
        work["open_time"],
        utc=True,
        errors="coerce",
    )

    work["as_of_time"] = pd.to_datetime(
        work["as_of_time"],
        utc=True,
        errors="coerce",
    )

    work["funding_time"] = pd.to_datetime(
        work["funding_time"],
        utc=True,
        errors="coerce",
    )

    valid = work[
        work[
            "derivatives_regime_quality_valid"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if valid.empty:
        raise DerivativesSnapshotError(
            "No quality-valid derivatives regime rows"
        )

    valid = (
        valid
        .sort_values(
            [
                "as_of_time",
                "open_time",
            ]
        )
        .reset_index(drop=True)
    )

    row = valid.iloc[-1]

    if pd.isna(
        row["derivatives_regime_confidence"]
    ):
        raise DerivativesSnapshotError(
            "Latest quality-valid row has no regime confidence"
        )

    return DerivativesSnapshot(
        exchange=exchange.lower(),
        symbol=symbol.upper(),
        timeframe=timeframe,

        open_time=pd.Timestamp(
            row["open_time"]
        ),

        as_of_time=pd.Timestamp(
            row["as_of_time"]
        ),

        close=float(
            row["close"]
        ),

        price_return_1h=_optional_float(
            row["price_return_1h"]
        ),

        oi_quote_change_pct_1h=_optional_float(
            row["oi_quote_change_pct_1h"]
        ),

        spot_delta_pct_1h=_optional_float(
            row["spot_delta_pct_1h"]
        ),

        perp_delta_pct_1h=_optional_float(
            row["perp_delta_pct_1h"]
        ),

        funding_time=_optional_timestamp(
            row["funding_time"]
        ),

        funding_rate=_optional_float(
            row["funding_rate"]
        ),

        basis_rate=_optional_float(
            row["basis_rate"]
        ),

        long_liquidation_notional_1h=_optional_float(
            row[
                "long_liquidation_notional_1h"
            ]
        ),

        short_liquidation_notional_1h=_optional_float(
            row[
                "short_liquidation_notional_1h"
            ]
        ),

        liquidation_imbalance_1h=_optional_float(
            row[
                "liquidation_imbalance_1h"
            ]
        ),

        price_oi_state=str(
            row["price_oi_state"]
        ),

        derivatives_regime=str(
            row["derivatives_regime"]
        ),

        derivatives_regime_confidence=float(
            row[
                "derivatives_regime_confidence"
            ]
        ),

        derivatives_quality_score=float(
            row[
                "derivatives_quality_score"
            ]
        ),

        derivatives_quality_tier=str(
            row[
                "derivatives_quality_tier"
            ]
        ),

        derivatives_regime_evidence_count=int(
            row[
                "derivatives_regime_evidence_count"
            ]
        ),

        oi_valid=bool(
            row["oi_valid"]
        ),

        funding_valid=bool(
            row["funding_valid"]
        ),

        basis_valid=bool(
            row["basis_valid"]
        ),

        taker_flow_valid=bool(
            row["taker_flow_valid"]
        ),

        spot_perp_valid=bool(
            row["spot_perp_valid"]
        ),

        liquidation_valid=bool(
            row["liquidation_valid"]
        ),

        derivatives_core_valid=bool(
            row["derivatives_core_valid"]
        ),

        derivatives_regime_quality_valid=bool(
            row[
                "derivatives_regime_quality_valid"
            ]
        ),
    )


def derivatives_snapshot_to_dataframe(
    snapshot: DerivativesSnapshot,
) -> pd.DataFrame:
    """
    Convert snapshot dataclass into a one-row dataframe.
    """

    return pd.DataFrame(
        [
            asdict(snapshot)
        ]
    )
