from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ============================================================
# RESULT MODEL
# ============================================================


@dataclass
class CheckResult:
    step: str
    name: str
    status: str
    detail: str


VALID_STATUSES = {
    "PASS",
    "FAIL",
    "MISSING",
    "REVIEW",
}


def make_result(
    step: str,
    name: str,
    status: str,
    detail: str,
) -> CheckResult:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Unknown checkpoint status: {status}"
        )

    return CheckResult(
        step=step,
        name=name,
        status=status,
        detail=detail,
    )


def safe_check(
    step: str,
    name: str,
    fn: Callable[[], CheckResult],
) -> CheckResult:
    """
    Run one checkpoint without allowing one failure
    to abort the entire Step-8 review.
    """

    try:
        return fn()

    except ModuleNotFoundError as exc:
        return make_result(
            step,
            name,
            "MISSING",
            f"Module missing: {exc}",
        )

    except FileNotFoundError as exc:
        return make_result(
            step,
            name,
            "MISSING",
            f"File missing: {exc}",
        )

    except Exception as exc:
        return make_result(
            step,
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 8.2 OPEN INTEREST
# ============================================================


def check_open_interest() -> CheckResult:
    from cryptolab.pipelines.open_interest_storage import (
        read_open_interest,
    )

    df = read_open_interest(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    if df.empty:
        return make_result(
            "8.2",
            "Open Interest",
            "MISSING",
            "No local BTCUSDT 5m Open Interest rows.",
        )

    required = [
        "timestamp",
        "open_interest_base",
        "open_interest_quote",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        return make_result(
            "8.2",
            "Open Interest",
            "FAIL",
            f"Missing columns: {missing}",
        )

    duplicate_count = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    invalid_count = int(
        (
            df["open_interest_base"].isna()
            | df["open_interest_quote"].isna()
            | (
                df["open_interest_base"]
                <= 0
            )
            | (
                df["open_interest_quote"]
                <= 0
            )
        ).sum()
    )

    monotonic = bool(
        df["timestamp"]
        .is_monotonic_increasing
    )

    if (
        duplicate_count == 0
        and invalid_count == 0
        and monotonic
    ):
        return make_result(
            "8.2",
            "Open Interest",
            "PASS",
            (
                f"rows={len(df)}, "
                f"first={df['timestamp'].min()}, "
                f"last={df['timestamp'].max()}"
            ),
        )

    return make_result(
        "8.2",
        "Open Interest",
        "FAIL",
        (
            f"duplicates={duplicate_count}, "
            f"invalid={invalid_count}, "
            f"monotonic={monotonic}"
        ),
    )


# ============================================================
# 8.3 FUNDING
# ============================================================


def check_funding() -> CheckResult:
    from cryptolab.pipelines.funding_rate_storage import (
        read_funding_rate,
    )
    from cryptolab.quality.funding_rate import (
        audit_funding_rate,
    )

    df = read_funding_rate(
        exchange="binance",
        symbol="BTCUSDT",
    )

    if df.empty:
        return make_result(
            "8.3",
            "Funding Rate",
            "MISSING",
            "No local funding-rate rows.",
        )

    audit = audit_funding_rate(
        df
    )

    if audit.valid:
        return make_result(
            "8.3",
            "Funding Rate",
            "PASS",
            (
                f"rows={audit.rows}, "
                f"first={audit.first_funding_time}, "
                f"last={audit.last_funding_time}, "
                f"median_interval_h="
                f"{audit.median_interval_hours}"
            ),
        )

    return make_result(
        "8.3",
        "Funding Rate",
        "FAIL",
        (
            f"duplicates={audit.duplicate_count}, "
            f"invalid_rate={audit.invalid_rate_count}, "
            f"invalid_mark_price="
            f"{audit.invalid_mark_price_count}"
        ),
    )


# ============================================================
# 8.4 BASIS
# ============================================================


def check_basis() -> CheckResult:
    from cryptolab.pipelines.basis_storage import (
        read_basis,
    )
    from cryptolab.quality.basis import (
        audit_basis,
    )

    df = read_basis(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    if df.empty:
        return make_result(
            "8.4",
            "Premium / Basis",
            "MISSING",
            "No local BTCUSDT 5m basis rows.",
        )

    audit = audit_basis(
        df
    )

    if (
        audit.structurally_valid
        and audit.continuous
    ):
        return make_result(
            "8.4",
            "Premium / Basis",
            "PASS",
            (
                f"rows={audit.rows}, "
                f"first={audit.first_timestamp}, "
                f"last={audit.last_timestamp}, "
                f"gaps={audit.gap_count}, "
                f"rate_mismatch="
                f"{audit.basis_rate_identity_mismatch_count}. "
                "Local dataset clean; runtime incremental "
                "verification remains a separate check."
            ),
        )

    return make_result(
        "8.4",
        "Premium / Basis",
        "FAIL",
        (
            f"duplicates={audit.duplicate_count}, "
            f"gaps={audit.gap_count}, "
            f"basis_mismatch="
            f"{audit.basis_identity_mismatch_count}, "
            f"rate_mismatch="
            f"{audit.basis_rate_identity_mismatch_count}"
        ),
    )


# ============================================================
# 8.5 LIQUIDATIONS
# ============================================================


def check_liquidations() -> CheckResult:
    from cryptolab.pipelines.liquidation_storage import (
        read_liquidations,
    )
    from cryptolab.quality.liquidations import (
        audit_liquidations,
    )

    df = read_liquidations(
        exchange="binance",
        symbol="BTCUSDT",
    )

    if df.empty:
        return make_result(
            "8.5",
            "Liquidations",
            "REVIEW",
            (
                "No captured liquidation events. "
                "This does not prove zero liquidations; "
                "collector coverage/heartbeat is not yet "
                "implemented."
            ),
        )

    audit = audit_liquidations(
        df
    )

    if audit.valid:
        return make_result(
            "8.5",
            "Liquidations",
            "REVIEW",
            (
                f"Captured rows={audit.rows}, "
                f"first={audit.first_event_time}, "
                f"last={audit.last_event_time}. "
                "Event integrity passes, but continuous "
                "collector coverage cannot yet be proven "
                "without heartbeat."
            ),
        )

    return make_result(
        "8.5",
        "Liquidations",
        "FAIL",
        (
            f"duplicates={audit.duplicate_count}, "
            f"invalid_side={audit.invalid_side_count}, "
            f"invalid_notional="
            f"{audit.invalid_notional_count}, "
            f"notional_mismatch="
            f"{audit.notional_identity_mismatch_count}"
        ),
    )


# ============================================================
# 8.5A FUTURES TAKER FLOW
# ============================================================


def check_taker_flow() -> CheckResult:
    from cryptolab.pipelines.taker_flow_storage import (
        read_taker_flow,
    )
    from cryptolab.quality.taker_flow import (
        audit_taker_flow,
    )

    df = read_taker_flow(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    if df.empty:
        return make_result(
            "8.5A",
            "Futures Taker Flow",
            "MISSING",
            "No local BTCUSDT 5m futures taker-flow rows.",
        )

    audit = audit_taker_flow(
        df
    )

    if (
        audit.structurally_valid
        and audit.continuous
    ):
        return make_result(
            "8.5A",
            "Futures Taker Flow",
            "PASS",
            (
                f"rows={audit.rows}, "
                f"first={audit.first_timestamp}, "
                f"last={audit.last_timestamp}, "
                f"gaps={audit.gap_count}"
            ),
        )

    return make_result(
        "8.5A",
        "Futures Taker Flow",
        "FAIL",
        (
            f"duplicates={audit.duplicate_count}, "
            f"gaps={audit.gap_count}, "
            f"ratio_mismatch="
            f"{audit.ratio_identity_mismatch_count}"
        ),
    )


# ============================================================
# 8.6 DERIVATIVES FEATURES
# ============================================================


def check_derivatives_features() -> CheckResult:
    from cryptolab.pipelines.derivatives_features import (
        build_derivatives_feature_dataset,
    )

    df = build_derivatives_feature_dataset(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
    )

    if df.empty:
        return make_result(
            "8.6",
            "Derivatives Features",
            "MISSING",
            "Feature builder returned empty dataframe.",
        )

    required = [
        "timestamp",
        "open_interest_quote",
        "funding_time",
        "funding_rate",
        "basis_rate",
        "futures_buy_volume",
        "futures_sell_volume",
        "futures_taker_delta",
        "futures_taker_delta_pct",
        "total_liquidation_notional",
        "liquidation_delta",
        "liquidation_imbalance",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        return make_result(
            "8.6",
            "Derivatives Features",
            "FAIL",
            f"Missing feature columns: {missing}",
        )

    duplicate_count = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    monotonic = bool(
        df["timestamp"]
        .is_monotonic_increasing
    )

    taker_mask = (
        df["futures_taker_delta"].notna()
        & df["futures_buy_volume"].notna()
        & df["futures_sell_volume"].notna()
    )

    taker_identity = True

    if taker_mask.any():
        taker_identity = bool(
            np.allclose(
                df.loc[
                    taker_mask,
                    "futures_taker_delta",
                ],
                (
                    df.loc[
                        taker_mask,
                        "futures_buy_volume",
                    ]
                    - df.loc[
                        taker_mask,
                        "futures_sell_volume",
                    ]
                ),
                rtol=1e-12,
                atol=1e-8,
            )
        )

    funding_mask = (
        df["funding_time"].notna()
    )

    funding_causal = True

    if funding_mask.any():
        funding_causal = bool(
            (
                df.loc[
                    funding_mask,
                    "funding_time",
                ]
                <= df.loc[
                    funding_mask,
                    "timestamp",
                ]
            ).all()
        )

    liq_range = bool(
        df[
            "liquidation_imbalance"
        ]
        .dropna()
        .between(
            -1.0,
            1.0,
        )
        .all()
    )

    passed = all(
        [
            duplicate_count == 0,
            monotonic,
            taker_identity,
            funding_causal,
            liq_range,
        ]
    )

    if passed:
        return make_result(
            "8.6",
            "Derivatives Features",
            "PASS",
            (
                f"rows={len(df)}, "
                f"taker_identity={taker_identity}, "
                f"funding_causal={funding_causal}, "
                f"liq_range={liq_range}"
            ),
        )

    return make_result(
        "8.6",
        "Derivatives Features",
        "FAIL",
        (
            f"duplicates={duplicate_count}, "
            f"monotonic={monotonic}, "
            f"taker_identity={taker_identity}, "
            f"funding_causal={funding_causal}, "
            f"liq_range={liq_range}"
        ),
    )


# ============================================================
# 8.7 PRICE × OI
# ============================================================


def check_price_oi() -> CheckResult:
    from cryptolab.pipelines.price_oi_state import (
        build_price_oi_state_dataset,
    )

    df = build_price_oi_state_dataset(
        exchange="binance",
        symbol="BTCUSDT",
        price_timeframe="1h",
        derivatives_period="5m",
    )

    valid = df[
        df[
            "price_oi_state_valid"
        ]
    ]

    if valid.empty:
        return make_result(
            "8.7",
            "Price × OI State",
            "FAIL",
            "No valid overlapping Price × OI rows.",
        )

    allowed = {
        "position_build_up",
        "short_covering_candidate",
        "short_build_up_candidate",
        "long_deleveraging_candidate",
        "neutral",
    }

    states_ok = (
        set(
            valid[
                "price_oi_state"
            ]
            .dropna()
            .unique()
        )
        <= allowed
    )

    invalid_unknown = bool(
        df.loc[
            ~df[
                "price_oi_state_valid"
            ],
            "price_oi_state",
        ]
        .eq("unknown")
        .all()
    )

    duplicate_count = int(
        df["open_time"]
        .duplicated()
        .sum()
    )

    monotonic = bool(
        df["open_time"]
        .is_monotonic_increasing
    )

    if (
        states_ok
        and invalid_unknown
        and duplicate_count == 0
        and monotonic
    ):
        return make_result(
            "8.7",
            "Price × OI State",
            "PASS",
            (
                f"valid_rows={len(valid)}, "
                f"first_valid={valid['open_time'].min()}, "
                f"last_valid={valid['open_time'].max()}"
            ),
        )

    return make_result(
        "8.7",
        "Price × OI State",
        "FAIL",
        (
            f"states_ok={states_ok}, "
            f"invalid_unknown={invalid_unknown}, "
            f"duplicates={duplicate_count}, "
            f"monotonic={monotonic}"
        ),
    )


# ============================================================
# 8.8 SPOT × PERP
# ============================================================


def check_spot_perp() -> CheckResult:
    from cryptolab.pipelines.spot_perp_flow import (
        build_spot_perp_flow_dataset,
    )

    df = build_spot_perp_flow_dataset(
        exchange="binance",
        symbol="BTCUSDT",
        spot_timeframe="1m",
        derivatives_period="5m",
    )

    valid = df[
        df[
            "spot_perp_flow_valid"
        ]
    ]

    if valid.empty:
        return make_result(
            "8.8",
            "Spot × Perp Flow",
            "REVIEW",
            (
                "No valid Spot × Perp overlapping rows. "
                "Inspect Spot Trade Flow coverage and "
                "1m→5m alignment."
            ),
        )

    allowed = {
        "confirmed_buy",
        "confirmed_sell",
        "spot_led_buy",
        "spot_led_sell",
        "perp_led_buy",
        "perp_led_sell",
        "spot_buy_perp_sell",
        "spot_sell_perp_buy",
        "neutral",
    }

    states_ok = (
        set(
            valid[
                "spot_perp_flow_state"
            ]
            .dropna()
            .unique()
        )
        <= allowed
    )

    spread_error = (
        valid[
            "spot_perp_delta_spread"
        ]
        - (
            valid[
                "spot_delta_pct"
            ]
            - valid[
                "perp_delta_pct"
            ]
        )
    ).abs()

    spread_ok = bool(
        spread_error.max()
        < 1e-12
    )

    duplicate_ok = (
        df[
            "timestamp"
        ]
        .duplicated()
        .sum()
        == 0
    )

    monotonic_ok = bool(
        df[
            "timestamp"
        ]
        .is_monotonic_increasing
    )

    valid_share = (
        len(valid)
        / len(df)
        if len(df) > 0
        else 0.0
    )

    if (
        states_ok
        and spread_ok
        and duplicate_ok
        and monotonic_ok
    ):
        status = (
            "PASS"
            if valid_share >= 0.90
            else "REVIEW"
        )

        return make_result(
            "8.8",
            "Spot × Perp Flow",
            status,
            (
                f"valid={len(valid)}/{len(df)} "
                f"({valid_share:.1%}), "
                f"first_valid={valid['timestamp'].min()}, "
                f"last_valid={valid['timestamp'].max()}. "
                + (
                    ""
                    if status == "PASS"
                    else (
                        "Overlap exists but remains below "
                        "the 90% checkpoint threshold."
                    )
                )
            ),
        )

    return make_result(
        "8.8",
        "Spot × Perp Flow",
        "FAIL",
        (
            f"states_ok={states_ok}, "
            f"spread_ok={spread_ok}, "
            f"duplicates={not duplicate_ok}, "
            f"monotonic={monotonic_ok}"
        ),
    )


# ============================================================
# 8.9 DERIVATIVES REGIME
# ============================================================


def check_regime() -> CheckResult:
    from cryptolab.pipelines.derivatives_regime import (
        build_derivatives_regime_dataset,
    )

    df = build_derivatives_regime_dataset(
        exchange="binance",
        symbol="BTCUSDT",
        price_timeframe="1h",
        derivatives_period="5m",
    )

    required = [
        "open_time",
        "as_of_time",
        "funding_time",
        "funding_rate",
        "funding_rate_bps",
        "derivatives_regime",
        "derivatives_regime_confidence",
        "derivatives_regime_valid",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        return make_result(
            "8.9",
            "Derivatives Regime",
            "FAIL",
            f"Missing columns: {missing}",
        )

    valid = df[
        df[
            "derivatives_regime_valid"
        ]
    ]

    if valid.empty:
        return make_result(
            "8.9",
            "Derivatives Regime",
            "FAIL",
            "No valid regime rows.",
        )

    allowed = {
        "leveraged_long_build",
        "leveraged_short_build",
        "short_covering",
        "long_deleveraging",
        "short_squeeze",
        "long_flush",
        "balanced",
        "mixed",
    }

    states_ok = (
        set(
            valid[
                "derivatives_regime"
            ]
            .dropna()
            .unique()
        )
        <= allowed
    )

    confidence_ok = bool(
        valid[
            "derivatives_regime_confidence"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    expected_as_of = (
        df[
            "open_time"
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    as_of_contract = bool(
        df[
            "as_of_time"
        ]
        .eq(
            expected_as_of
        )
        .all()
    )

    funding_mask = (
        df[
            "funding_time"
        ].notna()
    )

    funding_causal = True

    if funding_mask.any():
        funding_causal = bool(
            (
                df.loc[
                    funding_mask,
                    "funding_time",
                ]
                <= df.loc[
                    funding_mask,
                    "as_of_time",
                ]
            )
            .all()
        )

    duplicate_ok = (
        df[
            "open_time"
        ]
        .duplicated()
        .sum()
        == 0
    )

    monotonic_ok = bool(
        df[
            "open_time"
        ]
        .is_monotonic_increasing
    )

    if (
        states_ok
        and confidence_ok
        and as_of_contract
        and funding_causal
        and duplicate_ok
        and monotonic_ok
    ):
        return make_result(
            "8.9",
            "Derivatives Regime",
            "PASS",
            (
                f"valid_rows={len(valid)}, "
                f"as_of_contract={as_of_contract}, "
                f"funding_causal={funding_causal}"
            ),
        )

    return make_result(
        "8.9",
        "Derivatives Regime",
        "FAIL",
        (
            f"states_ok={states_ok}, "
            f"confidence_ok={confidence_ok}, "
            f"as_of_contract={as_of_contract}, "
            f"funding_causal={funding_causal}, "
            f"duplicates={not duplicate_ok}, "
            f"monotonic={monotonic_ok}"
        ),
    )


# ============================================================
# 8.10 QUALITY
# ============================================================


def check_quality() -> CheckResult:
    from cryptolab.pipelines.derivatives_quality import (
        build_derivatives_quality_dataset,
    )

    df = build_derivatives_quality_dataset(
        exchange="binance",
        symbol="BTCUSDT",
        price_timeframe="1h",
        derivatives_period="5m",
    )

    quality_valid = df[
        df[
            "derivatives_regime_quality_valid"
        ]
    ]

    if quality_valid.empty:
        return make_result(
            "8.10",
            "Quality & Validity Masks",
            "FAIL",
            "No quality-valid regime rows.",
        )

    required = [
        "open_time",
        "as_of_time",
        "funding_time",
        "funding_available",
        "funding_causal",
        "funding_age_hours",
        "funding_valid",
        "derivatives_quality_score",
        "derivatives_regime_evidence_count",
        "derivatives_core_valid",
        "derivatives_regime_valid",
        "derivatives_regime_quality_valid",
        "liquidation_valid",
        "total_liquidation_notional_1h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        return make_result(
            "8.10",
            "Quality & Validity Masks",
            "FAIL",
            f"Missing quality columns: {missing}",
        )

    score_ok = bool(
        df[
            "derivatives_quality_score"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    evidence_ok = bool(
        df[
            "derivatives_regime_evidence_count"
        ]
        .between(
            0,
            6,
        )
        .all()
    )

    expected_as_of = (
        df[
            "open_time"
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    as_of_contract = bool(
        df[
            "as_of_time"
        ]
        .eq(
            expected_as_of
        )
        .all()
    )

    funding_causal_ok = bool(
        df.loc[
            df[
                "funding_available"
            ],
            "funding_causal",
        ]
        .eq(True)
        .all()
    )

    funding_age_ok = bool(
        df.loc[
            df[
                "funding_valid"
            ],
            "funding_age_hours",
        ]
        .between(
            0.0,
            12.0,
        )
        .all()
    )

    core_identity = bool(
        df[
            "derivatives_core_valid"
        ]
        .eq(
            df[
                [
                    "oi_valid",
                    "funding_valid",
                    "basis_valid",
                    "taker_flow_valid",
                ]
            ]
            .all(axis=1)
        )
        .all()
    )

    zero_liq_ok = bool(
        df.loc[
            df[
                "total_liquidation_notional_1h"
            ].eq(0),
            "liquidation_valid",
        ]
        .eq(False)
        .all()
    )

    regime_identity = bool(
        df[
            "derivatives_regime_quality_valid"
        ]
        .eq(
            df[
                "derivatives_regime_valid"
            ]
            .fillna(False)
            & df[
                "derivatives_core_valid"
            ]
        )
        .all()
    )

    duplicate_ok = (
        df[
            "open_time"
        ]
        .duplicated()
        .sum()
        == 0
    )

    monotonic_ok = bool(
        df[
            "open_time"
        ]
        .is_monotonic_increasing
    )

    passed = all(
        [
            score_ok,
            evidence_ok,
            as_of_contract,
            funding_causal_ok,
            funding_age_ok,
            core_identity,
            zero_liq_ok,
            regime_identity,
            duplicate_ok,
            monotonic_ok,
        ]
    )

    if passed:
        return make_result(
            "8.10",
            "Quality & Validity Masks",
            "PASS",
            (
                f"quality_valid={len(quality_valid)}, "
                f"latest_valid="
                f"{quality_valid['open_time'].max()}, "
                f"spot_perp_valid="
                f"{int(df['spot_perp_valid'].sum())}, "
                f"liquidation_valid="
                f"{int(df['liquidation_valid'].sum())}"
            ),
        )

    return make_result(
        "8.10",
        "Quality & Validity Masks",
        "FAIL",
        (
            f"score={score_ok}, "
            f"evidence={evidence_ok}, "
            f"as_of={as_of_contract}, "
            f"funding_causal={funding_causal_ok}, "
            f"funding_age={funding_age_ok}, "
            f"core={core_identity}, "
            f"zero_liq={zero_liq_ok}, "
            f"regime_identity={regime_identity}, "
            f"duplicates={not duplicate_ok}, "
            f"monotonic={monotonic_ok}"
        ),
    )


# ============================================================
# RUNTIME / PRODUCTION DEBT
# ============================================================


def check_basis_runtime() -> CheckResult:
    return make_result(
        "8.4R",
        "Basis incremental runtime",
        "REVIEW",
        (
            "Local basis audit passes, but the corrected "
            "startTime+endTime incremental request still needs "
            "one clean live verification."
        ),
    )


def check_liquidation_heartbeat() -> CheckResult:
    return make_result(
        "8.12D",
        "Liquidation collector heartbeat",
        "MISSING",
        (
            "No collector heartbeat/uptime contract yet. "
            "Required in Step 8.12 to distinguish "
            "'online with zero events' from "
            "'collector offline'."
        ),
    )


def check_shared_http_client() -> CheckResult:
    return make_result(
        "8.12R",
        "Shared Binance Futures HTTP client",
        "MISSING",
        (
            "OI/Funding/Basis/Taker Flow still need a common "
            "rate-limit/backoff/retry client in Step 8.12."
        ),
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    checks = [
        (
            "8.2",
            "Open Interest",
            check_open_interest,
        ),
        (
            "8.3",
            "Funding Rate",
            check_funding,
        ),
        (
            "8.4",
            "Premium / Basis",
            check_basis,
        ),
        (
            "8.5",
            "Liquidations",
            check_liquidations,
        ),
        (
            "8.5A",
            "Futures Taker Flow",
            check_taker_flow,
        ),
        (
            "8.6",
            "Derivatives Features",
            check_derivatives_features,
        ),
        (
            "8.7",
            "Price × OI State",
            check_price_oi,
        ),
        (
            "8.8",
            "Spot × Perp Flow",
            check_spot_perp,
        ),
        (
            "8.9",
            "Derivatives Regime",
            check_regime,
        ),
        (
            "8.10",
            "Quality & Validity Masks",
            check_quality,
        ),
        (
            "8.4R",
            "Basis incremental runtime",
            check_basis_runtime,
        ),
        (
            "8.12D",
            "Liquidation collector heartbeat",
            check_liquidation_heartbeat,
        ),
        (
            "8.12R",
            "Shared Binance HTTP client",
            check_shared_http_client,
        ),
    ]

    results: list[CheckResult] = []

    print()
    print("=" * 120)
    print(
        "STEP 8.10A — EVIDENCE INTEGRITY CHECKPOINT"
    )
    print("=" * 120)

    for (
        step,
        name,
        function,
    ) in checks:
        print(
            f"Checking {step:6s} "
            f"{name}..."
        )

        result = safe_check(
            step,
            name,
            function,
        )

        results.append(
            result
        )

    print()
    print("=" * 120)
    print("CHECKPOINT RESULTS")
    print("=" * 120)

    status_width = 8
    step_width = 7
    name_width = 34

    for result in results:
        print(
            f"{result.step:<{step_width}} "
            f"{result.status:<{status_width}} "
            f"{result.name:<{name_width}} "
            f"{result.detail}"
        )

    counts = {
        status: sum(
            result.status == status
            for result in results
        )
        for status in [
            "PASS",
            "FAIL",
            "MISSING",
            "REVIEW",
        ]
    }

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print(
        "PASS    :",
        counts["PASS"],
    )

    print(
        "FAIL    :",
        counts["FAIL"],
    )

    print(
        "MISSING :",
        counts["MISSING"],
    )

    print(
        "REVIEW  :",
        counts["REVIEW"],
    )

    # --------------------------------------------------------
    # Core evidence gates.
    #
    # 8.5 Liquidations is intentionally excluded as a hard
    # blocker because captured-event integrity is audited,
    # while continuous collector coverage belongs to 8.12.
    # --------------------------------------------------------

    core_steps = {
        "8.2",
        "8.3",
        "8.4",
        "8.5A",
        "8.6",
        "8.7",
        "8.8",
        "8.9",
        "8.10",
    }

    core_results = [
        result
        for result in results
        if result.step
        in core_steps
    ]

    core_failures = [
        result
        for result in core_results
        if result.status
        in {
            "FAIL",
            "MISSING",
        }
    ]

    core_reviews = [
        result
        for result in core_results
        if result.status
        == "REVIEW"
    ]

    print()
    print("=" * 120)

    if (
        not core_failures
        and not core_reviews
    ):
        print(
            "STEP 8.10A CORE EVIDENCE: PASSED"
        )

        print(
            "Ready to proceed to Step 8.11. "
            "Liquidation heartbeat, shared HTTP client, "
            "and live basis incremental verification "
            "remain explicit production debt."
        )

    elif not core_failures:
        print(
            "STEP 8.10A CORE EVIDENCE: REVIEW REQUIRED"
        )

        print(
            "No hard core failures, but one or more "
            "core evidence streams still require review."
        )

    else:
        print(
            "STEP 8.10A CORE EVIDENCE: FAILED"
        )

        print(
            "Resolve FAIL/MISSING core checks "
            "before Step 8.11."
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
