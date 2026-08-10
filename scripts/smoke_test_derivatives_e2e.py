from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from cryptolab.pipelines.derivatives_features import (
    build_derivatives_feature_dataset,
)
from cryptolab.pipelines.derivatives_quality import (
    build_derivatives_quality_dataset,
)
from cryptolab.pipelines.derivatives_quality_covered import (
    build_covered_derivatives_quality_dataset,
)
from cryptolab.pipelines.derivatives_regime import (
    build_derivatives_regime_dataset,
)
from cryptolab.pipelines.derivatives_snapshot import (
    build_derivatives_snapshot_dataset,
)
from cryptolab.pipelines.derivatives_update import (
    update_derivatives,
)
from cryptolab.pipelines.liquidation_heartbeat import (
    read_liquidation_heartbeat,
)
from cryptolab.pipelines.liquidation_storage import (
    read_liquidations,
)


EXCHANGE = "binance"
SYMBOL = "BTCUSDT"
PERIOD = "5m"
PRICE_TIMEFRAME = "1h"


def check(
    name: str,
    passed: bool,
    detail: str,
) -> tuple[str, bool, str]:
    return (
        name,
        passed,
        detail,
    )


def main() -> None:
    results: list[
        tuple[str, bool, str]
    ] = []

    print()
    print("=" * 120)
    print(
        "STEP 8.12.10 — DERIVATIVES END-TO-END "
        "PRODUCTION SMOKE TEST"
    )
    print("=" * 120)

    # ========================================================
    # 1. INCREMENTAL REST UPDATE
    # ========================================================

    print()
    print("1. REST UPDATE")
    print("-" * 120)

    update = update_derivatives(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        period=PERIOD,
        continue_on_error=True,
    )

    streams = [
        update.open_interest,
        update.funding_rate,
        update.basis,
        update.taker_flow,
    ]

    for stream in streams:
        status = (
            "PASS"
            if stream.success
            else "FAIL"
        )

        print(
            f"{stream.name:20s}: "
            f"{status:4s} "
            f"rows={stream.rows_fetched} "
            f"up_to_date={stream.up_to_date}"
        )

        if not stream.success:
            print(
                " " * 22
                + f"{stream.error_type}: "
                + f"{stream.error_message}"
            )

    core_rest_ok = all(
        [
            update.open_interest.success,
            update.funding_rate.success,
            update.taker_flow.success,
        ]
    )

    results.append(
        check(
            "REST core streams",
            core_rest_ok,
            (
                "OI/Funding/Taker Flow "
                f"success={core_rest_ok}"
            ),
        )
    )

    basis_degraded = (
        not update.basis.success
    )

    results.append(
        check(
            "Basis handled",
            True,
            (
                "PASS"
                if update.basis.success
                else (
                    "DEGRADED but isolated: "
                    f"{update.basis.error_type}"
                )
            ),
        )
    )

    # ========================================================
    # 2. DERIVATIVES FEATURES
    # ========================================================

    print()
    print("2. DERIVATIVES FEATURES")
    print("-" * 120)

    features = (
        build_derivatives_feature_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            period=PERIOD,
        )
    )

    feature_required = [
        "timestamp",
        "open_interest_quote",
        "basis_rate",
        "futures_taker_delta_pct",
        "funding_time",
        "funding_rate",
        "total_liquidation_notional",
    ]

    feature_missing = [
        column
        for column in feature_required
        if column not in features.columns
    ]

    feature_ok = (
        not features.empty
        and not feature_missing
        and features[
            "timestamp"
        ].duplicated().sum() == 0
    )

    print(
        "Rows              :",
        len(features),
    )

    print(
        "First             :",
        (
            features[
                "timestamp"
            ].min()
            if not features.empty
            else None
        ),
    )

    print(
        "Last              :",
        (
            features[
                "timestamp"
            ].max()
            if not features.empty
            else None
        ),
    )

    print(
        "Missing columns   :",
        feature_missing,
    )

    results.append(
        check(
            "Derivatives features",
            feature_ok,
            (
                f"rows={len(features)}, "
                f"missing={feature_missing}"
            ),
        )
    )

    # ========================================================
    # 3. REGIME
    # ========================================================

    print()
    print("3. DERIVATIVES REGIME")
    print("-" * 120)

    regime = (
        build_derivatives_regime_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe=PRICE_TIMEFRAME,
            derivatives_period=PERIOD,
        )
    )

    regime_valid = regime[
        regime[
            "derivatives_regime_valid"
        ]
    ].copy()

    regime_ok = (
        not regime.empty
        and not regime_valid.empty
        and (
            regime[
                "as_of_time"
            ]
            == (
                regime[
                    "open_time"
                ]
                + pd.Timedelta(
                    hours=1
                )
            )
        ).all()
    )

    print(
        "Rows              :",
        len(regime),
    )

    print(
        "Valid rows        :",
        len(regime_valid),
    )

    print(
        "Latest valid      :",
        (
            regime_valid[
                "open_time"
            ].max()
            if not regime_valid.empty
            else None
        ),
    )

    results.append(
        check(
            "Derivatives regime",
            regime_ok,
            (
                f"rows={len(regime)}, "
                f"valid={len(regime_valid)}"
            ),
        )
    )

    # ========================================================
    # 4. QUALITY
    # ========================================================

    print()
    print("4. DERIVATIVES QUALITY")
    print("-" * 120)

    quality = (
        build_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe=PRICE_TIMEFRAME,
            derivatives_period=PERIOD,
        )
    )

    quality_valid = quality[
        quality[
            "derivatives_regime_quality_valid"
        ]
    ].copy()

    quality_ok = (
        not quality.empty
        and not quality_valid.empty
        and quality[
            "derivatives_quality_score"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    )

    print(
        "Rows              :",
        len(quality),
    )

    print(
        "Quality valid     :",
        len(quality_valid),
    )

    print(
        "Latest valid      :",
        (
            quality_valid[
                "open_time"
            ].max()
            if not quality_valid.empty
            else None
        ),
    )

    results.append(
        check(
            "Derivatives quality",
            quality_ok,
            (
                f"rows={len(quality)}, "
                f"valid={len(quality_valid)}"
            ),
        )
    )

    # ========================================================
    # 5. COVERAGE-AWARE QUALITY
    # ========================================================

    print()
    print("5. LIQUIDATION COVERAGE-AWARE QUALITY")
    print("-" * 120)

    covered_quality = (
        build_covered_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe=PRICE_TIMEFRAME,
            derivatives_period=PERIOD,
            minimum_liquidation_coverage_ratio=0.80,
            heartbeat_interval_seconds=30.0,
        )
    )

    uncovered_contract_ok = bool(
        covered_quality.loc[
            ~covered_quality[
                "liquidation_collector_covered"
            ],
            "liquidation_valid",
        ]
        .eq(False)
        .all()
    )

    covered_hours = int(
        covered_quality[
            "liquidation_collector_covered"
        ].sum()
    )

    liq_valid_rows = int(
        covered_quality[
            "liquidation_valid"
        ].sum()
    )

    print(
        "Rows              :",
        len(covered_quality),
    )

    print(
        "Covered hours     :",
        covered_hours,
    )

    print(
        "Liquidation valid :",
        liq_valid_rows,
    )

    print(
        "Unknown contract  :",
        uncovered_contract_ok,
    )

    results.append(
        check(
            "Liquidation coverage",
            uncovered_contract_ok,
            (
                f"covered_hours={covered_hours}, "
                f"liq_valid={liq_valid_rows}"
            ),
        )
    )

    # ========================================================
    # 6. HEARTBEAT / LIQUIDATION RAW
    # ========================================================

    print()
    print("6. LIQUIDATION COLLECTOR STORAGE")
    print("-" * 120)

    heartbeat = (
        read_liquidation_heartbeat(
            exchange=EXCHANGE,
            symbol=SYMBOL,
        )
    )

    liquidations = (
        read_liquidations(
            exchange=EXCHANGE,
            symbol=SYMBOL,
        )
    )

    heartbeat_ok = (
        not heartbeat.empty
        and heartbeat[
            "heartbeat_time"
        ].duplicated().sum() == 0
        and heartbeat[
            "connected"
        ].any()
    )

    liquidation_ok = (
        liquidations[
            "event_time"
        ].duplicated().sum()
        == 0
        if not liquidations.empty
        else True
    )

    print(
        "Heartbeat rows    :",
        len(heartbeat),
    )

    print(
        "Connected HB      :",
        (
            int(
                heartbeat[
                    "connected"
                ].sum()
            )
            if not heartbeat.empty
            else 0
        ),
    )

    print(
        "Liquidation rows  :",
        len(liquidations),
    )

    results.append(
        check(
            "Heartbeat storage",
            heartbeat_ok,
            f"rows={len(heartbeat)}",
        )
    )

    results.append(
        check(
            "Liquidation storage",
            liquidation_ok,
            f"rows={len(liquidations)}",
        )
    )

    # ========================================================
    # 7. SNAPSHOT
    # ========================================================

    print()
    print("7. DERIVATIVES SNAPSHOT")
    print("-" * 120)

    snapshot, frame = (
        build_derivatives_snapshot_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe=PRICE_TIMEFRAME,
            derivatives_period=PERIOD,
        )
    )

    snapshot_ok = all(
        [
            len(frame) == 1,
            snapshot.derivatives_core_valid,
            snapshot.derivatives_regime_quality_valid,
            (
                0.0
                <= snapshot.derivatives_quality_score
                <= 1.0
            ),
            (
                snapshot.as_of_time
                == snapshot.open_time
                + pd.Timedelta(
                    hours=1
                )
            ),
        ]
    )

    print(
        "Open time         :",
        snapshot.open_time,
    )

    print(
        "As-of time        :",
        snapshot.as_of_time,
    )

    print(
        "Regime            :",
        snapshot.derivatives_regime,
    )

    print(
        "Confidence        :",
        snapshot.derivatives_regime_confidence,
    )

    print(
        "Quality score     :",
        snapshot.derivatives_quality_score,
    )

    print(
        "Quality tier      :",
        snapshot.derivatives_quality_tier,
    )

    print(
        "Core valid        :",
        snapshot.derivatives_core_valid,
    )

    print(
        "Regime valid      :",
        snapshot.derivatives_regime_quality_valid,
    )

    results.append(
        check(
            "Snapshot",
            snapshot_ok,
            (
                f"open_time={snapshot.open_time}, "
                f"regime={snapshot.derivatives_regime}"
            ),
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 120)
    print("END-TO-END RESULTS")
    print("=" * 120)

    for (
        name,
        passed,
        detail,
    ) in results:
        print(
            f"{name:28s} "
            f"{'PASS' if passed else 'FAIL':5s} "
            f"{detail}"
        )

    failures = [
        name
        for (
            name,
            passed,
            _
        ) in results
        if not passed
    ]

    print()
    print("=" * 120)

    if not failures:
        if basis_degraded:
            print(
                "STEP 8.12.10 END-TO-END: "
                "PASSED WITH BASIS DEGRADED"
            )
        else:
            print(
                "STEP 8.12.10 END-TO-END: PASSED"
            )
    else:
        print(
            "STEP 8.12.10 END-TO-END: "
            "REVIEW REQUIRED"
        )

        print(
            "Failed checks:",
            ", ".join(
                failures
            ),
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
