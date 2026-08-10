from __future__ import annotations

from cryptolab.pipelines.derivatives_snapshot import (
    build_and_save_derivatives_snapshot,
)


def main() -> None:
    snapshot, output = (
        build_and_save_derivatives_snapshot(
            exchange="binance",
            symbol="BTCUSDT",
            price_timeframe="1h",
            derivatives_period="5m",
        )
    )

    print()
    print("=" * 100)
    print("DERIVATIVES SNAPSHOT")
    print("=" * 100)

    print(
        "Exchange             :",
        snapshot.exchange,
    )

    print(
        "Symbol               :",
        snapshot.symbol,
    )

    print(
        "Timeframe            :",
        snapshot.timeframe,
    )

    print(
        "Open time            :",
        snapshot.open_time,
    )

    print(
        "As-of time           :",
        snapshot.as_of_time,
    )

    print(
        "Close                :",
        snapshot.close,
    )

    print()
    print("STATE")
    print("-" * 100)

    print(
        "Price × OI           :",
        snapshot.price_oi_state,
    )

    print(
        "Derivatives regime   :",
        snapshot.derivatives_regime,
    )

    print(
        "Regime confidence    :",
        snapshot.derivatives_regime_confidence,
    )

    print()
    print("EVIDENCE")
    print("-" * 100)

    print(
        "Price return 1h      :",
        snapshot.price_return_1h,
    )

    print(
        "OI change 1h         :",
        snapshot.oi_quote_change_pct_1h,
    )

    print(
        "Spot delta 1h        :",
        snapshot.spot_delta_pct_1h,
    )

    print(
        "Perp delta 1h        :",
        snapshot.perp_delta_pct_1h,
    )

    print(
        "Funding rate         :",
        snapshot.funding_rate,
    )

    print(
        "Basis rate           :",
        snapshot.basis_rate,
    )

    print(
        "Liquidation imbalance:",
        snapshot.liquidation_imbalance_1h,
    )

    print()
    print("QUALITY")
    print("-" * 100)

    print(
        "Evidence count       :",
        snapshot.derivatives_regime_evidence_count,
    )

    print(
        "Quality score        :",
        snapshot.derivatives_quality_score,
    )

    print(
        "Quality tier         :",
        snapshot.derivatives_quality_tier,
    )

    print(
        "OI valid             :",
        snapshot.oi_valid,
    )

    print(
        "Funding valid        :",
        snapshot.funding_valid,
    )

    print(
        "Basis valid          :",
        snapshot.basis_valid,
    )

    print(
        "Taker flow valid     :",
        snapshot.taker_flow_valid,
    )

    print(
        "Spot × Perp valid    :",
        snapshot.spot_perp_valid,
    )

    print(
        "Liquidation valid    :",
        snapshot.liquidation_valid,
    )

    print(
        "Core valid           :",
        snapshot.derivatives_core_valid,
    )

    print(
        "Regime quality valid :",
        snapshot.derivatives_regime_quality_valid,
    )

    print()
    print("OUTPUT")
    print("-" * 100)

    print(
        output,
    )

    print()
    print("=" * 100)
    print(
        "STEP 8.11 DERIVATIVES SNAPSHOT: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
