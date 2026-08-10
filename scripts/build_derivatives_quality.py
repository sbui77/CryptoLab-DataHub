from __future__ import annotations

from cryptolab.pipelines.derivatives_quality import (
    build_and_save_derivatives_quality,
)


def main() -> None:
    df, output = (
        build_and_save_derivatives_quality(
            exchange="binance",
            symbol="BTCUSDT",
            price_timeframe="1h",
            derivatives_period="5m",
            funding_max_age_hours=12.0,
            spot_perp_min_valid_share=0.80,
        )
    )

    print()
    print("=" * 100)
    print("DERIVATIVES DATA QUALITY")
    print("=" * 100)

    print(
        "Rows   :",
        len(df),
    )

    print(
        "First  :",
        df["open_time"].min(),
    )

    print(
        "Last   :",
        df["open_time"].max(),
    )

    print(
        "Output :",
        output,
    )

    print()
    print("VALIDITY SUMMARY")
    print("-" * 100)

    masks = [
        "oi_valid",
        "funding_valid",
        "basis_valid",
        "taker_flow_valid",
        "spot_flow_valid",
        "spot_perp_valid",
        "liquidation_observed",
        "liquidation_valid",
        "derivatives_core_valid",
        "derivatives_regime_quality_valid",
    ]

    for column in masks:
        count = int(
            df[column]
            .fillna(False)
            .sum()
        )

        print(
            f"{column:40s}: "
            f"{count}/{len(df)}"
        )

    print()
    print("QUALITY TIER")
    print("-" * 100)

    print(
        df[
            "derivatives_quality_tier"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("LATEST QUALITY STATE")
    print("-" * 100)

    columns = [
        "open_time",
        "derivatives_regime",
        "oi_valid",
        "funding_valid",
        "basis_valid",
        "taker_flow_valid",
        "spot_perp_valid",
        "liquidation_observed",
        "liquidation_valid",
        "derivatives_regime_evidence_count",
        "derivatives_quality_score",
        "derivatives_quality_tier",
        "derivatives_regime_quality_valid",
    ]

    print(
        df[
            columns
        ]
        .tail(30)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "STEP 8.10 DERIVATIVES QUALITY: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
