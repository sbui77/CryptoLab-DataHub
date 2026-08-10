from cryptolab.pipelines.derivatives_regime import (
    build_and_save_derivatives_regime,
)


def main() -> None:
    df, output = (
        build_and_save_derivatives_regime(
            exchange="binance",
            symbol="BTCUSDT",
            price_timeframe="1h",
            derivatives_period="5m",
            flow_threshold=0.10,
            liquidation_imbalance_threshold=0.50,
        )
    )

    print()
    print("=" * 100)
    print("DERIVATIVES REGIME")
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

    valid = df[
        df[
            "derivatives_regime_valid"
        ]
    ]

    print()
    print("VALIDITY")
    print("-" * 100)

    print(
        "Valid  :",
        len(valid),
    )

    print(
        "Invalid:",
        len(df)
        - len(valid),
    )

    print()
    print("REGIME DISTRIBUTION")
    print("-" * 100)

    if valid.empty:
        print(
            "No valid derivatives regime rows"
        )

    else:
        print(
            valid[
                "derivatives_regime"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print("LATEST REGIMES")
    print("-" * 100)

    columns = [
        "open_time",
        "close",

        "price_return_1h",
        "oi_quote_change_pct_1h",

        "spot_delta_pct_1h",
        "perp_delta_pct_1h",

        "funding_rate",
        "basis_rate",

        "long_liquidation_notional_1h",
        "short_liquidation_notional_1h",
        "liquidation_imbalance_1h",
        "liquidation_extreme",

        "price_oi_state",

        "derivatives_regime",
        "derivatives_regime_confidence",
        "derivatives_regime_valid",
    ]

    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[
            existing
        ]
        .tail(30)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print(
        "STEP 8.9 DERIVATIVES REGIME: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
