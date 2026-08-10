from cryptolab.pipelines.derivatives_features import (
    build_and_save_derivatives_features,
)


def main() -> None:
    df, output = (
        build_and_save_derivatives_features(
            exchange="binance",
            symbol="BTCUSDT",
            period="5m",
        )
    )

    print()
    print("=" * 100)
    print("DERIVATIVES FEATURE DATASET")
    print("=" * 100)

    print(
        "Rows   :",
        len(df),
    )

    print(
        "First  :",
        df["timestamp"].min(),
    )

    print(
        "Last   :",
        df["timestamp"].max(),
    )

    print(
        "Columns:",
        len(df.columns),
    )

    print(
        "Output :",
        output,
    )

    print()
    print("LATEST DERIVATIVES STATE")
    print("-" * 100)

    columns = [
        "timestamp",

        "open_interest_quote",
        "oi_quote_change_pct",
        "oi_quote_change_pct_1h",
        "oi_quote_zscore_24h",

        "funding_rate",
        "funding_rate_bps",
        "hours_since_funding",

        "basis_rate",
        "basis_bps",
        "basis_zscore_24h",

        "futures_taker_delta_pct",
        "futures_taker_delta_pct_1h",

        "long_liquidation_notional",
        "short_liquidation_notional",
        "liquidation_imbalance",
        "liquidation_notional_1h",
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
        .tail(20)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print(
        "STEP 8.6 DERIVATIVES FEATURES: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
