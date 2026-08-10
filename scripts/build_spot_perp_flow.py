from cryptolab.pipelines.spot_perp_flow import (
    build_and_save_spot_perp_flow,
)


def main() -> None:
    df, output = (
        build_and_save_spot_perp_flow(
            exchange="binance",
            symbol="BTCUSDT",
            spot_timeframe="1m",
            derivatives_period="5m",
            directional_threshold=0.10,
            leadership_ratio=1.50,
        )
    )

    print()
    print("=" * 100)
    print("SPOT × PERP FLOW INTERACTION")
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
        "Output :",
        output,
    )

    print()
    print("VALID ROWS")
    print("-" * 100)

    valid = df[
        df[
            "spot_perp_flow_valid"
        ]
    ]

    print(
        "Valid  :",
        len(valid),
    )

    print(
        "Invalid:",
        len(df) - len(valid),
    )

    print()
    print("STATE DISTRIBUTION")
    print("-" * 100)

    print(
        valid[
            "spot_perp_flow_state"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("LATEST CROSS-MARKET FLOW")
    print("-" * 100)

    columns = [
        "timestamp",

        "spot_quote_volume",
        "spot_delta_pct",

        "perp_total_volume",
        "perp_delta_pct",

        "spot_perp_delta_spread",
        "spot_perp_delta_agreement",

        "spot_perp_flow_state",
        "spot_perp_combined_strength",
        "spot_perp_flow_valid",
    ]

    print(
        df[
            columns
        ]
        .tail(30)
        .to_string(index=False)
    )

    print()
    print("=" * 100)
    print(
        "STEP 8.8 SPOT × PERP FLOW: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
