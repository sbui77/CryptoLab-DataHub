from cryptolab.pipelines.price_oi_state import (
    build_and_save_price_oi_state,
)


def main() -> None:
    df, output = (
        build_and_save_price_oi_state(
            exchange="binance",
            symbol="BTCUSDT",
            price_timeframe="1h",
            derivatives_period="5m",
            price_threshold=0.001,
            oi_threshold=0.002,
        )
    )

    print()
    print("=" * 100)
    print("PRICE × OPEN INTEREST STATE")
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
    print("STATE DISTRIBUTION")
    print("-" * 100)

    print(
        df[
            "price_oi_state"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("LATEST STATES")
    print("-" * 100)

    columns = [
        "open_time",
        "close",

        "price_return_1h",

        "open_interest_quote",
        "oi_quote_change_pct_1h",

        "price_direction",
        "oi_direction",

        "price_oi_state",
        "price_oi_strength",
        "price_oi_state_valid",
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
        "STEP 8.7 PRICE × OI STATE: COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
