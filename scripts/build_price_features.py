from cryptolab.pipelines.price_features import (
    build_and_save_price_features,
)
def main() -> None:
    df = build_and_save_price_features(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
    )
    print()
    print("PRICE FEATURES")
    print("-" * 100)
    print(
        "Rows :",
        len(df),
    )
    print(
        "First:",
        df["open_time"].min(),
    )
    print(
        "Last :",
        df["open_time"].max(),
    )
    columns = [
        "open_time",
        "close",
        "return_24h",
        "return_7d",
        "return_30d",
        "atr_14_pct",
        "rv_30d",
        "rv_30d_percentile_1y",
        "daily_vwap",
        "weekly_vwap",
        "monthly_vwap",
        "distance_to_daily_vwap_pct",
        "distance_to_weekly_vwap_pct",
        "distance_to_monthly_vwap_pct",
    ]
    print()
    print(
        df[columns]
        .tail(20)
        .to_string(
            index=False
        )
    )
if __name__ == "__main__":
    main()