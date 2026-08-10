from cryptolab.pipelines.ohlcv_service import (
    update_ohlcv,
)


df = update_ohlcv(
    symbol="BTCUSDT",
    timeframe="1h",
)

print()
print("New rows:", len(df))

if not df.empty:
    print(
        "First:",
        df["open_time"].min(),
    )

    print(
        "Last:",
        df["open_time"].max(),
    )
