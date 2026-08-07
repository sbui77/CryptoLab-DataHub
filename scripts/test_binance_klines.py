from cryptolab.quality.ohlcv import validate_ohlcv
from cryptolab.sources.binance import (
    BinanceSpotClient,
    klines_to_dataframe,
)


client = BinanceSpotClient()

raw = client.get_klines(
    symbol="BTCUSDT",
    interval="1h",
    limit=1000,
)

df = klines_to_dataframe(
    raw,
    symbol="BTCUSDT",
    timeframe="1h",
)

validate_ohlcv(df)

print(df)
print()
print(f"Rows: {len(df)}")
print("QUALITY CHECK: PASSED")