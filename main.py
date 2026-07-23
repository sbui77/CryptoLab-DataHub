from sources.binance import BinanceClient

client = BinanceClient()

df = client.download(
    symbol="BTCUSDT",
    interval="1h",
    start_time=1502928000000,
)

print(df.head())