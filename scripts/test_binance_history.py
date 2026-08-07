from datetime import datetime, timezone

from cryptolab.quality.ohlcv import (
    find_time_gaps,
    validate_ohlcv,
)
from cryptolab.sources.binance import (
    BinanceSpotClient,
    fetch_historical_klines,
)


def to_milliseconds(
    value: datetime,
) -> int:
    return int(value.timestamp() * 1000)


start = datetime(
    2026,
    7,
    1,
    tzinfo=timezone.utc,
)

end = datetime(
    2026,
    7,
    8,
    tzinfo=timezone.utc,
)

client = BinanceSpotClient()

df = fetch_historical_klines(
    client=client,
    symbol="BTCUSDT",
    interval="1h",
    start_time=to_milliseconds(start),
    end_time=to_milliseconds(end),
)

validate_ohlcv(df)
gaps = find_time_gaps(
    df,
    expected_frequency="1h",
)

print("Missing candles:", len(gaps))

if not gaps.empty:
    print(gaps)

print(df.head())
print()
print(df.tail())
print()
print("Rows:", len(df))
print("First:", df["open_time"].min())
print("Last:", df["open_time"].max())
print("QUALITY CHECK: PASSED")

from cryptolab.pipelines.price_ohlcv import (
    save_ohlcv,
)

files = save_ohlcv(df)

print()
print("Saved files:")

for file in files:
    print(file)