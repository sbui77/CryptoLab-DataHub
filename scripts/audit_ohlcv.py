from cryptolab.pipelines.ohlcv_audit import (
    audit_ohlcv,
)


result = audit_ohlcv(
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1h",
)

print()
print("OHLCV AUDIT")
print("-" * 40)

print(
    "Exchange:",
    result.exchange,
)

print(
    "Symbol:",
    result.symbol,
)

print(
    "Timeframe:",
    result.timeframe,
)

print(
    "Rows:",
    result.rows,
)

print(
    "First:",
    result.first_open_time,
)

print(
    "Last:",
    result.last_open_time,
)

print(
    "Duplicates:",
    result.duplicate_count,
)

print(
    "Gaps:",
    result.gap_count,
)

print(
    "Valid:",
    result.valid,
)
