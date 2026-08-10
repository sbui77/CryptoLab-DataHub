import pandas as pd

from cryptolab.pipelines.ohlcv_storage import read_ohlcv
from cryptolab.pipelines.price_ohlcv import save_ohlcv
from cryptolab.quality.ohlcv import (
    find_ohlcv_gaps,
    validate_ohlcv,
)


SYMBOL = "BTCUSDT"
EXCHANGE = "binance"


# --------------------------------------------------
# Load canonical 1h data
# --------------------------------------------------

df_1h = read_ohlcv(
    exchange=EXCHANGE,
    symbol=SYMBOL,
    timeframe="1h",
)

if df_1h.empty:
    raise RuntimeError(
        "1h dataset is empty"
    )


# --------------------------------------------------
# Load native Binance 4h data
# --------------------------------------------------

df_4h = read_ohlcv(
    exchange=EXCHANGE,
    symbol=SYMBOL,
    timeframe="4h",
)

if df_4h.empty:
    raise RuntimeError(
        "4h dataset is empty"
    )


# --------------------------------------------------
# Detect missing 4h candles
# --------------------------------------------------

gaps = find_ohlcv_gaps(
    df_4h,
    timeframe="4h",
)

print(
    f"Detected 4h gaps: {len(gaps)}"
)

if gaps.empty:
    print("No repair required")
    raise SystemExit(0)


# --------------------------------------------------
# Reconstruct each missing 4h candle
# from four 1h candles
# --------------------------------------------------

reconstructed = []

for missing_time in gaps[
    "missing_open_time"
]:

    end_time = (
        missing_time
        + pd.Timedelta(hours=4)
    )

    source = df_1h[
        (df_1h["open_time"] >= missing_time)
        & (df_1h["open_time"] < end_time)
    ].copy()

    if len(source) != 4:
        print(
            "SKIP:",
            missing_time,
            "1h candles found:",
            len(source),
        )
        continue

    source = source.sort_values(
        "open_time"
    )

    expected_times = pd.date_range(
        start=missing_time,
        periods=4,
        freq="1h",
        tz="UTC",
    )

    actual_times = pd.DatetimeIndex(
        source["open_time"]
    )

    if not actual_times.equals(
        expected_times
    ):
        print(
            "SKIP non-contiguous:",
            missing_time,
        )
        continue

    row = {
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
        "timeframe": "4h",

        "open_time": missing_time,

        "open": source.iloc[0]["open"],
        "high": source["high"].max(),
        "low": source["low"].min(),
        "close": source.iloc[-1]["close"],

        "base_volume":
            source["base_volume"].sum(),

        "quote_volume":
            source["quote_volume"].sum(),

        "close_time":
            source.iloc[-1]["close_time"],

        "trade_count":
            source["trade_count"].sum(),

        "taker_buy_base_volume":
            source[
                "taker_buy_base_volume"
            ].sum(),

        "taker_buy_quote_volume":
            source[
                "taker_buy_quote_volume"
            ].sum(),

        "ingested_at":
            pd.Timestamp.now(tz="UTC"),
    }

    reconstructed.append(row)

    print(
        "RECONSTRUCTED:",
        missing_time,
    )


# --------------------------------------------------
# Save reconstructed candles
# --------------------------------------------------

if not reconstructed:
    print(
        "No candles could be reconstructed"
    )
    raise SystemExit(1)


repair_df = pd.DataFrame(
    reconstructed
)

repair_df["trade_count"] = (
    repair_df["trade_count"]
    .astype("int64")
)

validate_ohlcv(
    repair_df
)

save_ohlcv(
    repair_df
)

print()
print(
    f"Saved reconstructed candles: "
    f"{len(repair_df)}"
)
