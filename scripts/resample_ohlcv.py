from cryptolab.features.price import (
    resample_ohlcv,
)
from cryptolab.pipelines.curated_ohlcv import (
    save_curated_ohlcv,
)
from cryptolab.pipelines.ohlcv_storage import (
    read_ohlcv,
)
from cryptolab.quality.ohlcv import (
    validate_ohlcv,
)


df_1h = read_ohlcv(
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1h",
)

print(
    "1h rows:",
    len(df_1h),
)

for target in [
    "4h",
    "1d",
    "1w",
]:
    print()
    print(
        "=" * 60
    )

    print(
        "Resampling:",
        target,
    )

    df = resample_ohlcv(
        df_1h,
        target_timeframe=target,
    )

    validate_ohlcv(
        df
    )

    files = save_curated_ohlcv(
        df
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "First:",
        df["open_time"].min(),
    )

    print(
        "Last:",
        df["open_time"].max(),
    )

    print(
        "Files saved:",
        len(files),
    )
