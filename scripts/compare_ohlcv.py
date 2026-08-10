from __future__ import annotations
from pathlib import Path
import pandas as pd
from cryptolab.pipelines.ohlcv_storage import (
    read_ohlcv,
)
CURATED_ROOT = Path(
    "data/curated/price/ohlcv_resampled"
)
def read_curated_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Read curated resampled OHLCV dataset
    from monthly Parquet partitions.
    """
    root = (
        CURATED_ROOT
        / f"exchange={exchange}"
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
    )
    files = sorted(
        root.glob(
            "year=*/month=*/data.parquet"
        )
    )
    if not files:
        return pd.DataFrame()
    frames = [
        pd.read_parquet(file)
        for file in files
    ]
    df = pd.concat(
        frames,
        ignore_index=True,
    )
    return (
        df
        .drop_duplicates(
            subset=[
                "exchange",
                "symbol",
                "timeframe",
                "open_time",
            ],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )
def compare_timeframe(
    exchange: str,
    symbol: str,
    timeframe: str,
) -> None:
    native = read_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    curated = read_curated_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    print()
    print("=" * 80)
    print(
        f"TIMEFRAME: {timeframe}"
    )
    print("=" * 80)
    print(
        "Native rows :",
        len(native),
    )
    print(
        "Curated rows:",
        len(curated),
    )
    if native.empty:
        print(
            "Native dataset is empty"
        )
        return
    if curated.empty:
        print(
            "Curated dataset is empty"
        )
        return
    print(
        "Native first:",
        native["open_time"].min(),
    )
    print(
        "Native last :",
        native["open_time"].max(),
    )
    print(
        "Curated first:",
        curated["open_time"].min(),
    )
    print(
        "Curated last :",
        curated["open_time"].max(),
    )
    merged = native.merge(
        curated,
        on="open_time",
        how="inner",
        suffixes=(
            "_native",
            "_curated",
        ),
    )
    print(
        "Matched rows:",
        len(merged),
    )
    if merged.empty:
        print(
            "No matching open_time values"
        )
        return
    native_only = native[
        ~native["open_time"].isin(
            curated["open_time"]
        )
    ]
    curated_only = curated[
        ~curated["open_time"].isin(
            native["open_time"]
        )
    ]
    print(
        "Native-only rows :",
        len(native_only),
    )
    print(
        "Curated-only rows:",
        len(curated_only),
    )
    print()
    print("VALUE DIFFERENCES")
    print("-" * 80)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        native_column = (
            f"{column}_native"
        )
        curated_column = (
            f"{column}_curated"
        )
        if (
            native_column
            not in merged.columns
            or curated_column
            not in merged.columns
        ):
            continue
        diff = (
            merged[native_column]
            - merged[curated_column]
        ).abs()
        nonzero_count = int(
            (diff > 0).sum()
        )
        print(
            f"{column:28s}"
            f" max_abs_diff={diff.max():.12g}"
            f" nonzero={nonzero_count}"
        )
    print()
    print("TIMESTAMP DIFFERENCES")
    print("-" * 80)
    if (
        "close_time_native"
        in merged.columns
        and "close_time_curated"
        in merged.columns
    ):
        close_time_diff = (
            merged["close_time_native"]
            != merged["close_time_curated"]
        )
        print(
            "close_time mismatches:",
            int(close_time_diff.sum()),
        )
    if not native_only.empty:
        print()
        print(
            "First native-only open_times:"
        )
        print(
            native_only[
                ["open_time"]
            ]
            .head(10)
            .to_string(index=False)
        )
    if not curated_only.empty:
        print()
        print(
            "First curated-only open_times:"
        )
        print(
            curated_only[
                ["open_time"]
            ]
            .head(10)
            .to_string(index=False)
        )
def main() -> None:
    exchange = "binance"
    symbol = "BTCUSDT"
    for timeframe in [
        "4h",
        "1d",
        "1w",
    ]:
        compare_timeframe(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
        )
if __name__ == "__main__":
    main()
