from __future__ import annotations

import pandas as pd

from cryptolab.config import (
    get_config_value,
    load_config,
)
from cryptolab.logger import get_logger
from cryptolab.pipelines.ohlcv_storage import (
    get_latest_open_time,
)
from cryptolab.pipelines.price_ohlcv import (
    save_ohlcv,
)
from cryptolab.quality.ohlcv import (
    remove_open_candles,
    validate_ohlcv,
)
from cryptolab.sources.binance import (
    BinanceSpotClient,
    fetch_historical_klines,
)
from cryptolab.utils.timeframes import (
    datetime_to_ms,
    timeframe_timedelta,
)
from cryptolab.pipelines.ohlcv_storage import (
    get_latest_open_time,
    read_ohlcv,
)

from cryptolab.quality.ohlcv import (
    find_ohlcv_gaps,
    remove_open_candles,
    validate_ohlcv,
)

logger = get_logger(__name__)


def _utc_timestamp(
    value: str | pd.Timestamp,
) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def backfill_ohlcv(
    symbol: str,
    timeframe: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    config = load_config()

    if start is None:
        start = get_config_value(
            config,
            "price.ohlcv.backfill_start",
            "2017-08-01T00:00:00Z",
        )

    start_ts = _utc_timestamp(start)

    if end is None:
        end_ts = pd.Timestamp.now(tz="UTC")
    else:
        end_ts = _utc_timestamp(end)

    logger.info(
        "Starting OHLCV backfill "
        "symbol=%s timeframe=%s "
        "start=%s end=%s",
        symbol,
        timeframe,
        start_ts,
        end_ts,
    )

    client = BinanceSpotClient()

    df = fetch_historical_klines(
        client=client,
        symbol=symbol,
        interval=timeframe,
        start_time=datetime_to_ms(start_ts),
        end_time=datetime_to_ms(end_ts),
        limit=1000,
    )

    if df.empty:
        logger.warning(
            "Backfill returned no data "
            "symbol=%s timeframe=%s",
            symbol,
            timeframe,
        )
        return df

    df = remove_open_candles(df)

    validate_ohlcv(df)

    save_ohlcv(df)

    logger.info(
        "OHLCV backfill completed "
        "symbol=%s timeframe=%s rows=%s "
        "first=%s last=%s",
        symbol,
        timeframe,
        len(df),
        df["open_time"].min(),
        df["open_time"].max(),
    )

    return df


def update_ohlcv(
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
) -> pd.DataFrame:
    latest = get_latest_open_time(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    if latest is None:
        logger.info(
            "No existing OHLCV found. "
            "Running initial backfill."
        )

        return backfill_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
        )

    next_open = (
        latest
        + timeframe_timedelta(timeframe)
    )

    now = pd.Timestamp.now(tz="UTC")

    if next_open >= now:
        logger.info(
            "OHLCV already current "
            "symbol=%s timeframe=%s",
            symbol,
            timeframe,
        )

        return pd.DataFrame()

    logger.info(
        "Incremental OHLCV update "
        "symbol=%s timeframe=%s from=%s",
        symbol,
        timeframe,
        next_open,
    )

    client = BinanceSpotClient()

    df = fetch_historical_klines(
        client=client,
        symbol=symbol,
        interval=timeframe,
        start_time=datetime_to_ms(next_open),
        end_time=datetime_to_ms(now),
        limit=1000,
    )

    if df.empty:
        logger.info(
            "No new OHLCV candles returned"
        )
        return df

    df = remove_open_candles(
        df,
        now=now,
    )

    if df.empty:
        logger.info(
            "Only an open candle was returned; "
            "nothing saved"
        )
        return df

    validate_ohlcv(df)

    save_ohlcv(df)

    logger.info(
        "Incremental update completed "
        "rows=%s first=%s last=%s",
        len(df),
        df["open_time"].min(),
        df["open_time"].max(),
    )

    return df

def repair_ohlcv_gaps(
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
) -> int:
    df = read_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    if df.empty:
        logger.warning(
            "No dataset available for gap repair"
        )
        return 0

    gaps = find_ohlcv_gaps(
        df,
        timeframe,
    )

    if gaps.empty:
        logger.info(
            "No OHLCV gaps detected"
        )
        return 0

    logger.warning(
        "Detected %s missing candles",
        len(gaps),
    )

    client = BinanceSpotClient()

    repaired = 0

    delta = timeframe_timedelta(
        timeframe
    )

    for missing_time in gaps[
        "missing_open_time"
    ]:
        raw_df = fetch_historical_klines(
            client=client,
            symbol=symbol,
            interval=timeframe,
            start_time=datetime_to_ms(
                missing_time
            ),
            end_time=datetime_to_ms(
                missing_time + delta
            ),
            limit=2,
        )

        if raw_df.empty:
            logger.warning(
                "Unable to repair gap at %s",
                missing_time,
            )
            continue

        exact = raw_df[
            raw_df["open_time"]
            == missing_time
        ].copy()

        if exact.empty:
            logger.warning(
                "Binance returned no candle "
                "for gap=%s",
                missing_time,
            )
            continue

        exact = remove_open_candles(
            exact
        )

        if exact.empty:
            continue

        validate_ohlcv(exact)

        save_ohlcv(exact)

        repaired += len(exact)

        logger.info(
            "Repaired OHLCV gap=%s",
            missing_time,
        )

    logger.info(
        "Gap repair finished repaired=%s",
        repaired,
    )

    return repaired
