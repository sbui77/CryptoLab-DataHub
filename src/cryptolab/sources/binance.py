from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from cryptolab.config import get_config_value, load_config
from cryptolab.logger import get_logger


logger = get_logger(__name__)


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class BinanceAPIError(RuntimeError):
    """Raised when a Binance API request fails."""


class BinanceSpotClient:
    def __init__(self) -> None:
        self.config = load_config()

        self.base_url = get_config_value(
            self.config,
            "binance.spot.base_url",
            "https://api.binance.com",
        )

        self.kline_endpoint = get_config_value(
            self.config,
            "binance.spot.endpoints.klines",
            "/api/v3/klines",
        )

        self.timeout = int(
            get_config_value(
                self.config,
                "binance.spot.request.timeout_seconds",
                30,
            )
        )

        self.max_retries = int(
            get_config_value(
                self.config,
                "binance.spot.request.max_retries",
                5,
            )
        )

        self.retry_backoff = float(
            get_config_value(
                self.config,
                "binance.spot.request.retry_backoff_seconds",
                2,
            )
        )

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> Any:
        url = f"{self.base_url}{endpoint}"

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                return response.json()

            except (
                requests.RequestException,
                ValueError,
            ) as exc:
                last_error = exc

                logger.warning(
                    "Binance request failed attempt=%s/%s error=%s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries:
                    time.sleep(
                        self.retry_backoff * attempt
                    )

        raise BinanceAPIError(
            f"Binance API request failed after "
            f"{self.max_retries} attempts"
        ) from last_error

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError(
                "Binance kline limit must be between 1 and 1000"
            )

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }

        if start_time is not None:
            params["startTime"] = start_time

        if end_time is not None:
            params["endTime"] = end_time

        logger.info(
            "Fetching Binance klines "
            "symbol=%s interval=%s limit=%s",
            symbol,
            interval,
            limit,
        )

        data = self._get(
            self.kline_endpoint,
            params=params,
        )

        if not isinstance(data, list):
            raise BinanceAPIError(
                "Unexpected Binance kline response"
            )

        return data


def klines_to_dataframe(
    data: list[list[Any]],
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
) -> pd.DataFrame:
    if not data:
        return pd.DataFrame(
            columns=[
                "exchange",
                "symbol",
                "timeframe",
                *KLINE_COLUMNS[:-1],
                "ingested_at",
            ]
        )

    df = pd.DataFrame(
        data,
        columns=KLINE_COLUMNS,
    )

    df = df.drop(columns=["ignore"])

    df["exchange"] = exchange
    df["symbol"] = symbol.upper()
    df["timeframe"] = timeframe

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    df[numeric_columns] = (
        df[numeric_columns].astype("float64")
    )

    df["trade_count"] = df["trade_count"].astype(
        "int64"
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    df["ingested_at"] = datetime.now(
        timezone.utc
    )

    columns = [
        "exchange",
        "symbol",
        "timeframe",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "close_time",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ingested_at",
    ]

    return (
        df[columns]
        .sort_values("open_time")
        .reset_index(drop=True)
    )
def fetch_historical_klines(
    client: BinanceSpotClient,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    current_start = start_time

    while True:
        raw = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=current_start,
            end_time=end_time,
        )

        if not raw:
            break

        df = klines_to_dataframe(
            raw,
            symbol=symbol,
            timeframe=interval,
        )

        if df.empty:
            break

        frames.append(df)

        logger.info(
            "Fetched batch rows=%s first=%s last=%s",
            len(df),
            df["open_time"].iloc[0],
            df["open_time"].iloc[-1],
        )

        last_open_ms = int(raw[-1][0])

        next_start = last_open_ms + 1

        if next_start <= current_start:
            raise BinanceAPIError(
                "Pagination did not advance"
            )

        current_start = next_start

        if len(raw) < limit:
            break

        if end_time is not None:
            if current_start > end_time:
                break

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
        .drop_duplicates(
            subset=[
                "exchange",
                "symbol",
                "timeframe",
                "open_time",
            ]
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    return result