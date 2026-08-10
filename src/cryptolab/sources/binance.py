from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import requests
from cryptolab.config import (
    get_config_value,
    load_config,
)
from cryptolab.logger import get_logger
from cryptolab.utils.timeframes import (
    timeframe_timedelta,
)
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
    """
    Simple Binance Spot REST client for public market data.
    """
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
        """
        Execute a GET request with retry handling.
        """
        url = f"{self.base_url}{endpoint}"
        last_error: Exception | None = None
        for attempt in range(
            1,
            self.max_retries + 1,
        ):
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
                    "Binance request failed "
                    "attempt=%s/%s error=%s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(
                        self.retry_backoff
                        * attempt
                    )
        raise BinanceAPIError(
            "Binance API request failed "
            f"after {self.max_retries} attempts"
        ) from last_error
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        """
        Fetch Binance Spot klines.
        Parameters
        ----------
        symbol:
            Trading pair, e.g. BTCUSDT.
        interval:
            Binance timeframe, e.g. 1h, 4h, 1d.
        limit:
            Number of candles per request.
            Binance maximum is 1000.
        start_time:
            Optional start timestamp in milliseconds UTC.
        end_time:
            Optional end timestamp in milliseconds UTC.
        """
        if not 1 <= limit <= 1000:
            raise ValueError(
                "Binance kline limit must be "
                "between 1 and 1000"
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
            "symbol=%s interval=%s "
            "limit=%s start_time=%s "
            "end_time=%s",
            symbol,
            interval,
            limit,
            start_time,
            end_time,
        )
        data = self._get(
            self.kline_endpoint,
            params=params,
        )
        if not isinstance(data, list):
            raise BinanceAPIError(
                "Unexpected Binance "
                "kline response"
            )
        return data
def klines_to_dataframe(
    data: list[list[Any]],
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
) -> pd.DataFrame:
    """
    Convert Binance raw kline response into
    CryptoLab standardized OHLCV dataframe.
    """
    output_columns = [
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
    if not data:
        return pd.DataFrame(
            columns=output_columns
        )
    df = pd.DataFrame(
        data,
        columns=KLINE_COLUMNS,
    )
    df = df.drop(
        columns=["ignore"]
    )
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
        df[numeric_columns]
        .astype("float64")
    )
    df["trade_count"] = (
        df["trade_count"]
        .astype("int64")
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
    # -------------------------------------------------
    # Binance historical anomaly correction
    #
    # Some very old Binance klines may contain
    # close_time <= open_time.
    #
    # We do NOT relax the OHLCV quality rule.
    # Instead, for those anomalous records only,
    # reconstruct the expected close time from:
    #
    # open_time + timeframe - 1 millisecond
    # -------------------------------------------------
    expected_close_time = (
        df["open_time"]
        + timeframe_timedelta(timeframe)
        - pd.Timedelta(milliseconds=1)
    )
    invalid_close_time = (
        df["close_time"]
        <= df["open_time"]
    )
    if invalid_close_time.any():
        logger.warning(
            "Correcting %s invalid Binance "
            "close_time values "
            "symbol=%s timeframe=%s",
            int(invalid_close_time.sum()),
            symbol,
            timeframe,
        )
        invalid_rows = df.loc[
            invalid_close_time,
            [
                "open_time",
                "close_time",
            ],
        ]
        logger.warning(
            "Invalid Binance timestamp sample: %s",
            invalid_rows
            .head(5)
            .to_dict("records"),
        )
        df.loc[
            invalid_close_time,
            "close_time",
        ] = expected_close_time[
            invalid_close_time
        ]
    df["ingested_at"] = datetime.now(
        timezone.utc
    )
    return (
        df[output_columns]
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
    request_pause: float = 0.05,
) -> pd.DataFrame:
    """
    Fetch historical Binance klines using pagination.
    Parameters
    ----------
    client:
        BinanceSpotClient instance.
    symbol:
        Trading pair, e.g. BTCUSDT.
    interval:
        Timeframe, e.g. 1h.
    start_time:
        Starting timestamp in milliseconds UTC.
    end_time:
        Optional ending timestamp in milliseconds UTC.
    limit:
        Candles per request.
        Maximum supported by Binance is 1000.
    request_pause:
        Delay between consecutive requests.
    """
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
            logger.info(
                "No more Binance klines returned "
                "symbol=%s interval=%s",
                symbol,
                interval,
            )
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
            "Fetched batch rows=%s "
            "first=%s last=%s",
            len(df),
            df["open_time"].iloc[0],
            df["open_time"].iloc[-1],
        )
        if request_pause > 0:
            time.sleep(
                request_pause
            )
        last_open_ms = int(
            raw[-1][0]
        )
        next_start = (
            last_open_ms + 1
        )
        if next_start <= current_start:
            raise BinanceAPIError(
                "Pagination did not advance"
            )
        current_start = next_start
        if len(raw) < limit:
            break
        if (
            end_time is not None
            and current_start > end_time
        ):
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
            ],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return result