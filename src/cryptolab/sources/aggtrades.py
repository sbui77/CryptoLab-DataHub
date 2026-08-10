from __future__ import annotations
import time
from typing import Any
import pandas as pd
import requests
from cryptolab.schemas.aggtrades import (
    normalize_aggtrades,
)
class BinanceAggTradesError(RuntimeError):
    """Raised when Binance aggTrades retrieval fails."""
BINANCE_MARKET_DATA_BASE_URL = (
    "https://data-api.binance.vision"
)
AGGTRADES_ENDPOINT = (
    "/api/v3/aggTrades"
)
MAX_LIMIT = 1000
def _request_aggtrades(
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Execute a Binance aggTrades REST request with retry logic.
    """
    url = (
        BINANCE_MARKET_DATA_BASE_URL
        + AGGTRADES_ENDPOINT
    )
    last_error: Exception | None = None
    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(
                payload,
                list,
            ):
                raise BinanceAggTradesError(
                    "Unexpected Binance aggTrades response"
                )
            return payload
        except (
            requests.RequestException,
            ValueError,
            BinanceAggTradesError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = (
                retry_backoff_seconds
                * attempt
            )
            time.sleep(
                sleep_seconds
            )
    raise BinanceAggTradesError(
        "Binance aggTrades request failed "
        f"after {max_retries} attempts: "
        f"{last_error}"
    )
def _parse_aggtrades(
    payload: list[dict[str, Any]],
    symbol: str,
) -> pd.DataFrame:
    """
    Parse Binance aggTrades JSON into canonical CryptoLab schema.
    Binance fields:
        a = aggregate trade id
        p = price
        q = quantity
        f = first trade id
        l = last trade id
        T = trade time
        m = buyer is maker
    """
    if not payload:
        return normalize_aggtrades(
            pd.DataFrame(
                columns=[
                    "exchange",
                    "symbol",
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "trade_time",
                    "buyer_is_maker",
                ]
            )
        )
    rows = []
    for item in payload:
        rows.append(
            {
                "exchange": "binance",
                "symbol": symbol,
                "agg_trade_id": item["a"],
                "price": item["p"],
                "quantity": item["q"],
                "first_trade_id": item["f"],
                "last_trade_id": item["l"],
                "trade_time": pd.to_datetime(
                    item["T"],
                    unit="ms",
                    utc=True,
                ),
                "buyer_is_maker": item["m"],
            }
        )
    return normalize_aggtrades(
        pd.DataFrame(rows)
    )
def fetch_recent_aggtrades(
    symbol: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch most recent aggregate trades.
    Maximum limit is 1000.
    """
    symbol = symbol.upper()
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )
    if limit > MAX_LIMIT:
        raise ValueError(
            f"limit cannot exceed {MAX_LIMIT}"
        )
    payload = _request_aggtrades(
        {
            "symbol": symbol,
            "limit": limit,
        }
    )
    return _parse_aggtrades(
        payload,
        symbol=symbol,
    )
def fetch_aggtrades_from_id(
    symbol: str,
    from_id: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch aggregate trades starting from agg_trade_id.
    This is the preferred primitive for deterministic
    forward pagination after an initial starting point.
    """
    symbol = symbol.upper()
    if from_id < 0:
        raise ValueError(
            "from_id cannot be negative"
        )
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )
    if limit > MAX_LIMIT:
        raise ValueError(
            f"limit cannot exceed {MAX_LIMIT}"
        )
    payload = _request_aggtrades(
        {
            "symbol": symbol,
            "fromId": int(from_id),
            "limit": limit,
        }
    )
    return _parse_aggtrades(
        payload,
        symbol=symbol,
    )
def fetch_aggtrades_by_time(
    symbol: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch aggregate trades within a time window.
    Timestamps are converted to UTC milliseconds.
    This function is best used to locate a starting aggTrade ID.
    Once an ID is known, forward pagination should normally use
    fetch_aggtrades_from_id().
    """
    symbol = symbol.upper()
    start_time = pd.Timestamp(
        start_time
    )
    end_time = pd.Timestamp(
        end_time
    )
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize(
            "UTC"
        )
    else:
        start_time = start_time.tz_convert(
            "UTC"
        )
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize(
            "UTC"
        )
    else:
        end_time = end_time.tz_convert(
            "UTC"
        )
    if end_time <= start_time:
        raise ValueError(
            "end_time must be after start_time"
        )
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )
    if limit > MAX_LIMIT:
        raise ValueError(
            f"limit cannot exceed {MAX_LIMIT}"
        )
    payload = _request_aggtrades(
        {
            "symbol": symbol,
            "startTime": int(
                start_time.timestamp()
                * 1000
            ),
            "endTime": int(
                end_time.timestamp()
                * 1000
            ),
            "limit": limit,
        }
    )
    return _parse_aggtrades(
        payload,
        symbol=symbol,
    )
