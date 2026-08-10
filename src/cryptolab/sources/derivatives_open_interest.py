from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests


BINANCE_FUTURES_BASE_URL = (
    "https://fapi.binance.com"
)

OPEN_INTEREST_ENDPOINT = (
    "/fapi/v1/openInterest"
)

OPEN_INTEREST_HISTORY_ENDPOINT = (
    "/futures/data/openInterestHist"
)

MAX_HISTORY_LIMIT = 500


class BinanceOpenInterestError(RuntimeError):
    """Raised when Binance OI retrieval fails."""


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> Any:
    """
    Execute Binance Futures public REST request.
    """

    url = (
        BINANCE_FUTURES_BASE_URL
        + endpoint
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

            return response.json()

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            time.sleep(
                retry_backoff_seconds
                * attempt
            )

    raise BinanceOpenInterestError(
        "Binance Futures request failed "
        f"after {max_retries} attempts: "
        f"{last_error}"
    )


def fetch_current_open_interest(
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """
    Fetch current Binance USDⓈ-M Futures Open Interest.

    This endpoint returns openInterest in base-asset units.

    Output
    ------
    exchange
    market
    symbol
    timestamp
    open_interest_base
    """

    symbol = symbol.upper()

    payload = _request_json(
        OPEN_INTEREST_ENDPOINT,
        {
            "symbol": symbol,
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise BinanceOpenInterestError(
            "Unexpected current OI response"
        )

    required = {
        "symbol",
        "openInterest",
        "time",
    }

    missing = (
        required
        - set(payload)
    )

    if missing:
        raise BinanceOpenInterestError(
            f"Missing current OI fields: {missing}"
        )

    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": str(
                    payload["symbol"]
                ).upper(),
                "timestamp": pd.to_datetime(
                    payload["time"],
                    unit="ms",
                    utc=True,
                ),
                "open_interest_base": float(
                    payload["openInterest"]
                ),
            }
        ]
    )


def fetch_open_interest_history(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch historical Binance USDⓈ-M Open Interest Statistics.

    Output
    ------
    exchange
    market
    symbol
    period
    timestamp
    open_interest_base
    open_interest_quote
    """

    symbol = symbol.upper()

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    if limit > MAX_HISTORY_LIMIT:
        raise ValueError(
            f"limit cannot exceed {MAX_HISTORY_LIMIT}"
        )

    params: dict[str, Any] = {
        "symbol": symbol,
        "period": period,
        "limit": int(limit),
    }

    if start_time is not None:
        start = pd.Timestamp(
            start_time
        )

        if start.tzinfo is None:
            start = start.tz_localize(
                "UTC"
            )
        else:
            start = start.tz_convert(
                "UTC"
            )

        params["startTime"] = int(
            start.timestamp()
            * 1000
        )

    if end_time is not None:
        end = pd.Timestamp(
            end_time
        )

        if end.tzinfo is None:
            end = end.tz_localize(
                "UTC"
            )
        else:
            end = end.tz_convert(
                "UTC"
            )

        params["endTime"] = int(
            end.timestamp()
            * 1000
        )

    if (
        start_time is not None
        and end_time is not None
        and end <= start
    ):
        raise ValueError(
            "end_time must be after start_time"
        )

    payload = _request_json(
        OPEN_INTEREST_HISTORY_ENDPOINT,
        params,
    )

    if not isinstance(
        payload,
        list,
    ):
        raise BinanceOpenInterestError(
            "Unexpected historical OI response"
        )

    if not payload:
        return pd.DataFrame(
            columns=[
                "exchange",
                "market",
                "symbol",
                "period",
                "timestamp",
                "open_interest_base",
                "open_interest_quote",
            ]
        )

    rows = []

    for item in payload:
        rows.append(
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": str(
                    item["symbol"]
                ).upper(),
                "period": period,
                "timestamp": pd.to_datetime(
                    item["timestamp"],
                    unit="ms",
                    utc=True,
                ),
                "open_interest_base": float(
                    item[
                        "sumOpenInterest"
                    ]
                ),
                "open_interest_quote": float(
                    item[
                        "sumOpenInterestValue"
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return (
        result
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )
