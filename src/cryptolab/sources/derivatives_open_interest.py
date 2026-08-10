from __future__ import annotations

from typing import Any

import pandas as pd

from cryptolab.sources.binance_futures_http import (
    BinanceFuturesHTTPClient,
    BinanceFuturesHTTPError,
)


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
    Execute Binance Futures public REST request through
    the shared USD-M Futures HTTP client.

    Endpoint-specific modules no longer implement their own
    retry/rate-limit policy.
    """

    client = BinanceFuturesHTTPClient(
        base_url=BINANCE_FUTURES_BASE_URL,
        timeout_seconds=timeout,
        max_retries=max_retries,
        backoff_seconds=retry_backoff_seconds,
    )

    try:
        return client.get_json(
            endpoint,
            params=params,
        )

    except BinanceFuturesHTTPError as exc:
        raise BinanceOpenInterestError(
            "Binance Open Interest request failed: "
            f"{exc}"
        ) from exc


def _utc_timestamp(
    value: pd.Timestamp,
) -> pd.Timestamp:
    result = pd.Timestamp(
        value
    )

    if result.tzinfo is None:
        return result.tz_localize(
            "UTC"
        )

    return result.tz_convert(
        "UTC"
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

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    if start_time is not None:
        start = _utc_timestamp(
            start_time
        )

        params["startTime"] = int(
            start.timestamp()
            * 1000
        )

    if end_time is not None:
        end = _utc_timestamp(
            end_time
        )

        params["endTime"] = int(
            end.timestamp()
            * 1000
        )

    if (
        start is not None
        and end is not None
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

    columns = [
        "exchange",
        "market",
        "symbol",
        "period",
        "timestamp",
        "open_interest_base",
        "open_interest_quote",
    ]

    if not payload:
        return pd.DataFrame(
            columns=columns
        )

    rows: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            raise BinanceOpenInterestError(
                "Historical OI response contains "
                "non-object row"
            )

        required = {
            "symbol",
            "sumOpenInterest",
            "sumOpenInterestValue",
            "timestamp",
        }

        missing = (
            required
            - set(item)
        )

        if missing:
            raise BinanceOpenInterestError(
                "Missing historical OI fields: "
                f"{sorted(missing)}"
            )

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
        rows,
        columns=columns,
    )

    return (
        result
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )
