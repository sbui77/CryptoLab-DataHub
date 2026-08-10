from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests


BINANCE_FUTURES_BASE_URL = (
    "https://fapi.binance.com"
)

FUNDING_HISTORY_ENDPOINT = (
    "/fapi/v1/fundingRate"
)

MAX_FUNDING_LIMIT = 1000


class BinanceFundingRateError(RuntimeError):
    """Raised when Binance funding-rate retrieval fails."""


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

    raise BinanceFundingRateError(
        "Binance funding request failed "
        f"after {max_retries} attempts: "
        f"{last_error}"
    )


def fetch_funding_rate_history(
    symbol: str = "BTCUSDT",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch Binance USDⓈ-M perpetual funding history.

    Canonical output
    ----------------
    exchange
    market
    symbol
    funding_time
    funding_rate
    mark_price
    rate_type

    Binance returns rows in ascending funding-time order.
    """

    symbol = symbol.upper()

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    if limit > MAX_FUNDING_LIMIT:
        raise ValueError(
            f"limit cannot exceed {MAX_FUNDING_LIMIT}"
        )

    params: dict[str, Any] = {
        "symbol": symbol,
        "limit": int(limit),
    }

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

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
        start is not None
        and end is not None
        and end < start
    ):
        raise ValueError(
            "end_time cannot be before start_time"
        )

    payload = _request_json(
        FUNDING_HISTORY_ENDPOINT,
        params,
    )

    if not isinstance(
        payload,
        list,
    ):
        raise BinanceFundingRateError(
            "Unexpected funding history response"
        )

    columns = [
        "exchange",
        "market",
        "symbol",
        "funding_time",
        "funding_rate",
        "mark_price",
        "rate_type",
    ]

    if not payload:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    for item in payload:
        required = {
            "symbol",
            "fundingRate",
            "fundingTime",
        }

        missing = (
            required
            - set(item)
        )

        if missing:
            raise BinanceFundingRateError(
                "Missing funding fields: "
                f"{missing}"
            )

        mark_price_raw = item.get(
            "markPrice"
        )

        rate_type_raw = item.get(
            "rateType",
            "Regular",
        )

        rows.append(
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": str(
                    item["symbol"]
                ).upper(),
                "funding_time": pd.to_datetime(
                    item["fundingTime"],
                    unit="ms",
                    utc=True,
                ),
                "funding_rate": float(
                    item["fundingRate"]
                ),
                "mark_price": (
                    float(mark_price_raw)
                    if mark_price_raw
                    not in (
                        None,
                        "",
                    )
                    else float("nan")
                ),
                "rate_type": str(
                    rate_type_raw
                ),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=columns,
    )

    return (
        result
        .sort_values("funding_time")
        .drop_duplicates(
            subset=[
                "funding_time",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )
