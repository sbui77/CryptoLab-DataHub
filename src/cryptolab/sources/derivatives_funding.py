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
    Execute Binance Futures request through the shared
    USD-M Futures HTTP client.
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
        raise BinanceFundingRateError(
            "Binance funding request failed: "
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

    rows: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            raise BinanceFundingRateError(
                "Funding history contains non-object row"
            )

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
                f"{sorted(missing)}"
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
            subset=["funding_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )
