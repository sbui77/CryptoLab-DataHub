from __future__ import annotations

from typing import Any

import pandas as pd

from cryptolab.sources.binance_futures_http import (
    BinanceFuturesHTTPClient,
    BinanceFuturesHTTPError,
    BinanceFuturesIPBanError,
    BinanceFuturesRateLimitError,
)


BINANCE_FUTURES_BASE_URL = (
    "https://fapi.binance.com"
)

TAKER_FLOW_ENDPOINT = (
    "/futures/data/takerlongshortRatio"
)

MAX_TAKER_FLOW_LIMIT = 500


class BinanceTakerFlowError(RuntimeError):
    """Raised when Binance futures taker-flow retrieval fails."""


class BinanceTakerFlowRateLimitError(
    BinanceTakerFlowError
):
    """
    Raised when shared HTTP layer reports an exhausted
    rate limit or Binance IP ban.
    """


def _utc_timestamp(
    value: pd.Timestamp,
) -> pd.Timestamp:
    """
    Normalize timestamp-like value to UTC.
    """

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

    except (
        BinanceFuturesRateLimitError,
        BinanceFuturesIPBanError,
    ) as exc:
        raise BinanceTakerFlowRateLimitError(
            "Binance taker-flow rate-limit/IP-ban error: "
            f"{exc}"
        ) from exc

    except BinanceFuturesHTTPError as exc:
        raise BinanceTakerFlowError(
            "Binance taker-flow request failed: "
            f"{exc}"
        ) from exc


def fetch_taker_flow_history(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch Binance USDⓈ-M futures Taker Buy/Sell Volume.

    Canonical output
    ----------------
    exchange
    market
    symbol
    period
    timestamp
    buy_volume
    sell_volume
    buy_sell_ratio
    """

    symbol = symbol.upper()

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    if limit > MAX_TAKER_FLOW_LIMIT:
        raise ValueError(
            f"limit cannot exceed "
            f"{MAX_TAKER_FLOW_LIMIT}"
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
        and end < start
    ):
        raise ValueError(
            "end_time cannot be before start_time"
        )

    payload = _request_json(
        TAKER_FLOW_ENDPOINT,
        params,
    )

    if not isinstance(
        payload,
        list,
    ):
        raise BinanceTakerFlowError(
            "Unexpected Binance taker-flow "
            f"response type: {type(payload).__name__}"
        )

    columns = [
        "exchange",
        "market",
        "symbol",
        "period",
        "timestamp",
        "buy_volume",
        "sell_volume",
        "buy_sell_ratio",
    ]

    if not payload:
        return pd.DataFrame(
            columns=columns
        )

    rows: list[
        dict[str, Any]
    ] = []

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            raise BinanceTakerFlowError(
                "Taker-flow response contains "
                "non-object row"
            )

        required = {
            "buySellRatio",
            "buyVol",
            "sellVol",
            "timestamp",
        }

        missing = (
            required
            - set(item)
        )

        if missing:
            raise BinanceTakerFlowError(
                "Missing taker-flow fields: "
                f"{sorted(missing)}"
            )

        rows.append(
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": symbol,
                "period": period,
                "timestamp": pd.to_datetime(
                    item["timestamp"],
                    unit="ms",
                    utc=True,
                ),
                "buy_volume": float(
                    item["buyVol"]
                ),
                "sell_volume": float(
                    item["sellVol"]
                ),
                "buy_sell_ratio": float(
                    item["buySellRatio"]
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
