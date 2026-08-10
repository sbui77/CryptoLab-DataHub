from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests


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
    Raised when Binance rate-limits or bans the current IP.
    """


def _utc_timestamp(
    value: pd.Timestamp,
) -> pd.Timestamp:
    """
    Normalize timestamp-like value to UTC.
    """

    result = pd.Timestamp(value)

    if result.tzinfo is None:
        return result.tz_localize(
            "UTC"
        )

    return result.tz_convert(
        "UTC"
    )


def _extract_error(
    payload: Any,
) -> tuple[int | None, str | None]:
    """
    Extract Binance code/msg from an error object.
    """

    if not isinstance(
        payload,
        dict,
    ):
        return None, None

    code = payload.get("code")
    msg = payload.get("msg")

    try:
        code_int = (
            int(code)
            if code is not None
            else None
        )
    except (
        TypeError,
        ValueError,
    ):
        code_int = None

    return (
        code_int,
        str(msg)
        if msg is not None
        else None,
    )


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> Any:
    """
    Execute Binance public REST request.

    Retry:
        network errors
        timeouts
        HTTP 5xx

    Fail immediately:
        HTTP 418
        HTTP 429
        Binance code -1003
        other Binance API errors
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

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            time.sleep(
                retry_backoff_seconds
                * attempt
            )

            continue

        try:
            payload = response.json()

        except ValueError as exc:
            if (
                500
                <= response.status_code
                <= 599
            ):
                last_error = exc

                if attempt >= max_retries:
                    break

                time.sleep(
                    retry_backoff_seconds
                    * attempt
                )

                continue

            raise BinanceTakerFlowError(
                "Binance returned non-JSON response. "
                f"status={response.status_code} "
                f"text={response.text[:500]}"
            ) from exc

        code, msg = _extract_error(
            payload
        )

        if (
            response.status_code
            in (
                418,
                429,
            )
            or code == -1003
        ):
            raise BinanceTakerFlowRateLimitError(
                "Binance rate limit / IP ban detected. "
                f"status={response.status_code} "
                f"code={code} "
                f"msg={msg}. "
                "Request was NOT retried."
            )

        if isinstance(
            payload,
            dict,
        ) and (
            code is not None
            or msg is not None
        ):
            raise BinanceTakerFlowError(
                "Binance API error. "
                f"status={response.status_code} "
                f"code={code} "
                f"msg={msg}"
            )

        if (
            500
            <= response.status_code
            <= 599
        ):
            last_error = BinanceTakerFlowError(
                "Binance server error. "
                f"status={response.status_code}"
            )

            if attempt >= max_retries:
                break

            time.sleep(
                retry_backoff_seconds
                * attempt
            )

            continue

        try:
            response.raise_for_status()

        except requests.HTTPError as exc:
            raise BinanceTakerFlowError(
                "Binance HTTP error. "
                f"status={response.status_code} "
                f"payload={payload}"
            ) from exc

        return payload

    raise BinanceTakerFlowError(
        "Binance taker-flow request failed "
        f"after {max_retries} attempts: "
        f"{last_error}"
    )


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

    rows = []

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
