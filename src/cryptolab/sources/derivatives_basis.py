from __future__ import annotations
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
import pandas as pd
import requests
BINANCE_FUTURES_BASE_URL = (
    "https://fapi.binance.com"
)
BASIS_ENDPOINT = (
    "/futures/data/basis"
)
MAX_BASIS_LIMIT = 500
class BinanceBasisError(RuntimeError):
    """Raised when Binance basis retrieval fails."""
class BinanceBasisRateLimitError(
    BinanceBasisError
):
    """
    Raised when Binance rate-limits or bans the current IP.
    This error must not be retried immediately.
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
def _optional_float(
    value: Any,
) -> float:
    """
    Convert optional numeric field to float.
    Empty string and None become NaN.
    """
    if value in (
        None,
        "",
    ):
        return float("nan")
    return float(value)
def _parse_retry_after_seconds(
    response: requests.Response,
) -> float | None:
    """
    Parse Retry-After header when present.
    Retry-After may be:
        - integer seconds
        - HTTP date
    Returns seconds from now, or None.
    """
    raw = response.headers.get(
        "Retry-After"
    )
    if not raw:
        return None
    raw = raw.strip()
    # --------------------------------------------------------
    # Integer seconds
    # --------------------------------------------------------
    try:
        seconds = float(raw)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass
    # --------------------------------------------------------
    # HTTP date
    # --------------------------------------------------------
    try:
        retry_dt = parsedate_to_datetime(
            raw
        )
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(
                tzinfo=timezone.utc
            )
        now = datetime.now(
            timezone.utc
        )
        seconds = (
            retry_dt
            - now
        ).total_seconds()
        return max(
            0.0,
            seconds,
        )
    except Exception:
        return None
def _extract_binance_error(
    payload: Any,
) -> tuple[int | None, str | None]:
    """
    Extract Binance code/msg from a JSON error object.
    """
    if not isinstance(
        payload,
        dict,
    ):
        return (
            None,
            None,
        )
    code = payload.get(
        "code"
    )
    msg = payload.get(
        "msg"
    )
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
    msg_str = (
        str(msg)
        if msg is not None
        else None
    )
    return (
        code_int,
        msg_str,
    )
def _is_rate_limit_response(
    response: requests.Response,
    payload: Any,
) -> bool:
    """
    Detect Binance rate-limit / IP-ban responses.
    HTTP:
        429 = rate limited
        418 = IP auto-banned
    Binance JSON:
        code = -1003
    """
    if response.status_code in (
        418,
        429,
    ):
        return True
    code, _ = (
        _extract_binance_error(
            payload
        )
    )
    return code == -1003
def _format_rate_limit_error(
    response: requests.Response,
    payload: Any,
) -> str:
    """
    Build informative rate-limit error message.
    """
    code, msg = (
        _extract_binance_error(
            payload
        )
    )
    retry_after = (
        _parse_retry_after_seconds(
            response
        )
    )
    parts = [
        "Binance rate limit / IP ban detected.",
        f"status={response.status_code}",
    ]
    if code is not None:
        parts.append(
            f"code={code}"
        )
    if msg:
        parts.append(
            f"msg={msg}"
        )
    if retry_after is not None:
        parts.append(
            "retry_after_seconds="
            f"{retry_after:.0f}"
        )
    parts.append(
        "Request was NOT retried."
    )
    return " ".join(parts)
def _request_json(
    endpoint: str,
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> Any:
    """
    Execute Binance USDⓈ-M Futures public REST request.
    Retry policy
    ------------
    Retry:
        - connection errors
        - timeouts
        - HTTP 5xx
    Fail immediately:
        - HTTP 418
        - HTTP 429
        - Binance code -1003
        - other Binance API errors
        - malformed successful responses
    Important
    ---------
    When Binance rate-limits or bans an IP, immediate retries
    are harmful. Therefore rate-limit responses fail fast.
    """
    if max_retries <= 0:
        raise ValueError(
            "max_retries must be greater than zero"
        )
    if retry_backoff_seconds < 0:
        raise ValueError(
            "retry_backoff_seconds cannot be negative"
        )
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
            sleep_seconds = (
                retry_backoff_seconds
                * attempt
            )
            if sleep_seconds > 0:
                time.sleep(
                    sleep_seconds
                )
            continue
        # ====================================================
        # JSON PARSING
        # ====================================================
        try:
            payload = response.json()
        except ValueError as exc:
            # Server-side transient failures can occasionally
            # return HTML/plain text. Retry only for 5xx.
            if (
                500
                <= response.status_code
                <= 599
            ):
                last_error = (
                    BinanceBasisError(
                        "Binance returned non-JSON "
                        "server response. "
                        f"status={response.status_code} "
                        f"text={response.text[:500]}"
                    )
                )
                if attempt >= max_retries:
                    break
                sleep_seconds = (
                    retry_backoff_seconds
                    * attempt
                )
                if sleep_seconds > 0:
                    time.sleep(
                        sleep_seconds
                    )
                continue
            raise BinanceBasisError(
                "Binance returned non-JSON response. "
                f"status={response.status_code} "
                f"text={response.text[:500]}"
            ) from exc
        # ====================================================
        # RATE LIMIT / IP BAN
        # ====================================================
        if _is_rate_limit_response(
            response,
            payload,
        ):
            raise BinanceBasisRateLimitError(
                _format_rate_limit_error(
                    response,
                    payload,
                )
            )
        # ====================================================
        # BINANCE API ERROR OBJECT
        # ====================================================
        if isinstance(
            payload,
            dict,
        ):
            code, msg = (
                _extract_binance_error(
                    payload
                )
            )
            if (
                code is not None
                or msg is not None
            ):
                raise BinanceBasisError(
                    "Binance API error. "
                    f"status={response.status_code} "
                    f"code={code} "
                    f"msg={msg}"
                )
        # ====================================================
        # SERVER ERRORS
        # ====================================================
        if (
            500
            <= response.status_code
            <= 599
        ):
            last_error = (
                BinanceBasisError(
                    "Binance server error. "
                    f"status={response.status_code}"
                )
            )
            if attempt >= max_retries:
                break
            sleep_seconds = (
                retry_backoff_seconds
                * attempt
            )
            if sleep_seconds > 0:
                time.sleep(
                    sleep_seconds
                )
            continue
        # ====================================================
        # OTHER HTTP ERRORS
        # ====================================================
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise BinanceBasisError(
                "Binance HTTP error. "
                f"status={response.status_code} "
                f"payload={payload}"
            ) from exc
        # ====================================================
        # SUCCESS
        # ====================================================
        return payload
    raise BinanceBasisError(
        "Binance basis request failed "
        f"after {max_retries} attempts: "
        f"{last_error}"
    )
def fetch_basis_history(
    pair: str = "BTCUSDT",
    contract_type: str = "PERPETUAL",
    period: str = "5m",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch Binance USDⓈ-M Futures basis history.
    Canonical output
    ----------------
    exchange
    market
    symbol
    contract_type
    period
    timestamp
    index_price
    futures_price
    basis
    basis_rate
    annualized_basis_rate
    """
    pair = pair.upper()
    contract_type = (
        contract_type.upper()
    )
    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )
    if limit > MAX_BASIS_LIMIT:
        raise ValueError(
            f"limit cannot exceed "
            f"{MAX_BASIS_LIMIT}"
        )
    params: dict[str, Any] = {
        "pair": pair,
        "contractType": contract_type,
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
        BASIS_ENDPOINT,
        params,
    )
    # ========================================================
    # RESPONSE VALIDATION
    # ========================================================
    if isinstance(
        payload,
        dict,
    ):
        raise BinanceBasisError(
            "Unexpected Binance basis object response: "
            f"{payload}"
        )
    if not isinstance(
        payload,
        list,
    ):
        raise BinanceBasisError(
            "Unexpected Binance basis response type: "
            f"{type(payload).__name__}"
        )
    columns = [
        "exchange",
        "market",
        "symbol",
        "contract_type",
        "period",
        "timestamp",
        "index_price",
        "futures_price",
        "basis",
        "basis_rate",
        "annualized_basis_rate",
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
            raise BinanceBasisError(
                "Basis response contains "
                "non-object row"
            )
        required = {
            "pair",
            "contractType",
            "indexPrice",
            "futuresPrice",
            "basis",
            "basisRate",
            "timestamp",
        }
        missing = (
            required
            - set(item)
        )
        if missing:
            raise BinanceBasisError(
                "Missing basis fields: "
                f"{sorted(missing)}"
            )
        timestamp = pd.to_datetime(
            item["timestamp"],
            unit="ms",
            utc=True,
        )
        rows.append(
            {
                "exchange": "binance",
                "market": (
                    "usdm_perpetual"
                ),
                "symbol": str(
                    item["pair"]
                ).upper(),
                "contract_type": str(
                    item[
                        "contractType"
                    ]
                ).upper(),
                "period": period,
                "timestamp": timestamp,
                "index_price": float(
                    item[
                        "indexPrice"
                    ]
                ),
                "futures_price": float(
                    item[
                        "futuresPrice"
                    ]
                ),
                "basis": float(
                    item["basis"]
                ),
                "basis_rate": float(
                    item[
                        "basisRate"
                    ]
                ),
                "annualized_basis_rate": (
                    _optional_float(
                        item.get(
                            "annualizedBasisRate"
                        )
                    )
                ),
            }
        )
    result = pd.DataFrame(
        rows,
        columns=columns,
    )
    # ========================================================
    # RESPONSE CONSISTENCY
    # ========================================================
    if (
        result["symbol"]
        .ne(pair)
        .any()
    ):
        raise BinanceBasisError(
            "Returned basis pair does not "
            f"match requested pair={pair}"
        )
    if (
        result["contract_type"]
        .ne(contract_type)
        .any()
    ):
        raise BinanceBasisError(
            "Returned contract type does not "
            "match requested contract type"
        )
    return (
        result
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=[
                "timestamp",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )