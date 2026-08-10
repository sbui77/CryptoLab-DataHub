from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Callable

import requests


DEFAULT_BASE_URL = (
    "https://fapi.binance.com"
)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 5

DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
DEFAULT_JITTER_SECONDS = 0.25

RETRYABLE_HTTP_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

RATE_LIMIT_HTTP_STATUS_CODES = {
    418,
    429,
}

RATE_LIMIT_BINANCE_CODES = {
    -1003,
}

RETRYABLE_BINANCE_CODES = {
    -1000,
    -1001,
    -1006,
    -1007,
    -1008,
}


class BinanceFuturesHTTPError(RuntimeError):
    """
    Base error for Binance Futures HTTP operations.
    """


class BinanceFuturesResponseError(
    BinanceFuturesHTTPError
):
    """
    Raised for a non-retryable Binance/API response.
    """


class BinanceFuturesRateLimitError(
    BinanceFuturesHTTPError
):
    """
    Raised when request rate-limit retries are exhausted.
    """


class BinanceFuturesIPBanError(
    BinanceFuturesHTTPError
):
    """
    Raised immediately when Binance reports HTTP 418.
    """


class BinanceFuturesRetryError(
    BinanceFuturesHTTPError
):
    """
    Raised after retryable failures exhaust max attempts.
    """


@dataclass(frozen=True)
class BinanceFuturesErrorPayload:
    """
    Normalized Binance error payload.
    """

    http_status: int
    code: int | None
    message: str | None
    retry_after_seconds: float | None


@dataclass(frozen=True)
class BinanceFuturesRequestMetadata:
    """
    Metadata from the most recent successful request.
    """

    status_code: int

    used_weight_1m: int | None

    request_count_10s: int | None
    request_count_1m: int | None


def _safe_json(
    response: requests.Response,
) -> Any:
    """
    Decode JSON without allowing a JSON error to mask
    HTTP-level handling.
    """

    try:
        return response.json()

    except ValueError:
        return None


def _extract_binance_error(
    payload: Any,
) -> tuple[
    int | None,
    str | None,
]:
    """
    Extract Binance:
        code
        msg

    from a JSON error response.
    """

    if not isinstance(
        payload,
        dict,
    ):
        return (
            None,
            None,
        )

    raw_code = payload.get(
        "code"
    )

    raw_message = payload.get(
        "msg"
    )

    code: int | None = None

    if raw_code is not None:
        try:
            code = int(
                raw_code
            )

        except (
            TypeError,
            ValueError,
        ):
            code = None

    message = (
        str(raw_message)
        if raw_message is not None
        else None
    )

    return (
        code,
        message,
    )


def _parse_float(
    value: str | None,
) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if parsed < 0:
        return None

    return parsed


def _parse_int(
    value: str | None,
) -> int | None:
    if value is None:
        return None

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _retry_after_seconds(
    response: requests.Response,
) -> float | None:
    """
    Read Retry-After when Binance provides it.

    We intentionally support numeric seconds only.

    If the header is absent or malformed, caller falls back
    to exponential backoff.
    """

    return _parse_float(
        response.headers.get(
            "Retry-After"
        )
    )


def _build_error_payload(
    response: requests.Response,
    payload: Any,
) -> BinanceFuturesErrorPayload:
    code, message = (
        _extract_binance_error(
            payload
        )
    )

    return BinanceFuturesErrorPayload(
        http_status=int(
            response.status_code
        ),
        code=code,
        message=message,
        retry_after_seconds=(
            _retry_after_seconds(
                response
            )
        ),
    )


def _error_text(
    error: BinanceFuturesErrorPayload,
) -> str:
    parts = [
        (
            f"status="
            f"{error.http_status}"
        ),
    ]

    if error.code is not None:
        parts.append(
            f"code={error.code}"
        )

    if error.message:
        parts.append(
            f"msg={error.message}"
        )

    if (
        error.retry_after_seconds
        is not None
    ):
        parts.append(
            "retry_after="
            f"{error.retry_after_seconds}"
        )

    return " ".join(
        parts
    )


def _request_metadata(
    response: requests.Response,
) -> BinanceFuturesRequestMetadata:
    """
    Capture Binance rate-limit telemetry when present.

    Missing headers are normal and return None.
    """

    return BinanceFuturesRequestMetadata(
        status_code=int(
            response.status_code
        ),

        used_weight_1m=_parse_int(
            response.headers.get(
                "X-MBX-USED-WEIGHT-1M"
            )
        ),

        request_count_10s=_parse_int(
            response.headers.get(
                "X-MBX-ORDER-COUNT-10S"
            )
        ),

        request_count_1m=_parse_int(
            response.headers.get(
                "X-MBX-ORDER-COUNT-1M"
            )
        ),
    )


class BinanceFuturesHTTPClient:
    """
    Shared HTTP client for Binance USD-M Futures public REST.

    Responsibilities
    ----------------
    - persistent requests.Session
    - JSON decoding
    - timeout handling
    - exponential backoff
    - jitter
    - HTTP 429 handling
    - Binance code -1003 handling
    - HTTP 418 immediate IP-ban failure
    - retryable HTTP 5xx handling
    - retryable Binance infrastructure error handling
    - request-weight telemetry

    This client does not know endpoint-specific schemas.
    Source modules remain responsible for validating their
    own payload shapes.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = (
            DEFAULT_TIMEOUT_SECONDS
        ),
        max_retries: int = (
            DEFAULT_MAX_RETRIES
        ),
        backoff_seconds: float = (
            DEFAULT_BACKOFF_SECONDS
        ),
        max_backoff_seconds: float = (
            DEFAULT_MAX_BACKOFF_SECONDS
        ),
        jitter_seconds: float = (
            DEFAULT_JITTER_SECONDS
        ),
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        if not base_url:
            raise ValueError(
                "base_url cannot be empty"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive"
            )

        if max_retries <= 0:
            raise ValueError(
                "max_retries must be positive"
            )

        if backoff_seconds < 0:
            raise ValueError(
                "backoff_seconds cannot be negative"
            )

        if max_backoff_seconds < 0:
            raise ValueError(
                "max_backoff_seconds cannot be negative"
            )

        if jitter_seconds < 0:
            raise ValueError(
                "jitter_seconds cannot be negative"
            )

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout_seconds = float(
            timeout_seconds
        )

        self.max_retries = int(
            max_retries
        )

        self.backoff_seconds = float(
            backoff_seconds
        )

        self.max_backoff_seconds = float(
            max_backoff_seconds
        )

        self.jitter_seconds = float(
            jitter_seconds
        )

        self.session = (
            session
            if session is not None
            else requests.Session()
        )

        self.sleep_fn = sleep_fn
        self.random_fn = random_fn

        self.last_metadata: (
            BinanceFuturesRequestMetadata
            | None
        ) = None

    def _url(
        self,
        endpoint: str,
    ) -> str:
        if not endpoint:
            raise ValueError(
                "endpoint cannot be empty"
            )

        if not endpoint.startswith(
            "/"
        ):
            endpoint = (
                "/"
                + endpoint
            )

        return (
            self.base_url
            + endpoint
        )

    def _exponential_delay(
        self,
        attempt: int,
    ) -> float:
        """
        attempt:
            1-based request attempt number.
        """

        base = (
            self.backoff_seconds
            * (
                2 ** (
                    attempt - 1
                )
            )
        )

        base = min(
            base,
            self.max_backoff_seconds,
        )

        jitter = (
            self.random_fn()
            * self.jitter_seconds
        )

        return (
            base
            + jitter
        )

    def _retry_delay(
        self,
        attempt: int,
        response: requests.Response | None,
    ) -> float:
        """
        Prefer Retry-After for rate-limit responses.

        Otherwise use exponential backoff + jitter.
        """

        if response is not None:
            retry_after = (
                _retry_after_seconds(
                    response
                )
            )

            if retry_after is not None:
                return retry_after

        return self._exponential_delay(
            attempt
        )

    def get_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute one Binance Futures GET request.

        Returns decoded JSON for successful 2xx responses.

        Non-retryable 4xx responses fail immediately.

        HTTP 429 / Binance -1003:
            retry with Retry-After when available,
            otherwise exponential backoff.

        HTTP 418:
            fail immediately as explicit IP ban.

        HTTP 5xx / transport failures:
            retry until max_retries.
        """

        url = self._url(
            endpoint
        )

        request_params = (
            dict(params)
            if params is not None
            else {}
        )

        last_error_text: (
            str | None
        ) = None

        last_was_rate_limit = False

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            response: (
                requests.Response | None
            ) = None

            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout_seconds,
                )

            except requests.RequestException as exc:
                last_error_text = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if (
                    attempt
                    >= self.max_retries
                ):
                    raise BinanceFuturesRetryError(
                        "Binance Futures transport "
                        "request failed after "
                        f"{self.max_retries} attempts: "
                        f"{last_error_text}"
                    ) from exc

                delay = self._retry_delay(
                    attempt,
                    None,
                )

                self.sleep_fn(
                    delay
                )

                continue

            payload = _safe_json(
                response
            )

            self.last_metadata = (
                _request_metadata(
                    response
                )
            )

            # =================================================
            # SUCCESS
            # =================================================

            if (
                200
                <= response.status_code
                < 300
            ):
                # Binance may exceptionally encode an error
                # object in a successful HTTP envelope.
                code, message = (
                    _extract_binance_error(
                        payload
                    )
                )

                if (
                    code is not None
                    and code < 0
                ):
                    error = (
                        _build_error_payload(
                            response,
                            payload,
                        )
                    )

                    if (
                        code
                        in RATE_LIMIT_BINANCE_CODES
                    ):
                        last_was_rate_limit = True

                        last_error_text = (
                            _error_text(
                                error
                            )
                        )

                        if (
                            attempt
                            >= self.max_retries
                        ):
                            raise (
                                BinanceFuturesRateLimitError(
                                    "Binance Futures rate "
                                    "limit exhausted: "
                                    f"{last_error_text}"
                                )
                            )

                        delay = (
                            self._retry_delay(
                                attempt,
                                response,
                            )
                        )

                        self.sleep_fn(
                            delay
                        )

                        continue

                    if (
                        code
                        in RETRYABLE_BINANCE_CODES
                    ):
                        last_error_text = (
                            _error_text(
                                error
                            )
                        )

                        if (
                            attempt
                            >= self.max_retries
                        ):
                            raise (
                                BinanceFuturesRetryError(
                                    "Binance Futures "
                                    "retryable API error "
                                    "exhausted: "
                                    f"{last_error_text}"
                                )
                            )

                        delay = (
                            self._retry_delay(
                                attempt,
                                response,
                            )
                        )

                        self.sleep_fn(
                            delay
                        )

                        continue

                    raise (
                        BinanceFuturesResponseError(
                            "Binance Futures API error: "
                            f"{_error_text(error)}"
                        )
                    )

                if payload is None:
                    raise (
                        BinanceFuturesResponseError(
                            "Binance Futures returned "
                            "non-JSON successful response. "
                            f"status={response.status_code}"
                        )
                    )

                return payload

            # =================================================
            # ERROR RESPONSE
            # =================================================

            error = (
                _build_error_payload(
                    response,
                    payload,
                )
            )

            error_text = (
                _error_text(
                    error
                )
            )

            last_error_text = (
                error_text
            )

            # -------------------------------------------------
            # Explicit IP ban.
            # Do NOT automatically hammer a banned IP.
            # -------------------------------------------------

            if (
                response.status_code
                == 418
            ):
                raise BinanceFuturesIPBanError(
                    "Binance Futures IP ban: "
                    f"{error_text}"
                )

            # -------------------------------------------------
            # Rate limit.
            # -------------------------------------------------

            rate_limited = (
                response.status_code
                == 429
                or error.code
                in RATE_LIMIT_BINANCE_CODES
            )

            if rate_limited:
                last_was_rate_limit = True

                if (
                    attempt
                    >= self.max_retries
                ):
                    raise (
                        BinanceFuturesRateLimitError(
                            "Binance Futures rate "
                            "limit exhausted: "
                            f"{error_text}"
                        )
                    )

                delay = self._retry_delay(
                    attempt,
                    response,
                )

                self.sleep_fn(
                    delay
                )

                continue

            # -------------------------------------------------
            # Retryable HTTP/backend failures.
            # -------------------------------------------------

            retryable_http = (
                response.status_code
                in RETRYABLE_HTTP_STATUS_CODES
            )

            retryable_code = (
                error.code
                in RETRYABLE_BINANCE_CODES
            )

            if (
                retryable_http
                or retryable_code
            ):
                if (
                    attempt
                    >= self.max_retries
                ):
                    raise (
                        BinanceFuturesRetryError(
                            "Binance Futures retry "
                            "exhausted: "
                            f"{error_text}"
                        )
                    )

                delay = self._retry_delay(
                    attempt,
                    response,
                )

                self.sleep_fn(
                    delay
                )

                continue

            # -------------------------------------------------
            # Other 4xx = caller/input problem.
            # -------------------------------------------------

            raise BinanceFuturesResponseError(
                "Binance Futures request failed: "
                f"{error_text}"
            )

        # Defensive fallback.
        if last_was_rate_limit:
            raise BinanceFuturesRateLimitError(
                "Binance Futures rate limit "
                "exhausted: "
                f"{last_error_text}"
            )

        raise BinanceFuturesRetryError(
            "Binance Futures request retries "
            "exhausted: "
            f"{last_error_text}"
        )
