from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from cryptolab.pipelines.basis_update import (
    update_basis,
)
from cryptolab.pipelines.funding_rate_update import (
    update_funding_rate,
)
from cryptolab.pipelines.open_interest_update import (
    update_open_interest,
)
from cryptolab.pipelines.taker_flow_update import (
    update_taker_flow,
)


class DerivativesUpdateError(RuntimeError):
    """
    Raised when derivatives orchestration itself fails.

    Individual source failures are normally captured into
    DerivativesStreamUpdateResult rather than raised.
    """


@dataclass(frozen=True)
class DerivativesStreamUpdateResult:
    """
    Normalized result for one derivatives source update.
    """

    name: str
    success: bool

    rows_fetched: int | None
    up_to_date: bool | None

    first_timestamp: str | None
    last_timestamp: str | None

    error_type: str | None
    error_message: str | None

    raw_result: dict[str, Any] | None


@dataclass(frozen=True)
class DerivativesUpdateResult:
    """
    Summary of one complete derivatives REST update run.
    """

    exchange: str
    symbol: str
    period: str

    started_at: datetime
    finished_at: datetime

    open_interest: DerivativesStreamUpdateResult
    funding_rate: DerivativesStreamUpdateResult
    basis: DerivativesStreamUpdateResult
    taker_flow: DerivativesStreamUpdateResult

    success: bool
    failure_count: int
    failures: tuple[str, ...]


def _timestamp_to_string(
    value: Any,
) -> str | None:
    """
    Normalize a timestamp-like result field for compact
    orchestration reporting.

    None remains None.
    """

    if value is None:
        return None

    try:
        return str(value)

    except Exception:
        return repr(value)


def _extract_timestamp(
    result_dict: dict[str, Any],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in result_dict:
            value = result_dict[
                candidate
            ]

            if value is not None:
                return _timestamp_to_string(
                    value
                )

    return None


def _normalize_result(
    name: str,
    result: Any,
) -> DerivativesStreamUpdateResult:
    """
    Convert updater-specific dataclass into a common result.

    Supported updater result differences
    ------------------------------------
    OI:
        first_timestamp
        last_timestamp

    Funding:
        first_funding_time
        last_funding_time

    Basis / Taker:
        first_timestamp
        last_timestamp
    """

    if hasattr(
        result,
        "__dataclass_fields__",
    ):
        raw = asdict(
            result
        )

    elif isinstance(
        result,
        dict,
    ):
        raw = dict(
            result
        )

    else:
        raise DerivativesUpdateError(
            f"{name} updater returned unsupported "
            f"result type: {type(result).__name__}"
        )

    rows_fetched_raw = raw.get(
        "rows_fetched"
    )

    rows_fetched = (
        int(rows_fetched_raw)
        if rows_fetched_raw
        is not None
        else None
    )

    up_to_date_raw = raw.get(
        "up_to_date"
    )

    up_to_date = (
        bool(up_to_date_raw)
        if up_to_date_raw
        is not None
        else None
    )

    first_timestamp = _extract_timestamp(
        raw,
        (
            "first_timestamp",
            "first_funding_time",
        ),
    )

    last_timestamp = _extract_timestamp(
        raw,
        (
            "last_timestamp",
            "last_funding_time",
        ),
    )

    return DerivativesStreamUpdateResult(
        name=name,
        success=True,
        rows_fetched=rows_fetched,
        up_to_date=up_to_date,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        error_type=None,
        error_message=None,
        raw_result=raw,
    )


def _failed_result(
    name: str,
    exc: Exception,
) -> DerivativesStreamUpdateResult:
    return DerivativesStreamUpdateResult(
        name=name,
        success=False,
        rows_fetched=None,
        up_to_date=None,
        first_timestamp=None,
        last_timestamp=None,
        error_type=type(
            exc
        ).__name__,
        error_message=str(
            exc
        ),
        raw_result=None,
    )


def _run_stream(
    name: str,
    function: Callable[..., Any],
    kwargs: dict[str, Any],
) -> DerivativesStreamUpdateResult:
    """
    Run one updater and capture failure without aborting
    the whole derivatives update.
    """

    try:
        result = function(
            **kwargs
        )

        return _normalize_result(
            name=name,
            result=result,
        )

    except Exception as exc:
        return _failed_result(
            name=name,
            exc=exc,
        )


def update_derivatives(
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    period: str = "5m",
    continue_on_error: bool = True,
) -> DerivativesUpdateResult:
    """
    Run all incremental Binance derivatives REST updaters.

    Order
    -----
    1. Open Interest
    2. Funding Rate
    3. Premium / Basis
    4. Futures Taker Flow

    Failure semantics
    -----------------
    continue_on_error=True:
        complete all four streams and return a full report.

    continue_on_error=False:
        stop after the first failure, but still return a
        structured result for every stream. Unrun streams are
        marked failed with error_type='Skipped'.
    """

    exchange = exchange.lower()
    symbol = symbol.upper()

    if exchange != "binance":
        raise ValueError(
            "Derivatives updater currently supports "
            "Binance only"
        )

    if not period:
        raise ValueError(
            "period cannot be empty"
        )

    started_at = datetime.now(
        timezone.utc
    )

    stream_specs: list[
        tuple[
            str,
            Callable[..., Any],
            dict[str, Any],
        ]
    ] = [
        (
            "open_interest",
            update_open_interest,
            {
                "exchange": exchange,
                "symbol": symbol,
                "period": period,
            },
        ),
        (
            "funding_rate",
            update_funding_rate,
            {
                "exchange": exchange,
                "symbol": symbol,
            },
        ),
        (
            "basis",
            update_basis,
            {
                "exchange": exchange,
                "symbol": symbol,
                "contract_type": "PERPETUAL",
                "period": period,
            },
        ),
        (
            "taker_flow",
            update_taker_flow,
            {
                "exchange": exchange,
                "symbol": symbol,
                "period": period,
            },
        ),
    ]

    results: dict[
        str,
        DerivativesStreamUpdateResult,
    ] = {}

    stop = False

    for (
        name,
        function,
        kwargs,
    ) in stream_specs:

        if stop:
            results[
                name
            ] = DerivativesStreamUpdateResult(
                name=name,
                success=False,
                rows_fetched=None,
                up_to_date=None,
                first_timestamp=None,
                last_timestamp=None,
                error_type="Skipped",
                error_message=(
                    "Skipped because an earlier stream "
                    "failed and continue_on_error=False"
                ),
                raw_result=None,
            )

            continue

        stream_result = _run_stream(
            name=name,
            function=function,
            kwargs=kwargs,
        )

        results[
            name
        ] = stream_result

        if (
            not stream_result.success
            and not continue_on_error
        ):
            stop = True

    finished_at = datetime.now(
        timezone.utc
    )

    failures = tuple(
        name
        for name, result
        in results.items()
        if not result.success
    )

    return DerivativesUpdateResult(
        exchange=exchange,
        symbol=symbol,
        period=period,
        started_at=started_at,
        finished_at=finished_at,
        open_interest=results[
            "open_interest"
        ],
        funding_rate=results[
            "funding_rate"
        ],
        basis=results[
            "basis"
        ],
        taker_flow=results[
            "taker_flow"
        ],
        success=(
            len(failures)
            == 0
        ),
        failure_count=len(
            failures
        ),
        failures=failures,
    )
