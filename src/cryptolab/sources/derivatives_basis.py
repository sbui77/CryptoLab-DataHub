from __future__ import annotations

from typing import Any

import pandas as pd

from cryptolab.sources.binance_futures_http import (
    BinanceFuturesHTTPClient,
    BinanceFuturesHTTPError,
    BinanceFuturesIPBanError,
    BinanceFuturesRateLimitError,
)
from cryptolab.time_contract import (
    attach_runtime_time_metadata,
)


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
    Raised when the shared Binance Futures client reports
    an exhausted rate limit or IP ban.
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

    Rate-limit and IP-ban errors are mapped back into the
    source-specific BinanceBasisRateLimitError for backward
    compatibility.
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
        raise BinanceBasisRateLimitError(
            "Binance basis rate-limit/IP-ban error: "
            f"{exc}"
        ) from exc

    except BinanceFuturesHTTPError as exc:
        raise BinanceBasisError(
            "Binance basis request failed: "
            f"{exc}"
        ) from exc


def _empty_basis_frame() -> pd.DataFrame:
    """
    Return empty Basis DataFrame with complete
    runtime point-in-time schema.
    """

    result = pd.DataFrame(
        columns=[
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
            "available_at",
            "available_at_quality",
            "ingested_at",
            "ingested_at_quality",
        ]
    )

    result["timestamp"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["available_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    result["ingested_at"] = pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    return result


def fetch_basis_history(
    pair: str = "BTCUSDT",
    contract_type: str = "PERPETUAL",
    period: str = "5m",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch Binance USD-M Futures basis history.

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
    available_at
    available_at_quality
    ingested_at
    ingested_at_quality

    Point-in-time semantics
    -----------------------
    available_at
        Derived from Binance timestamp.

    available_at_quality
        derived

    ingested_at
        Actual CryptoLab runtime timestamp for the
        current REST response.

    ingested_at_quality
        exact
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

    if not payload:
        return _empty_basis_frame()

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
                "market": "usdm_perpetual",
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
        rows
    )

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

    result = attach_runtime_time_metadata(
        result,
        event_time_column="timestamp",
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
