from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from cryptolab.sources.binance_futures_http import (
    BinanceFuturesHTTPClient,
    BinanceFuturesIPBanError,
    BinanceFuturesRateLimitError,
    BinanceFuturesResponseError,
    BinanceFuturesRetryError,
)


def make_response(
    status_code: int,
    payload,
    headers: dict[str, str] | None = None,
):
    response = Mock()

    response.status_code = (
        status_code
    )

    response.headers = (
        headers
        if headers is not None
        else {}
    )

    if isinstance(
        payload,
        Exception,
    ):
        response.json.side_effect = (
            payload
        )

    else:
        response.json.return_value = (
            payload
        )

    return response


def make_client(
    responses,
    max_retries: int = 5,
):
    session = Mock()

    session.get.side_effect = (
        responses
    )

    sleeps: list[float] = []

    client = BinanceFuturesHTTPClient(
        base_url=(
            "https://fapi.binance.com"
        ),
        timeout_seconds=30,
        max_retries=max_retries,
        backoff_seconds=1.0,
        max_backoff_seconds=60.0,
        jitter_seconds=0.0,
        session=session,
        sleep_fn=sleeps.append,
        random_fn=lambda: 0.0,
    )

    return (
        client,
        session,
        sleeps,
    )


def test_success_returns_json():
    response = make_response(
        200,
        [
            {
                "symbol": "BTCUSDT",
            }
        ],
    )

    client, session, sleeps = (
        make_client(
            [
                response,
            ]
        )
    )

    payload = client.get_json(
        "/fapi/v1/test",
        params={
            "symbol": "BTCUSDT",
        },
    )

    assert payload == [
        {
            "symbol": "BTCUSDT",
        }
    ]

    assert (
        session.get.call_count
        == 1
    )

    assert sleeps == []


def test_endpoint_normalization():
    response = make_response(
        200,
        [],
    )

    client, session, _ = (
        make_client(
            [
                response,
            ]
        )
    )

    client.get_json(
        "fapi/v1/test"
    )

    args, kwargs = (
        session.get.call_args
    )

    assert args[0] == (
        "https://fapi.binance.com"
        "/fapi/v1/test"
    )


def test_retry_transport_error():
    good = make_response(
        200,
        [],
    )

    client, session, sleeps = (
        make_client(
            [
                requests.Timeout(
                    "timeout"
                ),
                good,
            ],
            max_retries=2,
        )
    )

    result = client.get_json(
        "/fapi/v1/test"
    )

    assert result == []

    assert (
        session.get.call_count
        == 2
    )

    assert sleeps == [
        1.0,
    ]


def test_transport_exhaustion():
    client, _, _ = (
        make_client(
            [
                requests.Timeout(
                    "timeout"
                ),
                requests.Timeout(
                    "timeout"
                ),
            ],
            max_retries=2,
        )
    )

    with pytest.raises(
        BinanceFuturesRetryError
    ):
        client.get_json(
            "/fapi/v1/test"
        )


def test_429_uses_retry_after():
    limited = make_response(
        429,
        {
            "code": -1003,
            "msg": "Too many requests",
        },
        headers={
            "Retry-After": "7",
        },
    )

    good = make_response(
        200,
        [],
    )

    client, session, sleeps = (
        make_client(
            [
                limited,
                good,
            ],
            max_retries=2,
        )
    )

    result = client.get_json(
        "/fapi/v1/test"
    )

    assert result == []

    assert (
        session.get.call_count
        == 2
    )

    assert sleeps == [
        7.0,
    ]


def test_429_falls_back_to_exponential_backoff():
    limited = make_response(
        429,
        {
            "code": -1003,
            "msg": "Too many requests",
        },
    )

    good = make_response(
        200,
        [],
    )

    client, _, sleeps = (
        make_client(
            [
                limited,
                good,
            ],
            max_retries=2,
        )
    )

    client.get_json(
        "/fapi/v1/test"
    )

    assert sleeps == [
        1.0,
    ]


def test_rate_limit_exhaustion():
    limited_1 = make_response(
        429,
        {
            "code": -1003,
            "msg": "Too many requests",
        },
    )

    limited_2 = make_response(
        429,
        {
            "code": -1003,
            "msg": "Too many requests",
        },
    )

    client, _, sleeps = (
        make_client(
            [
                limited_1,
                limited_2,
            ],
            max_retries=2,
        )
    )

    with pytest.raises(
        BinanceFuturesRateLimitError
    ):
        client.get_json(
            "/fapi/v1/test"
        )

    assert sleeps == [
        1.0,
    ]


def test_http_418_fails_immediately():
    banned = make_response(
        418,
        {
            "code": -1003,
            "msg": "IP banned",
        },
        headers={
            "Retry-After": "120",
        },
    )

    client, session, sleeps = (
        make_client(
            [
                banned,
            ]
        )
    )

    with pytest.raises(
        BinanceFuturesIPBanError
    ):
        client.get_json(
            "/fapi/v1/test"
        )

    assert (
        session.get.call_count
        == 1
    )

    assert sleeps == []


def test_http_500_retries():
    failed = make_response(
        500,
        {
            "code": -1000,
            "msg": "Internal error",
        },
    )

    good = make_response(
        200,
        [
            1,
            2,
            3,
        ],
    )

    client, _, sleeps = (
        make_client(
            [
                failed,
                good,
            ],
            max_retries=2,
        )
    )

    result = client.get_json(
        "/fapi/v1/test"
    )

    assert result == [
        1,
        2,
        3,
    ]

    assert sleeps == [
        1.0,
    ]


def test_non_retryable_400_fails_immediately():
    failed = make_response(
        400,
        {
            "code": -1102,
            "msg": (
                "Mandatory parameter "
                "was not sent"
            ),
        },
    )

    client, session, sleeps = (
        make_client(
            [
                failed,
            ]
        )
    )

    with pytest.raises(
        BinanceFuturesResponseError
    ):
        client.get_json(
            "/fapi/v1/test"
        )

    assert (
        session.get.call_count
        == 1
    )

    assert sleeps == []


def test_200_binance_minus_1003_retries():
    limited = make_response(
        200,
        {
            "code": -1003,
            "msg": "Too many requests",
        },
    )

    good = make_response(
        200,
        [],
    )

    client, _, sleeps = (
        make_client(
            [
                limited,
                good,
            ],
            max_retries=2,
        )
    )

    result = client.get_json(
        "/fapi/v1/test"
    )

    assert result == []

    assert sleeps == [
        1.0,
    ]


def test_200_non_retryable_binance_error():
    failed = make_response(
        200,
        {
            "code": -1102,
            "msg": "Missing parameter",
        },
    )

    client, _, sleeps = (
        make_client(
            [
                failed,
            ]
        )
    )

    with pytest.raises(
        BinanceFuturesResponseError
    ):
        client.get_json(
            "/fapi/v1/test"
        )

    assert sleeps == []


def test_successful_non_json_fails():
    failed = make_response(
        200,
        ValueError(
            "not json"
        ),
    )

    client, _, _ = (
        make_client(
            [
                failed,
            ]
        )
    )

    with pytest.raises(
        BinanceFuturesResponseError
    ):
        client.get_json(
            "/fapi/v1/test"
        )


def test_request_weight_metadata():
    response = make_response(
        200,
        [],
        headers={
            "X-MBX-USED-WEIGHT-1M":
                "123",
        },
    )

    client, _, _ = (
        make_client(
            [
                response,
            ]
        )
    )

    client.get_json(
        "/fapi/v1/test"
    )

    assert (
        client.last_metadata
        is not None
    )

    assert (
        client
        .last_metadata
        .used_weight_1m
        == 123
    )


def test_exponential_backoff():
    failure_1 = make_response(
        500,
        {},
    )

    failure_2 = make_response(
        500,
        {},
    )

    good = make_response(
        200,
        [],
    )

    client, _, sleeps = (
        make_client(
            [
                failure_1,
                failure_2,
                good,
            ],
            max_retries=3,
        )
    )

    client.get_json(
        "/fapi/v1/test"
    )

    assert sleeps == [
        1.0,
        2.0,
    ]
