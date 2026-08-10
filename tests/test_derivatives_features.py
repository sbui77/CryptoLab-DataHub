from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.derivatives import (
    aggregate_liquidations_5m,
    build_derivatives_features,
)


def make_oi(
    rows: int = 300,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "exchange": [
                "binance",
            ] * rows,

            "market": [
                "usdm_perpetual",
            ] * rows,

            "symbol": [
                "BTCUSDT",
            ] * rows,

            "period": [
                "5m",
            ] * rows,

            "timestamp": timestamp,

            "open_interest_base": (
                np.arange(
                    rows,
                    dtype=float,
                )
                + 100_000.0
            ),

            "open_interest_quote": (
                np.arange(
                    rows,
                    dtype=float,
                )
                * 1_000_000.0
                + 6_000_000_000.0
            ),
        }
    )


def make_basis(
    rows: int = 300,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    basis_rate = (
        np.sin(
            np.arange(rows)
            / 20.0
        )
        * 0.0001
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "basis": (
                basis_rate
                * 60_000.0
            ),
            "basis_rate": basis_rate,
            "annualized_basis_rate": [
                np.nan
            ] * rows,
        }
    )


def make_taker_flow(
    rows: int = 300,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "buy_volume": [
                120.0
            ] * rows,
            "sell_volume": [
                100.0
            ] * rows,
            "buy_sell_ratio": [
                1.2
            ] * rows,
        }
    )


def make_funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                [
                    "2026-08-01T00:00:00Z",
                    "2026-08-01T08:00:00Z",
                    "2026-08-01T16:00:00Z",
                ],
                utc=True,
            ),

            "funding_rate": [
                0.0001,
                0.0002,
                -0.0001,
            ],

            "mark_price": [
                60000.0,
                60100.0,
                60200.0,
            ],

            "rate_type": [
                "Regular",
                "Regular",
                "Regular",
            ],
        }
    )


def make_liquidations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-08-01T00:01:00Z",
                    "2026-08-01T00:02:00Z",
                ],
                utc=True,
            ),

            "liquidation_side": [
                "long",
                "short",
            ],

            "liquidation_notional": [
                100_000.0,
                40_000.0,
            ],
        }
    )


def test_liquidation_aggregation():
    result = aggregate_liquidations_5m(
        make_liquidations()
    )

    first = result.iloc[0]

    assert (
        first[
            "long_liquidation_count"
        ]
        == 1
    )

    assert (
        first[
            "short_liquidation_count"
        ]
        == 1
    )

    assert np.isclose(
        first[
            "total_liquidation_notional"
        ],
        140_000.0,
    )


def test_liquidation_delta_sign():
    result = aggregate_liquidations_5m(
        make_liquidations()
    )

    first = result.iloc[0]

    assert np.isclose(
        first[
            "liquidation_delta"
        ],
        -60_000.0,
    )


def test_build_derivatives_features():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    assert len(result) == 300

    assert (
        result["timestamp"]
        .is_monotonic_increasing
    )


def test_required_feature_columns():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    required = [
        "oi_quote_change_pct",
        "oi_quote_change_pct_1h",
        "oi_quote_change_pct_4h",
        "oi_quote_change_pct_24h",
        "oi_quote_zscore_24h",

        "funding_rate",
        "funding_rate_bps",
        "hours_since_funding",

        "basis_bps",
        "basis_zscore_24h",

        "futures_taker_delta",
        "futures_taker_delta_pct",
        "futures_taker_delta_1h",
        "futures_taker_delta_pct_1h",

        "long_liquidation_notional",
        "short_liquidation_notional",
        "liquidation_delta",
        "liquidation_imbalance",
    ]

    for column in required:
        assert column in result.columns


def test_futures_taker_identity():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    assert np.allclose(
        result[
            "futures_taker_delta"
        ],
        result[
            "futures_buy_volume"
        ]
        - result[
            "futures_sell_volume"
        ],
    )


def test_futures_ratio_identity():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    assert np.allclose(
        result[
            "futures_buy_ratio"
        ]
        + result[
            "futures_sell_ratio"
        ],
        1.0,
    )


def test_funding_asof_is_causal():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    row = result[
        result["timestamp"]
        == pd.Timestamp(
            "2026-08-01T07:55:00Z"
        )
    ].iloc[0]

    assert np.isclose(
        row["funding_rate"],
        0.0001,
    )

    assert (
        row["funding_time"]
        == pd.Timestamp(
            "2026-08-01T00:00:00Z"
        )
    )


def test_no_duplicate_timestamps():
    result = build_derivatives_features(
        open_interest=make_oi(),
        funding_rate=make_funding(),
        basis=make_basis(),
        taker_flow=make_taker_flow(),
        liquidations=make_liquidations(),
    )

    assert (
        result["timestamp"]
        .duplicated()
        .sum()
        == 0
    )
